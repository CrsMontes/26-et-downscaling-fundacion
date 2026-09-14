from http.client import IncompleteRead
from unittest.mock import MagicMock, Mock, call

import pytest

from et_downscaling import ee_download


@pytest.mark.parametrize("success_attempt", [2, 6, None])
def test_incomplete_read_retries_and_preserves_download_or_final_error(monkeypatch, success_attempt):
    payload = b"complete GeoTIFF response"
    parameters = {"format": "GEO_TIFF", "bands": ["ET_mm_period"]}
    attempts = success_attempt or ee_download.DIRECT_DOWNLOAD_MAX_ATTEMPTS
    failures = attempts - 1 if success_attempt else attempts
    errors = [IncompleteRead(b"partial response", len(payload)) for _ in range(failures)]
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.side_effect = errors + ([payload] if success_attempt else [])
    urlopen = Mock(return_value=response)
    image = Mock()
    image.getDownloadURL.return_value = "https://example.invalid/download"
    sleep = Mock()
    monkeypatch.setattr(ee_download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ee_download.time, "sleep", sleep)
    monkeypatch.setattr(ee_download.random, "uniform", lambda low, high: 0.0)

    if success_attempt:
        assert ee_download.download_ee_bytes(image, parameters, 600) == payload
    else:
        with pytest.raises(RuntimeError, match="failed after repeated network errors") as caught:
            ee_download.download_ee_bytes(image, parameters, 600)
        assert caught.value.__cause__ is errors[-1]

    assert ee_download.DIRECT_DOWNLOAD_MAX_ATTEMPTS == 6
    assert image.getDownloadURL.call_args_list == [call(parameters)] * attempts
    assert urlopen.call_args_list == [call("https://example.invalid/download", timeout=600)] * attempts
    assert response.read.call_count == attempts
    assert response.__exit__.call_count == attempts
    assert sleep.call_args_list == [call(delay) for delay in [2.0, 4.0, 8.0, 16.0, 30.0][:attempts - 1]]



def test_remote_disconnected_is_retried(monkeypatch):
    from http.client import RemoteDisconnected

    import et_downscaling.ee_download as module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b"complete-payload"

    calls = {"count": 0}

    def fake_urlopen(url, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RemoteDisconnected(
                "Remote end closed connection without response"
            )
        return Response()

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        module,
        "_retry_delay_seconds",
        lambda attempt: 0.0,
    )

    class FakeImage:
        def getDownloadURL(self, parameters):
            return "https://example.test"

    result = module.download_ee_bytes(FakeImage(), {}, 30)

    assert result == b"complete-payload"
    assert calls["count"] == 2


def test_remote_disconnected_exhausts_retry_budget(monkeypatch):
    from http.client import RemoteDisconnected

    import pytest
    import et_downscaling.ee_download as module

    calls = {"count": 0}

    def fake_urlopen(url, timeout):
        calls["count"] += 1
        raise RemoteDisconnected(
            "Remote end closed connection without response"
        )

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        module,
        "_retry_delay_seconds",
        lambda attempt: 0.0,
    )

    class FakeImage:
        def getDownloadURL(self, parameters):
            return "https://example.test"

    with pytest.raises(RuntimeError) as exc_info:
        module.download_ee_bytes(FakeImage(), {}, 30)

    assert isinstance(
        exc_info.value.__cause__,
        RemoteDisconnected,
    )
    assert calls["count"] == module.DIRECT_DOWNLOAD_MAX_ATTEMPTS
