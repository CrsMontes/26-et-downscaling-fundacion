import hashlib

from et_downscaling.field_product_identity import (
    DOWNLOADER,
    downloader_scientific_hash,
    scientific_contract,
)


BASE_SOURCE = """\
from http.client import IncompleteRead

DIRECT_DOWNLOAD_MAX_ATTEMPTS = 6
TRANSIENT_HTTP_STATUS_CODES = {429, 500}
OUTPUT_NODATA = -9999.0

def _retry_delay_seconds(attempt):
    return attempt + 1

def download_ee_bytes(url):
    for attempt in range(DIRECT_DOWNLOAD_MAX_ATTEMPTS):
        try:
            return url.read()
        except IncompleteRead:
            continue
    raise RuntimeError("download failed")
"""


def _execution_contract(source):
    return {
        DOWNLOADER: hashlib.sha256(source.encode()).hexdigest(),
        "models/model": "model-hash",
        "models/aoa": "aoa-hash",
    }


def test_retry_only_change_does_not_change_scientific_identity():
    changed = BASE_SOURCE.replace(
        "DIRECT_DOWNLOAD_MAX_ATTEMPTS = 6",
        "DIRECT_DOWNLOAD_MAX_ATTEMPTS = 9",
    )

    assert hashlib.sha256(BASE_SOURCE.encode()).hexdigest() != hashlib.sha256(
        changed.encode()
    ).hexdigest()

    assert downloader_scientific_hash(
        BASE_SOURCE
    ) == downloader_scientific_hash(changed)

    assert scientific_contract(
        _execution_contract(BASE_SOURCE),
        BASE_SOURCE,
    ) == scientific_contract(
        _execution_contract(changed),
        changed,
    )


def test_payload_affecting_change_changes_scientific_identity():
    changed = BASE_SOURCE.replace(
        "OUTPUT_NODATA = -9999.0",
        "OUTPUT_NODATA = -9998.0",
    )

    assert downloader_scientific_hash(
        BASE_SOURCE
    ) != downloader_scientific_hash(changed)

    assert scientific_contract(
        _execution_contract(BASE_SOURCE),
        BASE_SOURCE,
    ) != scientific_contract(
        _execution_contract(changed),
        changed,
    )


def test_payload_read_change_changes_scientific_identity():
    changed = BASE_SOURCE.replace(
        "return url.read()",
        "return url.read(1024)",
    )

    assert downloader_scientific_hash(
        BASE_SOURCE
    ) != downloader_scientific_hash(changed)



def test_remote_disconnected_retry_is_transport_only():
    changed = BASE_SOURCE.replace(
        "from http.client import IncompleteRead",
        "from http.client import IncompleteRead, RemoteDisconnected",
    ).replace(
        "except IncompleteRead:",
        "except (IncompleteRead, RemoteDisconnected):",
    )

    assert downloader_scientific_hash(
        BASE_SOURCE
    ) == downloader_scientific_hash(changed)

    assert scientific_contract(
        _execution_contract(BASE_SOURCE),
        BASE_SOURCE,
    ) == scientific_contract(
        _execution_contract(changed),
        changed,
    )
