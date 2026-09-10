"""Small model-agnostic helpers for direct Earth Engine GeoTIFF downloads."""

from __future__ import annotations

import random
import time
import urllib.request
from urllib.error import HTTPError, URLError

import ee
from rasterio.io import MemoryFile


OUTPUT_NODATA = -9999.0
DIRECT_DOWNLOAD_MAX_ATTEMPTS = 6
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _retry_delay_seconds(attempt: int) -> float:
    base = min(30.0, 2.0 ** attempt)
    return base + random.uniform(0.0, min(1.0, base * 0.25))


def download_ee_bytes(image: ee.Image, parameters: dict, timeout_seconds: int) -> bytes:
    """Download an Earth Engine image with bounded transient-error retries."""
    for attempt in range(1, DIRECT_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            url = image.getDownloadURL(parameters)
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                return response.read()
        except HTTPError as error:
            try:
                body = error.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            transient = int(error.code) in TRANSIENT_HTTP_STATUS_CODES
            if transient and attempt < DIRECT_DOWNLOAD_MAX_ATTEMPTS:
                delay = _retry_delay_seconds(attempt)
                print(
                    f"Transient Earth Engine HTTP {error.code}; retry "
                    f"{attempt}/{DIRECT_DOWNLOAD_MAX_ATTEMPTS} in {delay:.1f} s..."
                )
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"Earth Engine direct download failed (HTTP {error.code}): "
                f"{body or error.reason}"
            ) from error
        except (URLError, TimeoutError) as error:
            if attempt < DIRECT_DOWNLOAD_MAX_ATTEMPTS:
                delay = _retry_delay_seconds(attempt)
                print(
                    "Transient network error during Earth Engine download; "
                    f"retry {attempt}/{DIRECT_DOWNLOAD_MAX_ATTEMPTS} "
                    f"in {delay:.1f} s: {error}"
                )
                time.sleep(delay)
                continue
            raise RuntimeError(
                "Earth Engine direct download failed after repeated network errors: "
                f"{error}"
            ) from error
    raise RuntimeError("Earth Engine direct download attempts exhausted.")


def read_downloaded_array(payload: bytes):
    """Read a downloaded in-memory GeoTIFF as masked raster bands."""
    with MemoryFile(payload) as memory_file:
        with memory_file.open() as source:
            return source.read(masked=True), source.transform, source.crs
