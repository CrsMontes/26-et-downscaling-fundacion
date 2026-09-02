import time
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import ee

from .config import OUTPUT_FILENAME
from .schema import SATELLITE_EXPORT_SELECTORS
from .workspace import get_workspace_paths


def get_project_root():
    return Path(__file__).resolve().parents[2]


def get_output_directory():
    """Return the external generated-data workspace root."""
    return get_workspace_paths(get_project_root()).ensure().root


def export_feature_collection(
    feature_collection,
    output_filename,
    selectors,
    max_attempts=5,
    retry_sleep_seconds=10,
):
    """Download an Earth Engine FeatureCollection to a local CSV."""
    feature_collection = ee.FeatureCollection(feature_collection)
    output_path = get_output_directory() / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        try:
            if temporary_path.exists():
                temporary_path.unlink()

            download_url = feature_collection.getDownloadURL(
                filetype="CSV",
                selectors=list(selectors),
                filename=output_path.stem,
            )
            urlretrieve(download_url, temporary_path)
            temporary_path.replace(output_path)
            return output_path

        except HTTPError as error:
            if temporary_path.exists():
                temporary_path.unlink()
            retryable = error.code in {500, 502, 503, 504}
            if not retryable or attempt == max_attempts:
                raise
            print(f"Earth Engine returned HTTP {error.code}.")

        except (
            URLError,
            RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as error:
            if temporary_path.exists():
                temporary_path.unlink()
            if attempt == max_attempts:
                raise
            print("Network error while downloading Earth Engine output.")
            print("Error:", error)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

        print(f"Retrying download ({attempt}/{max_attempts})...")
        time.sleep(retry_sleep_seconds)

    raise RuntimeError("Earth Engine download failed unexpectedly.")


def export_training_dataset(
    output_table,
    output_filename=OUTPUT_FILENAME,
):
    """Backward-compatible wrapper for satellite footprint exports."""
    return export_feature_collection(
        feature_collection=output_table,
        output_filename=output_filename,
        selectors=SATELLITE_EXPORT_SELECTORS,
    )
