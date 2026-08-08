import time
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.request import urlretrieve

import ee

from .config import (
    OUTPUT_DIRECTORY,
    OUTPUT_FILENAME,
)

from .schema import (
    EXPORT_SELECTORS,
)


# ============================================================
# Get project output directory
# ============================================================

def get_output_directory():
    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    output_directory = (
        project_root
        / OUTPUT_DIRECTORY
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


# ============================================================
# Export Earth Engine FeatureCollection to local CSV
# ============================================================

def export_training_dataset(
    output_table,
    output_filename=OUTPUT_FILENAME,
):
    output_table = ee.FeatureCollection(
        output_table
    )

    output_directory = (
        get_output_directory()
    )

    output_path = (
        output_directory
        / output_filename
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".part"
        )
    )

    max_attempts = 5

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            if temporary_path.exists():
                temporary_path.unlink()

            download_url = (
                output_table.getDownloadURL(
                    filetype="CSV",
                    selectors=EXPORT_SELECTORS,
                    filename=output_path.stem,
                )
            )

            urlretrieve(
                download_url,
                temporary_path,
            )

            temporary_path.replace(
                output_path
            )

            break

        except HTTPError as error:
            if temporary_path.exists():
                temporary_path.unlink()

            if (
                error.code
                not in {
                    500,
                    502,
                    503,
                    504,
                }
                or attempt == max_attempts
            ):
                raise

            print(
                f"Earth Engine returned HTTP "
                f"{error.code}."
            )

            print(
                f"Retrying download "
                f"({attempt}/{max_attempts})..."
            )

            time.sleep(10)

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

            print(
                "Network error while downloading "
                "Earth Engine output."
            )

            print(
                "Error:",
                error,
            )

            print(
                f"Retrying download "
                f"({attempt}/{max_attempts})..."
            )

            time.sleep(10)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()

            raise

    return output_path
