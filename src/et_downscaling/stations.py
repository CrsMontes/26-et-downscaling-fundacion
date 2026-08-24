from __future__ import annotations

import json
from pathlib import Path

import ee
import pandas as pd

from .config import STATIONS_GEOJSON_PATH


REQUIRED_STATION_PROPERTIES = (
    "station_id",
    "station",
    "station_slug",
)


def get_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_station_geojson_path() -> Path:
    return (
        get_repository_root()
        / STATIONS_GEOJSON_PATH
    ).resolve()


def load_station_geojson() -> dict:
    path = get_station_geojson_path()

    if not path.is_file():
        raise FileNotFoundError(
            f"Station GeoJSON not found: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if data.get("type") != "FeatureCollection":
        raise ValueError(
            "Station input must be a GeoJSON FeatureCollection."
        )

    features = data.get(
        "features",
        [],
    )

    if not features:
        raise ValueError(
            "Station GeoJSON contains no features."
        )

    station_ids = []

    for index, feature in enumerate(features):
        geometry = feature.get("geometry")
        properties = feature.get(
            "properties",
            {},
        )

        if (
            not geometry
            or geometry.get("type") != "Point"
        ):
            raise ValueError(
                f"Station feature {index} must have Point geometry."
            )

        missing = [
            key
            for key in REQUIRED_STATION_PROPERTIES
            if properties.get(key) in (None, "")
        ]

        if missing:
            raise ValueError(
                f"Station feature {index} is missing: {missing}"
            )

        station_ids.append(
            str(
                properties["station_id"]
            )
        )

    if len(station_ids) != len(set(station_ids)):
        raise ValueError(
            "station_id values must be unique."
        )

    return data


def load_station_dataframe() -> pd.DataFrame:
    data = load_station_geojson()
    rows = []

    for feature in data["features"]:
        properties = dict(
            feature["properties"]
        )

        longitude, latitude = (
            feature["geometry"]["coordinates"]
        )

        properties["station_id"] = str(
            properties["station_id"]
        )
        properties["longitude"] = float(
            longitude
        )
        properties["latitude"] = float(
            latitude
        )

        rows.append(
            properties
        )

    return (
        pd.DataFrame(rows)
        .sort_values("station_id")
        .reset_index(drop=True)
    )


def get_station_collection() -> ee.FeatureCollection:
    data = load_station_geojson()
    features = []

    for feature in data["features"]:
        properties = dict(
            feature["properties"]
        )

        properties["station_id"] = str(
            properties["station_id"]
        )

        features.append(
            ee.Feature(
                ee.Geometry(
                    feature["geometry"]
                ),
                properties,
            )
        )

    return ee.FeatureCollection(
        features
    )
