"""Versioned identities of physical field stations; independent of Virtual10 IDs."""
from __future__ import annotations

VERSION = "field_station_nomenclature_v2_20260914"
OLD_TO_NEW = {"ST01": "ST02", "ST02": "ST03", "ST03": "ST04", "ST04": "ST01", "ST05": "ST05"}
NEW_TO_OLD = {new: old for old, new in OLD_TO_NEW.items()}
STATIONS = {
    "ST01": dict(station="Mangrove", station_uid="mangrove", source_sheet="Mangrove",
                 longitude=-74.360002, latitude=10.766952, reference_et="ETr",
                 installation_date="2022-03-17", removal_date="2022-07-01", canvas="#54",
                 inside_basin=False, installation_conforms_manual=False, kc_rule="ndvi",
                 historical_kc=None, fao_kc=None, validation_domain="validation_extension",
                 coastal_era5_support=True),
    "ST02": dict(station="Pasture", station_uid="clean_pasture", source_sheet="Pastos Limpios",
                 longitude=-74.421552, latitude=10.658272, reference_et="ETo",
                 installation_date="2022-03-10", removal_date="2022-07-01", canvas="#30",
                 inside_basin=True, installation_conforms_manual=True, kc_rule="fixed",
                 historical_kc=0.85, fao_kc=0.75, validation_domain="fundacion_basin",
                 coastal_era5_support=False),
    "ST03": dict(station="Oil palm plantation", station_uid="oil_palm", source_sheet="Palmera",
                 longitude=-74.362748, latitude=10.620151, reference_et="ETr",
                 installation_date="2022-03-11", removal_date="2022-07-01", canvas="#54",
                 inside_basin=True, installation_conforms_manual=False, kc_rule="fixed",
                 historical_kc=0.95, fao_kc=1.0, validation_domain="fundacion_basin",
                 coastal_era5_support=False),
    "ST04": dict(station="Banana plantation", station_uid="banana", source_sheet="Banana",
                 longitude=-74.346754, latitude=10.626894, reference_et="ETr",
                 installation_date="2022-03-11", removal_date="2022-07-01", canvas="#54",
                 inside_basin=True, installation_conforms_manual=False, kc_rule="fixed",
                 historical_kc=1.1, fao_kc=1.1, validation_domain="fundacion_basin",
                 coastal_era5_support=False),
    "ST05": dict(station="Dry forest", station_uid="dry_forest", source_sheet="Fragmento de bosque",
                 longitude=-74.117804, latitude=10.485199, reference_et="ETr",
                 installation_date="2022-03-10", removal_date="2022-07-01", canvas="#54",
                 inside_basin=True, installation_conforms_manual=False, kc_rule="ndvi",
                 historical_kc=None, fao_kc=None, validation_domain="fundacion_basin",
                 coastal_era5_support=False),
}
MANGROVE_STATION_ID = "ST01"
NDVI_STATION_IDS = tuple(key for key, value in STATIONS.items() if value["kc_rule"] == "ndvi")
BASIN_STATION_IDS = tuple(key for key, value in STATIONS.items() if value["inside_basin"])
FIXED_KC = {
    "historical": {key: value["historical_kc"] for key, value in STATIONS.items() if value["kc_rule"] == "fixed"},
    "fao_sensitivity": {key: value["fao_kc"] for key, value in STATIONS.items() if value["kc_rule"] == "fixed"},
    "ndvi20_all": {},  # The pre-existing all-station sensitivity is preserved.
}
# This historical algorithm identifier is part of the scientific signature.
# It describes the same mangrove method, even though mangrove is now ST01.
MANGROVE_METHOD = "st04_validation_extension_era5_nearest_valid_land_pixel_fill_v1"


def validate_station_geometry(geojson):
    """Reject a label-only permutation, not just an invalid set of five IDs."""
    features = geojson["features"]
    if len(features) != 5 or {f["properties"]["station_id"] for f in features} != set(STATIONS):
        raise ValueError("Expected exactly five versioned physical field stations.")
    for feature in features:
        properties = feature["properties"]
        station_id = properties["station_id"]
        expected = STATIONS[station_id]
        if properties.get("station_nomenclature_version") != VERSION:
            raise ValueError("Field station naming version is missing or legacy.")
        if feature["geometry"]["coordinates"] != [expected["longitude"], expected["latitude"]]:
            raise ValueError(f"Physical coordinates do not match {station_id}.")
        for key in ("station", "station_uid", "reference_et", "installation_date", "removal_date",
                    "canvas", "inside_basin", "installation_conforms_manual", "kc_rule",
                    "validation_domain", "coastal_era5_support"):
            if properties.get(key) != expected[key]:
                raise ValueError(f"Physical attribute {key} does not match {station_id}.")


def validate_station_table(table, *, require_uid=False):
    """Validate available physical identity evidence without changing any values."""
    if require_uid and not {"station_uid", "station_nomenclature_version"}.issubset(table.columns):
        raise ValueError("Field input requires station_uid and station_nomenclature_version; migrate legacy station keys explicitly.")
    if table.station_id.isna().any() or (require_uid and table[["station_uid", "station_nomenclature_version"]].isna().any().any()):
        raise ValueError("Missing physical station identity.")
    for station_id, group in table.groupby("station_id"):
        if station_id not in STATIONS:
            raise ValueError(f"Unknown physical field station {station_id}.")
        expected = STATIONS[station_id]
        for column, value in {"station": expected["station"], "land_cover": expected["station"],
                              "station_uid": expected["station_uid"],
                              "station_nomenclature_version": VERSION,
                              "source_sheet": expected["source_sheet"]}.items():
            if column in group and not group[column].dropna().eq(value).all():
                raise ValueError(f"Physical identity mismatch for {station_id}: {column}.")


def attach_station_identity(table):
    """Label newly acquired current-geometry records, never migrate old inputs."""
    result = table.copy()
    result["station_uid"] = result.station_id.map({key: value["station_uid"] for key, value in STATIONS.items()})
    result["station_nomenclature_version"] = VERSION
    validate_station_table(result, require_uid=True)
    return result
