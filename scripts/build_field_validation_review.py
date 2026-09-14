"""Build the two-sheet field review from local inputs and existing halo pixels only.

Uses repository QC, harmonization, aggregation and scenario functions. No metrics,
Earth Engine calls or raster writes. XLSX packaging uses the standard library so
no package installation or network access is needed. period_end is exclusive.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

from et_downscaling.field_validation import (
    KEYS, aggregate_field_periods, apply_scenarios, load_field_inputs,
    modis_periods, ndvi_kc, prepare_field_daily, sha256, unique,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/evaluation/field_validation"
HALOS = OUTPUT / "local_halo7_products"
CACHE = OUTPUT / "cache/b4f368f0a990c09a976ce25b57ea1befc43cd1a62972f3b7fb574dedea4858b2"
WORKBOOK = OUTPUT / "field_validation_review.xlsx"
DAILY_COLUMNS = [
    "station_id", "land_cover", "date", "period_start", "etgage_raw_cm",
    "etgage_mm_day", "etgage_qc_valid", "eto_mm_day", "etr_mm_day",
    "etr_eto_ratio", "etgage_eto_equivalent_mm_day",
]
PERIOD_COLUMNS = [
    "station_id", "land_cover", "period_start", "period_end", "period_days",
    "valid_field_days", "field_eto_equivalent_mm_period", "kc_historical",
    "kc_fao_sensitivity", "ndvi_s2_20m", "kc_ndvi_s2_20m",
    "field_et_historical_mm_period", "field_et_fao_mm_period", "field_et_ndvi_mm_period",
    "modis_et_mm_period", "rf25_kc_raw", "rf25_et_unreconciled_mm_period",
    "rf25_et_reconciled_mm_period", "aoa_inside", "publishable",
    "parent_coarse_eligible", "validation_domain",
]
BANDS = {
    "rf25_kc_raw": "Kc_raw",
    "rf25_et_unreconciled_mm_period": "ET_initial_pre_exact_overlap_mm_period",
    "rf25_et_reconciled_mm_period": "ET_reconciled_published_mm_period",
    "aoa_inside": "AOA_inside", "publishable": "publishable",
    "parent_coarse_eligible": "parent_coarse_eligible",
}
FLAGS = {"etgage_qc_valid", "aoa_inside", "publishable", "parent_coarse_eligible"}


def sample(path, longitude, latitude):
    with rasterio.open(path) as src:
        x, y = transform("EPSG:4326", src.crs, [longitude], [latitude])
        row, col = src.index(x[0], y[0])
        assert 0 <= row < src.height and 0 <= col < src.width, path
        values = next(src.sample([(x[0], y[0])], masked=True)).astype(float).filled(np.nan)
        return dict(zip(src.descriptions, values)), values, src.res, (row, col)


def build_tables():
    manifest = json.loads((HALOS / "deterministic_signature_migration_manifest.json").read_text())
    assert manifest["status"] == "completed"
    hashes_before = {name: sha256(ROOT / name) for name in manifest["tiff_sha256_after"]}
    assert hashes_before == manifest["tiff_sha256_after"]
    entries = {entry["metadata_file"]: entry for entry in manifest["metadata_files"]}
    contract = json.loads((CACHE / "contract.json").read_text())
    for name in ("data/field/field_etgage.csv", "data/stations/fundacion_stations.geojson"):
        assert sha256(ROOT / name) == contract[name], "Cache station/field inputs changed"
    field, stations = load_field_inputs(ROOT)
    assert pd.to_datetime(field.date).dt.year.eq(2022).all()
    reference = pd.read_csv(CACHE / "reference_et_daily.csv")
    daily = prepare_field_daily(field, stations, reference)
    windows = modis_periods(field)
    periods = aggregate_field_periods(daily, windows, stations)
    satellite = pd.read_csv(CACHE / "field_satellite.csv", parse_dates=["period_start"])
    unique(satellite, KEYS, "cached satellite")
    periods = periods.merge(satellite[KEYS + ["NDVI_local_20m", "number_days"]], on=KEYS,
                            how="left", validate="one_to_one", suffixes=("", "_satellite"))
    present = periods.number_days_satellite.notna()
    assert periods.loc[present, "number_days"].eq(periods.loc[present, "number_days_satellite"]).all()
    # NDVI sensitivity is explicitly limited to ST04/ST05 in this review.
    periods.loc[~periods.station_id.isin(["ST04", "ST05"]), "NDVI_local_20m"] = np.nan
    periods.loc[periods.NDVI_local_20m.le(-9990), "NDVI_local_20m"] = np.nan
    pixel_rows = []
    for period in periods.itertuples(index=False):
        date, station = period.period_start.strftime("%Y-%m-%d"), period.station_id
        folder = HALOS / date / station
        metadata_path = folder / "metadata.json"
        record = {"station_id": station, "period_start": period.period_start,
                  "ET_MODIS_mm_period": np.nan, **{name: np.nan for name in BANDS}}
        if metadata_path.is_file():
            entry = entries[metadata_path.relative_to(ROOT).as_posix()]
            assert sha256(metadata_path) == entry["metadata_sha256_after"]
            metadata = json.loads(metadata_path.read_text())
            assert (metadata["station_id"], metadata["period_start"]) == (station, date)
            assert (metadata["longitude"], metadata["latitude"]) == (period.longitude, period.latitude)
            signature = manifest["st04_extension_signature"] if station == "ST04" else manifest["canonical_production_scientific_signature"]
            assert metadata["scientific_signature"] == signature
            raster = folder / f"RF25_halo7_{station}_{date}_20m.tif"
            values, _, resolution, pixel = sample(raster, period.longitude, period.latitude)
            assert resolution == (20.0, 20.0)
            assert Path(metadata["raster"]).resolve() == raster.resolve()
            assert all(name in values for name in BANDS.values())
            record.update({name: values[band] for name, band in BANDS.items()})
            _, modis, _, _ = sample(folder / f"MODIS_halo7_{station}_{date}_native.tif",
                                    period.longitude, period.latitude)
            record["ET_MODIS_mm_period"] = modis[0]
            np.testing.assert_allclose(modis[0], metadata["central_modis_et_mm_period"], rtol=0, atol=1e-6)
            np.testing.assert_allclose(modis[0], values["MODIS_ET_parent_mm_period"], rtol=0, atol=1e-6)
            assert bool(record["parent_coarse_eligible"]) == metadata["central_parent_eligible"]
            if record["publishable"] != 1:
                assert np.isnan(record["rf25_et_reconciled_mm_period"])
            else:
                assert record["aoa_inside"] == record["parent_coarse_eligible"] == 1
            record["sample_pixel"] = str(pixel)
        pixel_rows.append(record)
    assert sum("sample_pixel" in row for row in pixel_rows) == 70
    periods = periods.merge(pd.DataFrame(pixel_rows), on=KEYS, validate="one_to_one")
    periods["ET_RF25_mm_period"] = periods.rf25_et_reconciled_mm_period
    scenarios = apply_scenarios(periods)  # No metrics are computed.
    for scenario, kc, et in (
        ("historical", "kc_historical", "field_et_historical_mm_period"),
        ("fao_sensitivity", "kc_fao_sensitivity", "field_et_fao_mm_period"),
        ("ndvi20_all", "kc_ndvi_s2_20m", "field_et_ndvi_mm_period"),
    ):
        subset = scenarios.loc[scenarios.scenario.eq(scenario), KEYS + ["Kc_field_proxy", "ET_field_proxy_mm_period"]]
        periods = periods.merge(subset.rename(columns={"Kc_field_proxy": kc, "ET_field_proxy_mm_period": et}),
                                on=KEYS, validate="one_to_one")
    periods["period_end"] = periods.period_start + pd.to_timedelta(periods.number_days, unit="D")
    periods["validation_domain"] = np.where(periods.station_id.eq("ST04"), "validation_extension", "fundacion_basin")
    period_table = periods.rename(columns={
        "station": "land_cover", "number_days": "period_days", "n_valid_field_days": "valid_field_days",
        "field_reference_eto_mm_period": "field_eto_equivalent_mm_period",
        "NDVI_local_20m": "ndvi_s2_20m", "ET_MODIS_mm_period": "modis_et_mm_period",
    })[PERIOD_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    # Assign observation days using the very same MODIS windows used by aggregation.
    daily["period_start"] = pd.NaT
    for window in windows.itertuples():
        belongs = daily.date.ge(window.period_start) & daily.date.lt(window.period_start + pd.Timedelta(days=window.number_days))
        daily.loc[belongs, "period_start"] = window.period_start
    assert daily.period_start.notna().all()
    daily_table = daily.rename(columns={
        "station": "land_cover", "etgage_daily_raw": "etgage_raw_cm",
        "etgage_scaled_mm_day": "etgage_mm_day", "field_daily_valid": "etgage_qc_valid",
        "ETo_mm_day": "eto_mm_day", "ETr_mm_day": "etr_mm_day", "ETr_ETo_ratio": "etr_eto_ratio",
    })[DAILY_COLUMNS].sort_values(["station_id", "date"]).reset_index(drop=True)
    for name in FLAGS:
        table = daily_table if name in daily_table else period_table
        table[name] = table[name].astype("boolean")
    assert len(daily_table) == 615 and len(period_table) == 80
    assert period_table.loc[period_table.station_id.isin(["ST01", "ST02", "ST03"]),
                            ["ndvi_s2_20m", "kc_ndvi_s2_20m", "field_et_ndvi_mm_period"]].isna().all().all()
    return daily_table, period_table, hashes_before


def col_letter(number):
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", NS)


def element(name, **attrs):
    return ET.Element(f"{{{NS}}}{name}", {key: str(value) for key, value in attrs.items()})


def child(parent, name, **attrs):
    node = element(name, **attrs)
    parent.append(node)
    return node


def sheet_xml(table):
    sheet = element("worksheet")
    extent = f"A1:{col_letter(len(table.columns))}{len(table) + 1}"
    child(sheet, "dimension", ref=extent)
    view = child(child(sheet, "sheetViews"), "sheetView", workbookViewId=0, showGridLines=0, zoomScale=85)
    child(view, "pane", ySplit=1, topLeftCell="A2", activePane="bottomLeft", state="frozen")
    child(view, "selection", pane="bottomLeft", activeCell="A2", sqref="A2")
    child(sheet, "sheetFormatPr", defaultRowHeight=18)
    columns = child(sheet, "cols")
    for index, name in enumerate(table.columns, 1):
        width = 14 if name in {"station_id", "date", "period_start", "period_end", "period_days"} else min(30, max(19, len(name) * 0.85))
        child(columns, "col", min=index, max=index, width=width, customWidth=1)
    data = child(sheet, "sheetData")
    for number, values in enumerate([table.columns.tolist(), *table.itertuples(index=False, name=None)], 1):
        row = child(data, "row", r=number, **({"ht": 60, "customHeight": 1} if number == 1 else {}))
        for index, value in enumerate(values, 1):
            name = table.columns[index - 1]
            cell = child(row, "c", r=f"{col_letter(index)}{number}")
            if number == 1 or pd.isna(value) or isinstance(value, str):
                cell.set("t", "inlineStr")
                if number == 1:
                    cell.set("s", "1")
                child(child(cell, "is"), "t").text = "NA" if pd.isna(value) else str(value)
            elif isinstance(value, (bool, np.bool_)):
                cell.set("t", "b")
                child(cell, "v").text = str(int(value))
            elif isinstance(value, (pd.Timestamp, datetime)):
                cell.set("s", "2")
                child(cell, "v").text = str((value - pd.Timestamp("1899-12-30")).days)
            else:
                integer = name in {"period_days", "valid_field_days"}
                precision = "4" if "kc" in name or "ndvi" in name or "ratio" in name else "3"
                cell.set("s", "0" if integer else precision)
                child(cell, "v").text = str(value)
    child(sheet, "autoFilter", ref=extent)
    return ET.tostring(sheet, encoding="utf-8", xml_declaration=True)


def save_workbook(daily, periods):
    styles = f'''<styleSheet xmlns="{NS}">
    <numFmts count="3"><numFmt numFmtId="164" formatCode="yyyy-mm-dd"/><numFmt numFmtId="165" formatCode="0.000"/><numFmt numFmtId="166" formatCode="0.0000"/></numFmts>
    <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
    <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF203D54"/><bgColor indexed="64"/></patternFill></fill></fills>
    <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
    <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
    <cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>
    <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    relationships = "http://schemas.openxmlformats.org/package/2006/relationships"
    office = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with ZipFile(WORKBOOK, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' + ''.join(f'<Override PartName="/{part}" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.{kind}+xml"/>' for part, kind in [("xl/workbook.xml", "sheet.main"), ("xl/styles.xml", "styles"), ("xl/worksheets/sheet1.xml", "worksheet"), ("xl/worksheets/sheet2.xml", "worksheet")]) + '</Types>')
        archive.writestr("_rels/.rels", f'<Relationships xmlns="{relationships}"><Relationship Id="rId1" Type="{office}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{NS}" xmlns:r="{office}"><sheets><sheet name="DAILY" sheetId="1" r:id="rId1"/><sheet name="PERIOD" sheetId="2" r:id="rId2"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{relationships}"><Relationship Id="rId1" Type="{office}/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="{office}/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="{office}/styles" Target="styles.xml"/></Relationships>')
        archive.writestr("xl/styles.xml", styles)
        for index, table in enumerate((daily, periods), 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(table))
    # Reopen the finished archive and verify every cell, sheet name and filter.
    with ZipFile(WORKBOOK) as archive:
        assert archive.testzip() is None
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        assert [sheet.attrib["name"] for sheet in workbook.find(f"{{{NS}}}sheets")] == ["DAILY", "PERIOD"]
        for index, table in enumerate((daily, periods), 1):
            sheet = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))
            assert sheet.find(f"{{{NS}}}autoFilter") is not None
            assert sheet.find(f".//{{{NS}}}pane").attrib["state"] == "frozen"
            rows = sheet.find(f"{{{NS}}}sheetData")
            assert len(rows) == len(table) + 1
            for row, expected in zip(rows, [list(table.columns), *table.itertuples(index=False, name=None)]):
                assert len(row) == len(expected)
                for cell, value in zip(row, expected):
                    if cell.get("t") == "inlineStr":
                        assert cell.find(f"{{{NS}}}is/{{{NS}}}t").text == ("NA" if pd.isna(value) else str(value))
                    else:
                        actual = float(cell.find(f"{{{NS}}}v").text)
                        target = (value - pd.Timestamp("1899-12-30")).days if isinstance(value, (pd.Timestamp, datetime)) else float(value)
                        assert actual == target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save after the row checks have been reviewed")
    args = parser.parse_args()
    daily, periods, raster_hashes = build_tables()
    print("DAILY CHECK: 2022-03-14")
    print(daily.loc[daily.date.eq("2022-03-14")].to_string(index=False))
    print("PERIOD CHECK: 2022-03-14 (period_end exclusive)")
    print(periods.loc[periods.period_start.eq("2022-03-14")].to_string(index=False))
    for name, table in (("DAILY", daily), ("PERIOD", periods)):
        print(name, "rows", len(table), "per station", table.groupby("station_id").size().to_dict())
        print(name, "missing", table.isna().sum().to_dict())
    if args.save:
        save_workbook(daily, periods)
        assert {name: sha256(ROOT / name) for name in raster_hashes} == raster_hashes
        print("SAVED AND VERIFIED", WORKBOOK)


if __name__ == "__main__":
    main()
