"""Read the validated field workbook without optional Excel engines or writes."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import posixpath

import numpy as np
import pandas as pd


def read_review_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relationship_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        properties = workbook.find("s:workbookPr", namespace)
        assert properties is None or properties.get("date1904", "0") in ("0", "false")
        selected = next(sheet for sheet in workbook.findall("s:sheets/s:sheet", namespace)
                        if sheet.get("name") == sheet_name)
        targets = {link.get("Id"): link.get("Target") for link in
                   ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        target = targets[selected.get(f"{{{relationship_ns}}}id")]
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(archive.read(member)).findall("s:sheetData/s:row", namespace):
            values = {}
            for cell in row:
                assert cell.find("s:f", namespace) is None, "Unexpected workbook formula."
                column = 0
                for letter in filter(str.isalpha, cell.get("r")):
                    column = column * 26 + ord(letter.upper()) - ord("A") + 1
                kind = cell.get("t")
                raw = cell.findtext("s:v", default="", namespaces=namespace)
                if kind == "inlineStr":
                    value = "".join(cell.find("s:is", namespace).itertext())
                elif kind == "s":
                    value = shared[int(raw)]
                elif kind == "b":
                    value = raw == "1"
                elif kind == "e":
                    raise ValueError(f"Excel error: {sheet_name}!{cell.get('r')}")
                elif kind in ("str", "d"):
                    value = raw
                else:
                    value = float(raw) if raw else np.nan
                values[column - 1] = np.nan if isinstance(value, str) and value in ("", "NA") else value
            rows.append(values)
    header = [rows[0][index] for index in range(len(rows[0]))]
    frame = pd.DataFrame([[row.get(index, np.nan) for index in range(len(header))] for row in rows[1:]], columns=header)
    for column in ("date", "period_start", "period_end"):
        if column in frame:
            frame[column] = (pd.to_datetime(frame[column], unit="D", origin="1899-12-30")
                             if pd.api.types.is_numeric_dtype(frame[column]) else pd.to_datetime(frame[column]))
    for column in ("etgage_qc_valid", "publishable", "aoa_inside", "parent_coarse_eligible"):
        if column in frame:
            frame[column] = frame[column].astype("boolean")
    return frame
