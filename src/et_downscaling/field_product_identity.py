"""Separate reviewed retry mechanics from RF25 payload/decoding semantics.

Only named retry declarations and transport imports are excluded. All other
module statements, the successful request/read path, nodata and the GeoTIFF
decoder remain scientific. Full source bytes remain in execution evidence.
"""
import ast
import copy
import hashlib
import json

DOWNLOADER = "src/et_downscaling/ee_download.py"
PAYLOAD_CONTRACT = "scientific/ee_download_payload_ast_v1"


def contract_identity(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def downloader_scientific_hash(source: str) -> str:
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import) and all(n.name in {"random", "time", "urllib.request"} and n.asname is None for n in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and all(n.asname is None for n in node.names) and (
            (node.module == "http.client" and all(n.name in {"IncompleteRead", "RemoteDisconnected"} for n in node.names))
            or (node.module == "urllib.error" and all(n.name in {"HTTPError", "URLError"} for n in node.names))
        ):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in {
            "DIRECT_DOWNLOAD_MAX_ATTEMPTS", "TRANSIENT_HTTP_STATUS_CODES"
        } for t in node.targets):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "_retry_delay_seconds":
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "download_ee_bytes":
            # This reviewed downloader retries one unchanged successful request.
            # Unknown control-flow layouts fail closed instead of being ignored.
            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
            if not (len(body) == 2 and isinstance(body[0], ast.For) and isinstance(body[1], ast.Raise)
                    and not body[0].orelse and len(body[0].body) == 1
                    and isinstance(body[0].body[0], ast.Try)):
                raise ValueError("Downloader structure needs a new scientific/transport review.")
            attempt = body[0].body[0]
            if attempt.orelse or attempt.finalbody:
                raise ValueError("Downloader success/finally path needs scientific review.")
            if any(isinstance(n, ast.Return) for h in attempt.handlers for n in ast.walk(h)):
                raise ValueError("Retry handlers must not supply replacement payloads.")
            node = copy.deepcopy(node)
            node.body = copy.deepcopy(attempt.body)
        kept.append(node)
    canonical = ast.dump(ast.Module(body=kept, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def scientific_contract(execution_contract: dict, downloader_source: str) -> dict:
    if execution_contract.get(DOWNLOADER) != hashlib.sha256(downloader_source.encode()).hexdigest():
        raise ValueError("Downloader source snapshot does not match execution SHA-256.")
    result = {k: v for k, v in execution_contract.items() if k != DOWNLOADER}
    result[PAYLOAD_CONTRACT] = downloader_scientific_hash(downloader_source)
    return dict(sorted(result.items()))
