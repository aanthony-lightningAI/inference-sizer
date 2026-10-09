"""API integration: health, catalog, size, field errors, CORS, CLI agreement."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "inference-sizer"


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import os

    os.environ.pop("CORS_ORIGINS", None)
    sys.path.insert(0, str(PKG))
    from server import app

    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["schema_version"] == 2


def test_catalog(client):
    r = client.get("/api/catalog")
    assert r.status_code == 200
    c = r.json()
    assert c["catalog_version"] == 3
    assert len(c["hardware_profiles"]) == 7
    assert any(p["id"] == "h200_sxm_nvl8" for p in c["hardware_profiles"])
    # Extended catalog: every preset carries family + confidence; MLA preset enabled
    assert all("family" in p and "spec_confidence" in p for p in c["presets"])
    dsv3 = next(p for p in c["presets"] if p["id"] == "deepseek_v3_mla")
    assert dsv3["unsupported_reason"] is None


def test_size_ok(client):
    from sizer.cli import _example_request

    r = client.post("/api/size", json=_example_request())
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == 2
    assert body["feasibility"] == "feasible"


def test_size_field_errors(client):
    """Field-specific 422s, not IndexError or silent clamping."""
    from sizer.cli import _example_request

    bad = _example_request()
    bad["deployment"]["max_gpus_per_replica"] = 0
    r = client.post("/api/size", json=bad)
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any("max_gpus_per_replica" in e["field"] for e in errors)
    assert isinstance(errors, list) and errors[0]["message"]


def test_cli_api_agreement(client, tmp_path):
    """CLI and API must agree on the same input (same size() path)."""
    from sizer.cli import _example_request
    from sizer.engine import size
    from sizer.schemas import SizeRequest

    req = SizeRequest.model_validate(_example_request())
    api = client.post("/api/size", json=req.model_dump(mode="json")).json()
    cli = size(req).model_dump(mode="json")
    assert api == cli


def test_cors_default_closed(client):
    r = client.options(
        "/api/size",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_cors_env_open():
    import os

    os.environ["CORS_ORIGINS"] = "https://sizer.internal.lightning.ai"
    sys.path.insert(0, str(PKG))
    from importlib import reload

    import server

    reload(server)
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    r = c.options(
        "/api/size",
        headers={"Origin": "https://sizer.internal.lightning.ai", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "https://sizer.internal.lightning.ai"
    r2 = c.options(
        "/api/size",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert r2.headers.get("access-control-allow-origin") != "https://evil.example"
    del os.environ["CORS_ORIGINS"]


def test_cli_export_roundtrip(tmp_path):
    """CLI --export writes a full scenario export that imports back to the same result."""
    from sizer.cli import _example_request, _result_from_payload, main  # noqa
    from sizer.engine import size
    from sizer.schemas import SizeRequest

    req_file = tmp_path / "req.json"
    out_file = tmp_path / "export.json"
    req_file.write_text(json.dumps(_example_request()))

    sys.path.insert(0, str(PKG))
    import sizer.cli as cli

    orig = sys.argv
    sys.argv = ["cli", str(req_file), "--export", str(out_file)]
    try:
        cli.main()
    finally:
        sys.argv = orig

    export = json.loads(out_file.read_text())
    assert export["export_schema_version"] == 1
    assert export["request"] and export["result"]
    assert export["result"]["memory_components"] is not None

    # Re-import reproduces the identical result.
    req2, bench = _result_from_payload(export)
    assert req2.model_dump(mode="json") == SizeRequest.model_validate(export["request"]).model_dump(mode="json")
    assert size(req2).model_dump(mode="json") == export["result"]


def test_customer_text_not_executed():
    """Sizing result carries customer text verbatim for text rendering (no HTML)."""
    from sizer.engine import size
    from sizer.cli import _example_request
    from sizer.schemas import SizeRequest

    evil = "<script>alert(1)</script>"
    base = json.loads(json.dumps(_example_request()))
    base["customer"] = evil
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    assert r["customer"] == evil  # stored verbatim; frontend renders as text