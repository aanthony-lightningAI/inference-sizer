"""FastAPI app: /api/health, /api/catalog, /api/size + built frontend (one origin).

CORS is constrained to configured origins (env CORS_ORIGINS, comma-separated);
the default is no cross-origin access.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from sizer.engine import size
from sizer.hardware import load_catalog
from sizer.normalize import preset_list
from sizer.schemas import SCHEMA_VERSION, SizeRequest

app = FastAPI(title="Lightning AI Inference Sizer", version="1.0.0")

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )


@app.exception_handler(ValidationError)
async def validation_handler(_req, exc: ValidationError):
    """Field-specific errors instead of a 500."""
    errors = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e.get("loc", []))
        errors.append({"field": loc, "message": e.get("msg", str(e))})
    return JSONResponse(status_code=422, content={"detail": "Validation failed", "errors": errors})


@app.get("/api/health")
def health():
    return {"ok": True, "schema_version": SCHEMA_VERSION, "calculator_version": "1.0.0"}


@app.get("/api/catalog")
def catalog():
    c = load_catalog()
    return {
        "catalog_version": c.catalog_version,
        "schema_version": SCHEMA_VERSION,
        "presets": preset_list(),
        "hardware_profiles": [p.model_dump() for p in c.hardware_profiles],
        "engines": c.engines,
    }


@app.post("/api/size")
def size_route(req: SizeRequest):
    return size(req).model_dump(mode="json")


# Serve the built frontend under one origin (mount last so /api wins).
_dist = Path(__file__).parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")