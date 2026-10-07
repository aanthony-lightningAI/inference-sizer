from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sizer.engine import GPUS, PRESETS, SizeRequest, size

app = FastAPI(title="Inference GPU sizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/catalog")
def catalog():
    return {"presets": PRESETS, "gpus": GPUS}


@app.post("/api/size")
def size_route(req: SizeRequest):
    return size(req)
