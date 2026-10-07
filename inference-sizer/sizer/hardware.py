"""Hardware catalog loader with source metadata and independent statuses."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schemas import Catalog, HardwareProfile, ModelPreset

DATA = Path(__file__).parent / "data" / "hardware.json"

CATALOG_VERSION = 2


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    raw = json.loads(DATA.read_text())
    return Catalog(
        catalog_version=raw["catalog_version"],
        hardware_profiles=[HardwareProfile.model_validate(p) for p in raw["hardware_profiles"]],
        presets=[ModelPreset.model_validate(p) for p in raw["presets"]],
        engines=raw["engines"],
    )


def get_profile(profile_id: str) -> HardwareProfile:
    for p in load_catalog().hardware_profiles:
        if p.id == profile_id:
            return p
    raise KeyError(
        f"Unknown hardware_profile_id {profile_id!r}. Valid ids: "
        + ", ".join(p.id for p in load_catalog().hardware_profiles)
    )


def get_preset(preset_id: str) -> ModelPreset:
    for p in load_catalog().presets:
        if p.id == preset_id:
            return p
    raise KeyError(
        f"Unknown preset_id {preset_id!r}. Valid ids: "
        + ", ".join(p.id for p in load_catalog().presets)
    )


def device_memory_bytes(profile: HardwareProfile, observed_gb: float | None) -> tuple[int, str]:
    """Per-device allocatable memory in bytes.

    Observed device memory overrides the published nominal value when supplied;
    per-device fit is validated, never pooled HBM.
    """
    if observed_gb is not None:
        return int(observed_gb * 1024**3), "observed"
    return int(profile.memory_per_device_gb * 1e9), "published"