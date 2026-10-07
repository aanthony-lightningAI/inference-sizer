"""Shared fixtures. Always runs against the extracted inference-sizer package."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "inference-sizer"))

from sizer.engine import size  # noqa: E402
from sizer.schemas import SizeRequest  # noqa: E402


@pytest.fixture
def base_request() -> SizeRequest:
    """70B GQA on HGX B200, interactive chat — the regression baseline request."""
    from sizer.cli import _example_request

    return SizeRequest.model_validate(_example_request())


@pytest.fixture
def size_it(base_request):
    def _run(**updates) -> dict:
        d = base_request.model_dump(mode="json")
        for k, v in updates.items():
            d[k] = v
        return size(SizeRequest.model_validate(d)).model_dump(mode="json")

    return _run


@pytest.fixture
def synthetic_profile() -> dict:
    p = REPO / "fixtures" / "benchmarks" / "synthetic_hgx_b200_70b.json"
    return json.loads(p.read_text())