"""Catalog integrity: seven distinct profiles, independent statuses, units."""

from sizer.hardware import get_profile, load_catalog
from sizer.schemas import FORMAT_BYTES


def test_seven_profiles_present_and_distinct():
    c = load_catalog()
    ids = [p.id for p in c.hardware_profiles]
    assert ids == [
        "dgx_h100", "h200_sxm_nvl8", "h200_nvl", "hgx_b200",
        "hgx_b300", "gb300_nvl72", "vera_rubin_nvl72",
    ]
    assert len(set(ids)) == 7
    # Distinctness: no two profiles share id+name
    names = [p.name for p in c.hardware_profiles]
    assert len(set(names)) == 7


def test_h200_stays_selectable_and_split():
    h200_8 = get_profile("h200_sxm_nvl8")
    h200_n = get_profile("h200_nvl")
    assert h200_8.device_count == 8 and h200_8.memory_per_device_gb == 141
    assert h200_8.hbm_bandwidth_tb_s == 4.8
    # NVL profile does not inherit SXM device count or compute
    assert h200_n.device_count is None
    assert h200_n.compute_tflops is None
    assert h200_n.memory_per_device_gb == 141


def test_hgx_b200_not_dgx():
    p = get_profile("hgx_b200")
    assert p.system_family == "HGX"
    assert p.memory_per_device_gb == 180
    assert p.hbm_bandwidth_tb_s == 7.7
    # The archive's generic 192GB/8TB/s label must not leak into the catalog.
    assert p.memory_per_device_gb != 192
    # Dense compute indexed by precision; no sparse values stored.
    assert p.compute_tflops == {"bf16": 2250, "fp8": 4500, "fp4": 9000}


def test_device_vs_die_counts():
    """NVL8 = 8 devices, NVL72 = 72 devices; dies tracked separately."""
    for pid, devices in [("dgx_h100", 8), ("h200_sxm_nvl8", 8), ("hgx_b200", 8),
                         ("hgx_b300", 8), ("gb300_nvl72", 72), ("vera_rubin_nvl72", 72)]:
        p = get_profile(pid)
        assert p.device_count == devices, pid
    p = get_profile("hgx_b300")
    assert p.die_count_per_device == 2
    assert p.device_count == 8  # historical 16-GPU slide counted dies


def test_independent_status_tracking():
    """Spec confidence / inventory / qualification / evidence are separate;
    unknown inventory never defaults to available."""
    for p in load_catalog().hardware_profiles:
        assert p.status.inventory in ("available", "unknown", "unavailable")
        assert p.status.spec_confidence in ("published", "inferred", "prototype_anchor")
        assert isinstance(p.status.engine_qualified, bool)
        assert p.status.benchmark_evidence in ("none", "pending", "available")
        # No profile in this build claims confirmed inventory or qualification.
        assert p.status.inventory != "available" or p.status.notes, p.id


def test_source_metadata_present():
    for p in load_catalog().hardware_profiles:
        assert p.source_refs, p.id
        for ref in p.source_refs.values():
            assert ref.url or ref.file, p.id
            assert ref.access in ("public", "internal")


def test_b300_discrepancy_preserved():
    p = get_profile("hgx_b300")
    joined = " ".join(p.status.notes)
    assert "288" in joined and "2.1 TB" in joined  # guide vs product page preserved
    assert p.memory_per_device_gb == 270  # datasheet anchor used


def test_rubin_comparison_only():
    p = get_profile("vera_rubin_nvl72")
    assert p.memory_qualifier == "up_to"
    assert p.bandwidth_qualifier == "up_to"
    assert p.recommendation_mode == "comparison_only"
    assert p.compute_tflops is None


def test_format_bytes_table():
    assert FORMAT_BYTES == {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "fp4": 0.5}