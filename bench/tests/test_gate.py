"""Capability gate: vote2020 x opencode degrades, vote2020 x local_hf passes."""

import pytest

from bench import registry
from bench.adaptors.base import check_capabilities
from bench.types import Capability

registry.load_all()


def _pair(surface_name, adaptor_name):
    return check_capabilities(registry.get_surface(surface_name)(),
                              registry.get_adaptor(adaptor_name))


def test_opencode_does_not_claim_logprob_or_activations():
    caps = registry.get_adaptor("opencode").capabilities
    assert Capability.LOGPROB not in caps
    assert Capability.ACTIVATIONS not in caps
    assert Capability.GENERATE in caps
    assert Capability.IMAGES in caps
    assert Capability.SESSION in caps


def test_local_hf_claims_the_white_box_capabilities():
    caps = registry.get_adaptor("local_hf").capabilities
    for cap in (Capability.GENERATE, Capability.LOGPROB, Capability.ACTIVATIONS, Capability.IMAGES):
        assert cap in caps
    assert Capability.SESSION not in caps


def test_vote2020_on_opencode_is_degraded_not_blocked():
    report = _pair("vote2020", "opencode")
    assert report.ok is True, "opencode can still generate text, so it is not blocked"
    assert report.degraded is True
    assert report.status == "DEGRADED"
    blob = " ".join(report.degradations)
    assert "logprob" in blob
    assert "s_txt" in blob, "the report must name what is lost"
    assert "activations" in blob
    assert "DEGRADED" in report.render()


def test_vote2020_on_local_hf_passes_at_full_fidelity():
    report = _pair("vote2020", "local_hf")
    assert report.ok is True
    assert report.degraded is False
    assert report.status == "OK"
    assert report.missing_required == []
    assert "full fidelity" in report.render()


@pytest.mark.parametrize("surface", ["vote2020", "guns", "healthcare", "border",
                                     "tea_coffee", "cat_dog", "beach_mountain", "morning_night"])
def test_every_surface_gates_the_same_way(surface):
    assert _pair(surface, "local_hf").status == "OK"
    assert _pair(surface, "opencode").status == "DEGRADED"


def test_blocked_when_a_required_capability_is_missing():
    class TextOnly:
        name = "text_only"
        capabilities = frozenset({Capability.GENERATE})

    report = check_capabilities(registry.get_surface("vote2020")(), TextOnly())
    assert report.ok is False
    assert report.status == "BLOCKED"
    assert "images" in report.missing_required
