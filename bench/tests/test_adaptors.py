"""Adaptor bookkeeping. No GPU: these check the contract, not the forward pass."""

import json

import pytest

from bench import registry
from bench.adaptors.local_hf import split_probe_id
from bench.store import trial_key

registry.load_all()


@pytest.mark.parametrize("name", ["local_hf", "opencode"])
def test_model_id_stays_a_string_and_describe_is_json_safe(name):
    adaptor = registry.get_adaptor(name)(seed=42)
    assert isinstance(adaptor.model, str) and adaptor.model
    payload = adaptor.describe()
    json.dumps(payload)                       # manifest is written with json.dump
    assert isinstance(payload["model"], str)
    assert payload["name"] == name


def test_local_hf_keeps_the_module_off_the_model_attribute():
    """setup() must not clobber the model *id* with the nn.Module.

    It did once, and the manifest write then blew up after the weights were
    already loaded -- i.e. the most expensive possible place to fail.
    """
    adaptor = registry.get_adaptor("local_hf")()
    assert adaptor.hf_model is None
    assert adaptor.model == adaptor.model_path
    trial_key("vote2020", "i", "C", adaptor.name, adaptor.model, 42, "rev")  # must not raise


def test_trial_key_refuses_empty_fields():
    for bad in [None, ""]:
        with pytest.raises(ValueError):
            trial_key("vote2020", "i", "C", "local_hf", bad, 42, "rev")


def test_split_probe_id():
    assert split_probe_id("combined_ideology_headwise_linear") == ("combined_ideology", "headwise_linear")
    assert split_probe_id("textual_ideology_layerwise_linear") == ("textual_ideology", "layerwise_linear")
    with pytest.raises(ValueError):
        split_probe_id("combined_ideology")


def test_opencode_prompt_carries_the_same_words_as_the_local_arm():
    from bench.adaptors.opencode import OpenCodeAdaptor
    from bench.surfaces.base import ASSISTANT_TURN_1, SHARE_LINE
    from bench.types import Item

    item = Item(item_id="i", images=["a"], image_paths=["/tmp/a.jpg"],
                image_scores=[0.1], decile=5)
    trial = registry.get_surface("vote2020")().build(item, "D")
    prompt = OpenCodeAdaptor._render_prompt(trial)
    assert SHARE_LINE in prompt and ASSISTANT_TURN_1 in prompt
    assert prompt.rstrip().endswith("ASSISTANT:")
