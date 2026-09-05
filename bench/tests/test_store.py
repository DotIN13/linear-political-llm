"""Store: dedup on trial_key, resume across process restarts, conversations by sha."""

import json
import os

import pytest

from bench.store import (
    ConversationStore, RunStore, measurement_inputs, measurement_rev, trial_key, write_items,
)
from bench.types import Conversation, canonical_variant

VARIANT = {"phrasing": 0, "order": "ab"}
ARGS = ("vote2020", "lvis3_00001", "C", VARIANT, "local_hf", "Qwen3-VL-8B-Instruct", 42, "a1b2c3d4e5f6")


def _record(key, **extra):
    payload = {"trial_key": key, "surface": "vote2020", "condition": "C",
               "item_id": "lvis3_00001", "decile": 3,
               "outcome": {"kind": "logprob_diff", "value": 1.0, "extra": {}},
               "probe": {"s_txt": 0.2, "s_img": 0.4}}
    payload.update(extra)
    return payload


def test_trial_key_is_stable_and_sensitive():
    assert trial_key(*ARGS) == trial_key(*ARGS)
    assert trial_key(*ARGS).startswith("sha256:")
    for i in range(len(ARGS)):
        mutated = list(ARGS)
        if isinstance(mutated[i], dict):
            mutated[i] = {"phrasing": 1, "order": "ab"}
        elif isinstance(mutated[i], int):
            mutated[i] = 999
        else:
            mutated[i] = str(mutated[i]) + "x"
        assert trial_key(*mutated) != trial_key(*ARGS), f"field {i} does not affect the key"


# --- task B: the variant is part of the identity ---------------------------- #
def test_variant_is_in_the_key_so_rephrasings_do_not_collide():
    """Without this, three rewordings of one question overwrite each other."""
    def key(variant):
        return trial_key("vote2020", "lvis3_00001", "C", variant,
                         "local_hf", "m", 42, "rev123456789")

    keys = [key({"phrasing": p, "order": o}) for p in (0, 1, 2) for o in ("ab", "ba")]
    assert len(set(keys)) == 6, "six repeated measures of one item must be six keys"
    # and the A/B order alone is enough to separate two trials
    assert key({"phrasing": 0, "order": "ab"}) != key({"phrasing": 0, "order": "ba"})


def test_variant_key_is_canonical_not_insertion_ordered():
    a = trial_key("s", "i", "C", {"order": "ab", "phrasing": 0}, "ad", "m", 42, "rev1")
    b = trial_key("s", "i", "C", {"phrasing": 0, "order": "ab"}, "ad", "m", 42, "rev1")
    assert a == b, "dict insertion order must not change the key"


def test_empty_variant_has_a_determinate_form():
    assert canonical_variant({}) == "{}"
    assert canonical_variant(None) == "{}"
    assert (trial_key("s", "i", "C", {}, "ad", "m", 42, "rev1")
            == trial_key("s", "i", "C", None, "ad", "m", 42, "rev1"))
    # but the empty variant is still a different trial from a filled one
    assert (trial_key("s", "i", "C", {}, "ad", "m", 42, "rev1")
            != trial_key("s", "i", "C", VARIANT, "ad", "m", 42, "rev1"))


def test_variant_must_be_a_dict():
    with pytest.raises(TypeError):
        trial_key("s", "i", "C", "phrasing=0", "ad", "m", 42, "rev1")


# --- task D: measurement_rev, not git HEAD ---------------------------------- #
def _fake_repo(tmp_path):
    """A miniature repo with the same layout measurement_rev walks."""
    import shutil
    root = tmp_path / "repo"
    for sub in ("bench/adaptors", "bench/surfaces", "bench/tests", "docs/bench"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in ("bench/types.py", "bench/store.py", "bench/cli.py", "bench/sample.py",
                 "bench/adaptors/local_hf.py", "bench/surfaces/choice.py",
                 "bench/tests/test_store.py", "README.md", "docs/bench/01.md"):
        (root / name).write_text(f"# {name}\n")
    (root / "weights.pkl").write_bytes(b"probe-weights-v1")
    return root


def test_measurement_rev_ignores_files_that_cannot_change_a_measurement(tmp_path):
    root = _fake_repo(tmp_path)
    before = measurement_rev(str(root))
    for irrelevant in ("README.md", "docs/bench/01.md", "bench/cli.py", "bench/sample.py",
                       "bench/tests/test_store.py"):
        (root / irrelevant).write_text("edited\n")
        assert measurement_rev(str(root)) == before, f"{irrelevant} must not invalidate data"


def test_measurement_rev_changes_when_a_surface_changes(tmp_path):
    root = _fake_repo(tmp_path)
    before = measurement_rev(str(root))
    (root / "bench/surfaces/choice.py").write_text("# reworded the question\n")
    assert measurement_rev(str(root)) != before

    after = measurement_rev(str(root))
    (root / "bench/adaptors/local_hf.py").write_text("# different logit read\n")
    assert measurement_rev(str(root)) != after


def test_measurement_rev_covers_the_probe_weights(tmp_path):
    root = _fake_repo(tmp_path)
    weights = str(root / "weights.pkl")
    with_weights = measurement_rev(str(root), extra_files=[weights])
    assert with_weights != measurement_rev(str(root))
    (root / "weights.pkl").write_bytes(b"probe-weights-v2")
    assert measurement_rev(str(root), extra_files=[weights]) != with_weights


def test_measurement_rev_is_short_stable_and_location_independent(tmp_path):
    import shutil
    root = _fake_repo(tmp_path)
    rev = measurement_rev(str(root))
    assert len(rev) == 12 and rev == measurement_rev(str(root))
    moved = tmp_path / "elsewhere"
    shutil.copytree(root, moved)
    assert measurement_rev(str(moved)) == rev
    assert "bench/surfaces/choice.py" in measurement_inputs(str(root))
    assert "bench/cli.py" not in measurement_inputs(str(root))


def test_deleting_a_measurement_file_still_changes_the_rev(tmp_path):
    root = _fake_repo(tmp_path)
    before = measurement_rev(str(root))
    (root / "bench/surfaces/choice.py").unlink()
    assert measurement_rev(str(root)) != before


def test_append_dedups_within_one_store(tmp_path):
    store = RunStore(run_dir=str(tmp_path / "run"), conversations_dir=str(tmp_path / "conv"))
    key = trial_key(*ARGS)
    assert store.append(_record(key)) is True
    assert store.append(_record(key)) is False
    store.close()
    lines = (tmp_path / "run" / "trials.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_resume_across_restart(tmp_path):
    run_dir, conv_dir = str(tmp_path / "run"), str(tmp_path / "conv")
    keys = [trial_key("vote2020", f"item{i}", "C", VARIANT, "local_hf", "m", 42, "rev")
            for i in range(5)]

    first = RunStore(run_dir=run_dir, conversations_dir=conv_dir)
    for key in keys[:3]:
        first.append(_record(key))
    first.close()

    second = RunStore(run_dir=run_dir, conversations_dir=conv_dir)
    assert second.n_done == 3
    assert all(second.has(k) for k in keys[:3])
    assert not any(second.has(k) for k in keys[3:])
    written = sum(1 for key in keys if second.append(_record(key)))
    second.close()
    assert written == 2

    third = RunStore(run_dir=run_dir, conversations_dir=conv_dir)
    assert third.n_done == 5
    assert len(list(third.read())) == 5


def test_torn_last_line_does_not_break_resume(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    key = trial_key(*ARGS)
    with open(run_dir / "trials.jsonl", "w") as handle:
        handle.write(json.dumps(_record(key)) + "\n")
        handle.write('{"trial_key": "sha256:half')  # killed mid-write
    store = RunStore(run_dir=str(run_dir), conversations_dir=str(tmp_path / "conv"))
    assert store.n_done == 1
    assert store.has(key)


def test_conversations_stored_by_sha_and_deduped(tmp_path):
    conv = Conversation(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                        images=["a.jpg"])
    cs = ConversationStore(str(tmp_path))
    sha1 = cs.put(conv)
    sha2 = cs.put(Conversation(messages=list(conv.messages), images=list(conv.images)))
    assert sha1 == sha2 == conv.sha
    assert cs.has(sha1)
    assert cs.get(sha1)["messages"] == conv.messages
    assert sum(1 for _, _, files in os.walk(tmp_path) for _ in files) == 1


def test_manifest_records_provenance(tmp_path):
    store = RunStore(run_dir=str(tmp_path / "run"), conversations_dir=str(tmp_path / "conv"))
    store.append(_record(trial_key(*ARGS)))
    path = store.write_manifest(["bench", "run"], "5d2ce41",
                                {"name": "local_hf", "model": "Qwen3-VL-8B-Instruct"},
                                started_at=0.0, finished_at=1.0,
                                measurement_rev="a1b2c3d4e5f6")
    store.close()
    manifest = json.loads(open(path).read())
    assert manifest["command"] == ["bench", "run"]
    # both are recorded; only measurement_rev decides identity
    assert manifest["code_rev"] == "5d2ce41"
    assert manifest["measurement_rev"] == "a1b2c3d4e5f6"
    assert manifest["adaptor"]["model"] == "Qwen3-VL-8B-Instruct"
    assert manifest["n_trials"] == 1
    assert manifest["started_at"] and manifest["finished_at"]


def test_items_are_write_once(tmp_path):
    path = str(tmp_path / "explore.jsonl")
    assert write_items(path, [{"item_id": "a"}]) == 1
    try:
        write_items(path, [{"item_id": "b"}])
    except FileExistsError:
        return
    raise AssertionError("write_items overwrote a frozen stimulus file")
