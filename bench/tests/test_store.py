"""Store: dedup on trial_key, resume across process restarts, conversations by sha."""

import json
import os

from bench.store import ConversationStore, RunStore, trial_key, write_items
from bench.types import Conversation

ARGS = ("vote2020", "lvis3_00001", "C", "local_hf", "Qwen3-VL-8B-Instruct", 42, "5d2ce41")


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
        mutated[i] = 999 if isinstance(mutated[i], int) else str(mutated[i]) + "x"
        assert trial_key(*mutated) != trial_key(*ARGS), f"field {i} does not affect the key"


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
    keys = [trial_key("vote2020", f"item{i}", "C", "local_hf", "m", 42, "rev") for i in range(5)]

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
                                started_at=0.0, finished_at=1.0)
    store.close()
    manifest = json.loads(open(path).read())
    assert manifest["command"] == ["bench", "run"]
    assert manifest["code_rev"] == "5d2ce41"
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
