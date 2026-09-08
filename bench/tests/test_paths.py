"""The roots, and the property that makes the change safe: defaults are the old layout."""

import os

import pytest

from bench import paths


@pytest.mark.parametrize("fn,env,tail", [
    (paths.images_root, "LPL_IMAGES_ROOT", "items"),
    (paths.data_root, "LPL_DATA_ROOT", ""),
    (paths.runs_root, "LPL_RUNS_ROOT", ""),
    (paths.datasets_root, "LPL_DATASETS_ROOT", "datasets"),
])
def test_an_unset_root_reproduces_the_in_repo_layout(fn, env, tail, monkeypatch):
    """The whole migration rests on this: unset means nothing changed."""
    monkeypatch.delenv(env, raising=False)
    assert fn() == (os.path.join(paths.REPO_ROOT, tail) if tail else paths.REPO_ROOT)


@pytest.mark.parametrize("fn,env", [
    (paths.images_root, "LPL_IMAGES_ROOT"),
    (paths.data_root, "LPL_DATA_ROOT"),
    (paths.runs_root, "LPL_RUNS_ROOT"),
    (paths.datasets_root, "LPL_DATASETS_ROOT"),
])
def test_a_set_root_wins_and_is_absolute(fn, env, monkeypatch, tmp_path):
    monkeypatch.setenv(env, str(tmp_path))
    assert fn() == str(tmp_path)


def test_judge_cache_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("LPL_JUDGE_CACHE", raising=False)
    assert paths.judge_cache_path().endswith(os.path.join("judge_cache", "judge.sqlite"))
    monkeypatch.setenv("LPL_JUDGE_CACHE", str(tmp_path / "j.sqlite"))
    assert paths.judge_cache_path() == str(tmp_path / "j.sqlite")


def test_resolve_image_uses_the_record_name_not_the_frozen_path(monkeypatch):
    monkeypatch.setenv("LPL_IMAGES_ROOT", "/work/nvme/bifr/tzhang30/lpl/shared/images")
    assert paths.resolve_image("train2017/000000000030.jpg") == \
        "/work/nvme/bifr/tzhang30/lpl/shared/images/train2017/000000000030.jpg"


def test_an_absolute_record_name_is_left_alone(monkeypatch):
    """Some older sets stored an absolute path in `images`. Do not mangle it."""
    monkeypatch.setenv("LPL_IMAGES_ROOT", "/somewhere/else")
    assert paths.resolve_image("/already/absolute.jpg") == "/already/absolute.jpg"


# --- the loader is the only way in -------------------------------------------
def test_from_dict_recomputes_image_paths_and_ignores_the_frozen_ones(monkeypatch):
    """The stale absolute paths on disk must not survive the load."""
    from bench.types import Item
    monkeypatch.setenv("LPL_IMAGES_ROOT", "/work/nvme/bifr/tzhang30/lpl/shared/images")
    item = Item.from_dict({
        "item_id": "x", "stratum": 0, "image_scores": [0.1],
        "images": ["train2017/000000000030.jpg"],
        "image_paths": ["/project/jevans/tzhang3/STALE/000000000030.jpg"],
    })
    assert item.image_paths == [
        "/work/nvme/bifr/tzhang30/lpl/shared/images/train2017/000000000030.jpg"]
    assert "STALE" not in item.image_paths[0]


def test_from_dict_leaves_a_set_with_no_record_names_alone(monkeypatch):
    from bench.types import Item
    monkeypatch.setenv("LPL_IMAGES_ROOT", "/anywhere")
    item = Item.from_dict({"item_id": "x", "stratum": 0, "image_scores": [],
                           "images": [], "image_paths": ["/kept/as/is.jpg"]})
    assert item.image_paths == ["/kept/as/is.jpg"]


def test_unset_root_leaves_the_old_behaviour_intact(monkeypatch):
    """With nothing set, a v1 items row resolves under <repo>/items, as before."""
    import os
    from bench.types import Item
    monkeypatch.delenv("LPL_IMAGES_ROOT", raising=False)
    item = Item.from_dict({"item_id": "x", "stratum": 0, "image_scores": [0.0],
                           "images": ["_image_cache/a.jpg"], "image_paths": ["ignored"]})
    assert item.image_paths == [os.path.join(paths.REPO_ROOT, "items", "_image_cache/a.jpg")]


def test_no_module_reads_image_paths_off_a_raw_dict():
    """The guard the plan calls the real deliverable.

    A missing image does not raise -- it produces a plausible answer with fewer
    pixels in it. So the failure this prevents is silent, and the only defence is
    that nothing reaches for the frozen field except the loader.
    """
    import pathlib
    import re
    root = pathlib.Path(paths.REPO_ROOT) / "bench"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        # types.py builds the Item, paths.py defines resolution, sample.py writes the
        # field, and stage_images.py reads it *as the manifest* -- it is the one tool
        # whose whole job is the record-name-to-file mapping, which lives nowhere else.
        if path.name in {"types.py", "paths.py", "sample.py", "stage_images.py"} \
                or "tests" in path.parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""\[["']image_paths["']\]|\.get\(\s*["']image_paths["']""", line):
                offenders.append(f"{path.relative_to(paths.REPO_ROOT)}:{n}")
    assert offenders == [], (
        "these read image_paths off a raw dict instead of going through "
        f"Item.from_dict: {offenders}")
