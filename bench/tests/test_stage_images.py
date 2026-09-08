"""Staging is the step that makes record-name resolution true on a new machine.

The files on disk are hash-named and the hash is not derivable from the record
name, so the items file is the only manifest there is. Copy without renaming and
every lookup misses -- silently, because a missing image produces an answer with
fewer pixels rather than an error.
"""

import json
import os

import pytest

from bench.tools.stage_images import manifest, stage, verify


def _items(tmp_path, rows):
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def _row(item_id, pairs):
    return {"item_id": item_id,
            "images": [r for r, _ in pairs],
            "image_paths": [f"results/whatever/_resized_images_800/{h}" for _, h in pairs]}


def test_manifest_maps_record_name_to_the_file_on_disk(tmp_path):
    p = _items(tmp_path, [_row("a", [("train2017/000000000030.jpg", "abc123.jpg")])])
    assert manifest(p) == {"train2017/000000000030.jpg": "abc123.jpg"}


def test_parallel_lists_of_different_length_are_a_hard_error(tmp_path):
    p = _items(tmp_path, [{"item_id": "a", "images": ["x.jpg", "y.jpg"],
                           "image_paths": ["dir/h1.jpg"]}])
    with pytest.raises(ValueError, match="parallel"):
        manifest(p)


def test_one_record_mapping_to_two_files_is_a_hard_error(tmp_path):
    p = _items(tmp_path, [_row("a", [("same.jpg", "h1.jpg")]),
                          _row("b", [("same.jpg", "h2.jpg")])])
    with pytest.raises(ValueError, match="maps to both"):
        manifest(p)


def test_staging_renames_hash_named_files_to_their_record_names(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "abc123.jpg").write_bytes(b"pixels")
    p = _items(tmp_path, [_row("a", [("train2017/000000000030.jpg", "abc123.jpg")])])

    staged, missing = stage(p, str(src), str(dest))
    assert (staged, missing) == (1, [])
    out = dest / "train2017" / "000000000030.jpg"
    assert out.is_file() and out.read_bytes() == b"pixels"
    assert verify(p, str(dest)) == []


def test_a_missing_source_file_is_reported_not_skipped(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    p = _items(tmp_path, [_row("a", [("train2017/x.jpg", "nope.jpg")])])
    staged, missing = stage(p, str(src), str(dest))
    assert staged == 0 and missing == ["train2017/x.jpg -> nope.jpg"]


def test_verify_catches_a_staging_run_that_did_nothing(tmp_path):
    """The check that matters: independent of stage(), so an empty dest fails."""
    dest = tmp_path / "dest"
    dest.mkdir()
    p = _items(tmp_path, [_row("a", [("train2017/x.jpg", "h.jpg")])])
    assert verify(p, str(dest)) == ["train2017/x.jpg"]


def test_staged_images_resolve_through_the_real_loader(tmp_path, monkeypatch):
    """End to end: stage, then ask bench.paths where the image is."""
    from bench.paths import resolve_image
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "deadbeef.jpg").write_bytes(b"x")
    p = _items(tmp_path, [_row("a", [("train2017/000000363606.jpg", "deadbeef.jpg")])])
    stage(p, str(src), str(dest))

    monkeypatch.setenv("LPL_IMAGES_ROOT", str(dest))
    assert os.path.isfile(resolve_image("train2017/000000363606.jpg"))
