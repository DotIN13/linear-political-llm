"""Where things live, and the single place that decides.

The project used to put every data directory inside the repo -- ``ROOT/items``,
``ROOT/runs``, ``ROOT/judge_cache`` -- all of them gitignored, so they were data
the code merely happened to keep in the working tree. That is fine with one
checkout on one machine. It stops being fine the moment you want a second
machine or a second worktree, because each one then needs its own copy of the
image set.

So each root is an environment variable **whose default reproduces the old
layout exactly**. Nothing changes for an existing checkout; a new machine sets
five variables and shares one copy of the data between every worktree.

    LPL_IMAGES_ROOT     resized stimulus images -- what items resolve against
    LPL_DATA_ROOT       holds items/            (read-only in every job)
    LPL_RUNS_ROOT       holds runs/ conversations/ results/   (written)
    LPL_JUDGE_CACHE     the judge sqlite        (shared on purpose)
    LPL_DATASETS_ROOT   raw COCO/LVIS -- only bench/sample.py reads this

On Delta these point at::

    /work/nvme/bifr/tzhang30/datasets              LPL_DATASETS_ROOT
    /work/nvme/bifr/tzhang30/lpl/shared/images     LPL_IMAGES_ROOT
    /work/nvme/bifr/tzhang30/lpl/shared            LPL_DATA_ROOT
    /work/nvme/bifr/tzhang30/lpl/shared/runs       LPL_RUNS_ROOT
    /work/nvme/bifr/tzhang30/lpl/shared/judge_cache/judge.sqlite

Models and containers are deliberately absent: the sbatch scripts already take
them as ``SIF=`` and ``MODELS=``, so hoisting those is a matter of pointing the
existing overrides somewhere else, not of new Python.
"""

from __future__ import annotations

import os
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root(env: str, *default_parts: str) -> str:
    value = os.environ.get(env)
    return os.path.abspath(value) if value else os.path.join(REPO_ROOT, *default_parts)


def images_root() -> str:
    """Where the resized stimulus images are.

    Defaults to ``<repo>/items`` because that is where the absolute paths frozen
    into existing items files point -- so an unset variable reproduces the old
    behaviour byte for byte.
    """
    return _root("LPL_IMAGES_ROOT", "items")


def data_root() -> str:
    """Holds ``items/``. Read-only in every job."""
    return _root("LPL_DATA_ROOT")


def runs_root() -> str:
    """Holds ``runs/``, ``conversations/``, ``results/``. Everything a job writes."""
    return _root("LPL_RUNS_ROOT")


def datasets_root() -> str:
    """Raw COCO/LVIS. Only ``bench/sample.py`` reads this -- a normal run never does."""
    return _root("LPL_DATASETS_ROOT", "datasets")


def items_dir() -> str:
    return os.path.join(data_root(), "items")


def runs_dir(name: Optional[str] = None) -> str:
    base = os.path.join(runs_root(), "runs")
    return os.path.join(base, name) if name else base


def conversations_dir() -> str:
    return os.path.join(runs_root(), "conversations")


def judge_cache_path() -> str:
    value = os.environ.get("LPL_JUDGE_CACHE")
    if value:
        return os.path.abspath(value)
    return os.path.join(REPO_ROOT, "judge_cache", "judge.sqlite")


def resolve_image(record_name: str) -> str:
    """``train2017/000000000030.jpg`` -> an absolute path under the images root.

    **This is the function that makes the project portable.** Items on disk carry
    two fields: ``images`` (record names) and ``image_paths`` ("resolved on-disk
    paths" -- its own comment). The resolved ones were written by whichever
    machine sampled the set, so they are wrong everywhere else, and they are the
    reason a worktree could not share an image directory.

    Record names are stable. Resolve them here, at load, and one items file is
    correct on midway, on Delta, and in ten worktrees at once.
    """
    if os.path.isabs(record_name):
        return record_name
    return os.path.join(images_root(), record_name)


def image_paths_of(row) -> list:
    """The resolved image paths for one items-file **row** (a dict, not an Item).

    Same rule as ``Item.from_dict``: prefer the record names, fall back to the
    frozen field only when a set has none. Pilots reach into rows directly all
    over the place, and every one of those was a place the project silently
    pointed at nothing the moment the images moved -- a missing image does not
    raise, it just produces an answer with fewer pixels behind it.

    ``test_no_module_reads_image_paths_off_a_raw_dict`` keeps it that way.
    """
    records = (row.get("images") if hasattr(row, "get") else None) or []
    if records:
        return [resolve_image(r) for r in records]
    return list((row.get("image_paths") if hasattr(row, "get") else None) or [])
