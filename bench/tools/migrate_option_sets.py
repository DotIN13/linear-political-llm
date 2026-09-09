"""Move the six option sets into the layout ``shared/prompts.py`` expects.

The six sets for s9-s14 were written while ``bench/surfaces/`` was mid-refactor,
so they live in ``bench/data/`` as one jsonl each with a ``{"record":"meta"}``
first line. That was the right place at the time and is the wrong place now, for
one reason:

**Under the current hashing policy they are not hashed at all.**
``MEASUREMENT_GLOBS`` covers ``bench/surfaces/**/*.jsonl`` and
``bench/surfaces/**/*.json``; ``MEASUREMENT_FILES`` names the two golden
snapshots. ``bench/data/`` is in neither. So today an edited option changes no
``trial_key``, and the only thing standing between a hand-edited coding field and
a dependent variable that quietly means something else is
``bench/tests/test_new_surface_datasets.py``.

Once a set sits in ``tasks/<task>/prompts/`` it is covered twice over, which is
the property s3's pool now has too: by content, because the pool file is inside
``bench/surfaces/**/*.jsonl``, and by behaviour, because rendering it changes the
golden snapshot and that snapshot is a measurement file. (An earlier version of
this file said s3 was covered only by the snapshot. That was true while
``s3_headlines_v2.json`` still sat in ``bench/data/``; it moved with the rest of
the restructure, so it is hashed by content as well.)

``prompts.pool()`` wants the rows and the header in two files and the ask out of
the header entirely, so this is a reshaping rather than a copy::

    bench/data/s9_neighborhoods_v1.jsonl
      -> tasks/s9_neighborhood/prompts/ask.txt                 <- meta["prompt"]
         tasks/s9_neighborhood/prompts/neighborhoods.jsonl     <- the option rows
         tasks/s9_neighborhood/prompts/neighborhoods.meta.json <- everything else

One ask per surface becomes ``ask.txt``. **Two or more become one ``.txt``
each**, named ``ask_<qid>.txt`` -- not a json map, because the convention
reserves json for text whose *trailing* whitespace carries meaning, and a
constant ask has none. One file per ask also means a reworded scenario shows up
in a diff at file granularity and is hashed on its own. s13's ask is a template
over the bug's symptom, so it becomes ``ask.j2``.

**Row order is preserved exactly**, because the surfaces fingerprint a pool by
hashing row order and that fingerprint goes into every trial's
``meta["dataset"]``.

Usage -- stage it anywhere to look at it first, then write it in place::

    python -m bench.tools.migrate_option_sets --into /tmp/staged
    python -m bench.tools.migrate_option_sets --check /tmp/staged
    python -m bench.tools.migrate_option_sets              # in place

``--check`` loads the result back through ``shared/prompts.py`` -- the real
loaders, not anything in this file -- and asserts every row is byte-identical to
``bench/data/`` including order. Run it after the move, not instead of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "bench", "data")
TASKS = os.path.join(ROOT, "bench", "surfaces", "tasks")

# source file -> (task package, pool name, how the ask is stored)
#
# The pool name is the noun the surface will pass to prompts.pool(), so it is the
# thing being chosen rather than the surface id: `neighborhoods`, not `s9`.
PLAN: Dict[str, Tuple[str, str, str]] = {
    "s9_neighborhoods_v1.jsonl":      ("s9_neighborhood", "neighborhoods", "txt"),
    "s10_grocery_platforms_v1.jsonl": ("s10_groceries",   "platforms",     "txt"),
    "s11_health_options_v1.jsonl":    ("s11_health",      "options",       "txt:scenarios"),
    "s12_explain_points_v1.jsonl":    ("s12_explain",     "points",        "txt:topics"),
    "s13_patch_choice_v1.jsonl":      ("s13_patch",       "patches",       "j2"),
    "s14_outfits_v1.jsonl":           ("s14_outfits",     "outfits",       "txt"),
}


def _read(fname: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = [json.loads(line) for line in
            open(os.path.join(DATA, fname), encoding="utf-8") if line.strip()]
    if not rows or rows[0].get("record") != "meta":
        raise ValueError(f"{fname}: first line is not the meta record")
    return rows[0], rows[1:]


def convert(into: str) -> List[str]:
    written: List[str] = []
    for fname, (task, pool_name, ask_kind) in PLAN.items():
        meta, options = _read(fname)
        out = os.path.join(into, task, "prompts")
        os.makedirs(out, exist_ok=True)
        header = {k: v for k, v in meta.items() if k != "record"}

        if ask_kind == "txt":
            # text() removes exactly one trailing newline, so write exactly one.
            with open(os.path.join(out, "ask.txt"), "w", encoding="utf-8") as fh:
                fh.write(header.pop("prompt") + "\n")
            written.append(f"{task}/prompts/ask.txt")
        elif ask_kind.startswith("txt:"):
            key = ask_kind.split(":", 1)[1]
            for qid, cfg in header[key].items():
                with open(os.path.join(out, f"ask_{qid}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(cfg["prompt"] + "\n")
                written.append(f"{task}/prompts/ask_{qid}.txt")
            # The asks now live in their own files. The design notes beside them
            # stay in the header; duplicating a prompt into both would let the two
            # copies drift, which is the whole reason s8 imports s5's ask.
            header[key] = {qid: {k: v for k, v in cfg.items() if k != "prompt"}
                           for qid, cfg in header[key].items()}
        elif ask_kind == "j2":
            tmpl = header.pop("prompt_template").replace("{symptom}", "{{ symptom }}")
            with open(os.path.join(out, "ask.j2"), "w", encoding="utf-8") as fh:
                fh.write(tmpl + "\n")
            written.append(f"{task}/prompts/ask.j2")
        else:
            raise ValueError(f"unknown ask kind {ask_kind!r}")

        with open(os.path.join(out, f"{pool_name}.jsonl"), "w", encoding="utf-8") as fh:
            for option in options:                      # file order, deliberately
                fh.write(json.dumps({k: v for k, v in option.items() if k != "record"},
                                    ensure_ascii=False) + "\n")
        header["n_rows"] = len(options)
        header["migrated_from"] = f"bench/data/{fname}"
        with open(os.path.join(out, f"{pool_name}.meta.json"), "w", encoding="utf-8") as fh:
            json.dump(header, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        written += [f"{task}/prompts/{pool_name}.jsonl",
                    f"{task}/prompts/{pool_name}.meta.json"]

        # prompts.pool() locates prompts/ from the caller's __file__, so the task
        # has to be a package for the surface module to sit beside its own data.
        init = os.path.join(into, task, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
            written.append(f"{task}/__init__.py")
    return written


def check(into: str) -> bool:
    """Load it back through the real loaders and compare against bench/data."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from bench.surfaces.shared import prompts

    ok = True
    for fname, (task, pool_name, ask_kind) in PLAN.items():
        entry = os.path.join(into, task, "__init__.py")
        try:
            # read_pool, not pool: this is somebody else's pool from here, and
            # that is exactly the case read_pool(path) exists for. The surface
            # itself will call pool(__file__, name).
            rows, meta = prompts.read_pool(
                os.path.join(into, task, "prompts", f"{pool_name}.jsonl"))
            assert meta.get("version"), "the header has no version"
            assert meta["n_rows"] == len(rows), "n_rows disagrees with the row count"
            if ask_kind == "txt":
                ask = prompts.text(entry)
                assert ask and not ask.endswith("\n"), "ask.txt kept a trailing newline"
            elif ask_kind.startswith("txt:"):
                qids = sorted(meta[ask_kind.split(":", 1)[1]])
                assert len(qids) >= 2, "expected several asks"
                for qid in qids:
                    one = prompts.text(entry, f"ask_{qid}.txt")
                    assert one and not one.endswith("\n"), f"ask_{qid}.txt kept a newline"
            else:
                out = prompts.template(entry, "ask.j2").render(symptom="x")
                assert "{{" not in out, "the template did not render"
            _, options = _read(fname)
            expect = [{k: v for k, v in o.items() if k != "record"} for o in options]
            assert expect == rows, "rows differ from bench/data (order included)"
            print(f"  OK  {task:18s} {len(rows):2d} rows, ask loads, rows byte-identical")
        except Exception as exc:                          # noqa: BLE001 -- reported, not raised
            ok = False
            print(f"  XX  {task:18s} {type(exc).__name__}: {exc}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--into", default=TASKS, help="where to write (default: in place)")
    parser.add_argument("--check", metavar="DIR", help="verify an already-written tree and exit")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check(args.check) else 1)
    for path in convert(args.into):
        print("  wrote", path)
    print()
    raise SystemExit(0 if check(args.into) else 1)
