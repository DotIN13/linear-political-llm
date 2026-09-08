"""Stage an items file's images into the record-name layout the loader expects.

**Why this exists, and it is not obvious.** An items file carries two parallel
lists per row::

    images      ["train2017/000000363606.jpg", …]     record names, stable
    image_paths [".../_resized_images_800/7dd68ba8….jpg", …]   what is on disk

The files on disk are named by a **content hash**, not by the record name, and
the hash is not derivable from the record name -- I checked: sha1 of the record
name, of its basename, and of the stem all fail to match. So the mapping between
"the image this item means" and "the file holding it" exists in exactly one
place: **the items file itself**. It is the manifest, and nothing else is.

That matters because ``bench.paths.resolve_image`` resolves a record name
against ``LPL_IMAGES_ROOT`` -- it assumes record name *is* the relative path. On
midway that assumption was carried by the frozen ``image_paths``; on any new
machine it has to be made true, by copying each file to the name its record
claims. That is all this does.

**Copy the images without renaming and every lookup misses silently** -- a
missing image does not raise, it produces an answer with fewer pixels behind it.
So the verify pass at the end is the point of the script, not a courtesy.

    python -m bench.tools.stage_images \\
        --items  explore_bucket_v1.jsonl \\
        --source /path/to/_resized_images_800 \\
        --dest   /work/nvme/bifr/tzhang30/lpl/shared/images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple


def manifest(items_path: str) -> Dict[str, str]:
    """``record name -> the basename on disk``, from the items file.

    Raises if one record name maps to two different files: that would mean the
    manifest disagrees with itself and no staging of it can be correct.
    """
    out: Dict[str, str] = {}
    with open(items_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records, paths = row.get("images") or [], row.get("image_paths") or []
            if len(records) != len(paths):
                raise ValueError(
                    f"{row.get('item_id')}: {len(records)} record names but "
                    f"{len(paths)} paths -- the two lists are meant to be parallel")
            for rec, path in zip(records, paths):
                base = os.path.basename(path)
                if rec in out and out[rec] != base:
                    raise ValueError(f"{rec} maps to both {out[rec]} and {base}")
                out[rec] = base
    return out


def stage(items_path: str, source: str, dest: str,
          link: bool = False, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Copy each file to ``dest/<record name>``. Returns (staged, missing)."""
    todo = manifest(items_path)
    missing: List[str] = []
    staged = 0
    for rec, base in sorted(todo.items()):
        src = os.path.join(source, base)
        if not os.path.exists(src):
            missing.append(f"{rec} -> {base}")
            continue
        dst = os.path.join(dest, rec)
        if dry_run:
            staged += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        if link:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        staged += 1
    return staged, missing


def verify(items_path: str, dest: str) -> List[str]:
    """Every record name in the items file must resolve to a real file under dest.

    This is the check the whole exercise is for. It is deliberately independent
    of ``stage`` -- it re-reads the manifest and looks at the filesystem, so it
    would catch a staging run that silently did nothing.
    """
    absent = []
    for rec in sorted(manifest(items_path)):
        if not os.path.isfile(os.path.join(dest, rec)):
            absent.append(rec)
    return absent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True, help="the items jsonl -- the manifest")
    p.add_argument("--source", help="directory holding the hash-named files")
    p.add_argument("--dest", required=True, help="images root to stage into")
    p.add_argument("--link", action="store_true", help="symlink instead of copy")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-only", action="store_true",
                   help="skip staging; just check dest against the manifest")
    a = p.parse_args(argv)

    if not a.verify_only:
        if not a.source:
            p.error("--source is required unless --verify-only")
        staged, missing = stage(a.items, a.source, a.dest, a.link, a.dry_run)
        verb = "would stage" if a.dry_run else ("linked" if a.link else "copied")
        print(f"[stage] {verb} {staged}")
        if missing:
            print(f"[stage] {len(missing)} source files absent:", file=sys.stderr)
            for m in missing[:10]:
                print(f"          {m}", file=sys.stderr)
            if len(missing) > 10:
                print(f"          … and {len(missing) - 10} more", file=sys.stderr)
            return 1
        if a.dry_run:
            return 0

    absent = verify(a.items, a.dest)
    total = len(manifest(a.items))
    if absent:
        print(f"[verify] {len(absent)} of {total} record names do not resolve under "
              f"{a.dest}:", file=sys.stderr)
        for x in absent[:10]:
            print(f"           {x}", file=sys.stderr)
        return 1
    print(f"[verify] all {total} record names resolve under {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
