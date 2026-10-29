"""face_swap v4 -- the best-token swap across the whole portrait, not only the face.

v3 with one change: the cells that can move are every cell of the 600 x 800 frame (background,
hair, clothes and the grey outside the photo included), not only the cells that are person in
both photos. The same 20 pairs of aligned faces, both directions, the same six shares, top-ranked
against random; the ranking is still the donor's own image-token probe score, so the top cells
may now be background. Each trial records how many moved cells were person in the base and in
the donor.

    python -m bench_llm.tasks.face_swap.v4.pilot plan
    python -m bench_llm.tasks.face_swap.v4.pilot run --model qwen3_vl
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_llm import sources
from bench_llm.tasks.face_swap.v3 import pilot as v3
from bench_llm.types import Item

TASK = "face_swap/v4"
TITLE = "best-token swap across the whole portrait"
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "face_swap" / "v4"
CONDITIONS = v3.CONDITIONS
SHARES, HOWS, DIRECTIONS = v3.SHARES, v3.HOWS, v3.DIRECTIONS
REGION = "all"


def load_items(root: Optional[str] = None) -> List[Item]:
    return v3.load_items(root, region=REGION)


variants, build, read, summarize = v3.variants, v3.build, v3.read, v3.summarize


def main(argv: Optional[List[str]] = None) -> int:
    return v3.main(argv, task=TASK, title=TITLE, out_dir=OUT_DIR, region=REGION)


if __name__ == "__main__":
    raise SystemExit(main())
