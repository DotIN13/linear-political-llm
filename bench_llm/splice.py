"""Splicing image tokens between images, at the vision encoder's output.

A vision-language model turns an image into a sequence of image tokens laid out on a grid
(raster order: row by row). This module reads those tokens from one image and writes a
spliced sequence in their place while another image runs, so the language model sees, say,
one photo's person with another photo's background, while every pixel the encoder saw was
real and unedited.

The pieces, from the model outwards:

* ``VisionTap`` -- wraps a model's vision function. ``capture(encoded)`` runs one forward
  pass and returns that image's tokens; ``override(tokens)`` is a context manager in which
  the vision function returns the given tokens instead of its own (generation, extra
  forward passes and probes all see them). Qwen3-VL's early-layer encoder features
  ("deepstack") are captured and overridden alongside, token for token.
* ``cell_fractions`` / ``infer_grid`` -- which cells of the token grid are person, from a
  person mask of the image.
* ``splice_plan`` -- pure Python: for each token, where it comes from (the base image, the
  donor image, or the mean of one image's background tokens). Tested without a GPU.
* ``apply_plan`` -- builds the spliced tokens from two captures and a plan.

Two images can only be spliced if their grids match, so both should be framed to the same
size first (``bench_llm/datasets/face_swap`` frames everything to 600 x 800).

    tap = VisionTap(model, "qwen3-vl")
    base = tap.capture(encoded_p2)               # the image that will run
    donor = tap.capture(encoded_p1)              # where background tokens come from
    h, w = infer_grid(base.n, aspect=800 / 600)
    plan = splice_plan(person_cells(mask_p2, h, w), person_cells(mask_p1, h, w), "swap")
    with tap.override(apply_plan(base, donor, plan)):
        ... run encoded_p2 as usual ...
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# family -> (attribute path to the object, name of the function to wrap)
VISION_FUNCTIONS: Dict[str, Tuple[str, str]] = {
    "qwen3-vl": ("model.visual", "forward"),
    "gemma4": ("model", "get_image_features"),
}

MODES = ("normal", "swap", "fill_check")


# --- the grid ----------------------------------------------------------------
def infer_grid(n: int, aspect: float) -> Tuple[int, int]:
    """The (rows, cols) with rows * cols == n whose rows / cols is closest to ``aspect``
    (height over width). Raises if n has no factorisation within 15% of the aspect."""
    best = None
    for h in range(1, n + 1):
        if n % h:
            continue
        w = n // h
        err = abs(math.log((h / w) / aspect))
        if best is None or err < best[0]:
            best = (err, h, w)
    if best is None or best[0] > math.log(1.15):
        raise ValueError(f"{n} image tokens do not form a grid of aspect {aspect:.2f}")
    return best[1], best[2]


def cell_fractions(mask: Any, h: int, w: int) -> List[float]:
    """Share of each grid cell covered by the mask (a 2-D boolean array), raster order."""
    import numpy as np
    m = np.asarray(mask, dtype=np.float32)
    H, W = m.shape
    ys = (np.arange(H) * h // H)
    xs = (np.arange(W) * w // W)
    out = []
    for i in range(h):
        rows = m[ys == i]
        for j in range(w):
            out.append(float(rows[:, xs == j].mean()) if rows.size else 0.0)
    return out


def person_cells(mask: Any, h: int, w: int, threshold: float = 0.5) -> List[bool]:
    """True where a cell is mostly person."""
    return [f > threshold for f in cell_fractions(mask, h, w)]


# --- the plan ----------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    """Where one spliced token comes from: ``kind`` is base | donor | base_mean | donor_mean."""

    kind: str
    index: Optional[int] = None


def splice_plan(base_person: Sequence[bool], donor_person: Sequence[bool], mode: str) -> List[Source]:
    """Per token, its source.

    * ``normal``: every token from the base image.
    * ``swap``: base person cells stay; base background cells take the donor's token at the
      same cell, or, where the donor's own person sat, the mean of the donor's background tokens.
    * ``fill_check``: the base image, with exactly the cells ``swap`` would fill from the
      donor's mean set to the mean of the base's own background tokens; isolates what the
      filling does on its own.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if len(base_person) != len(donor_person):
        raise ValueError("base and donor grids differ in size")
    plan = []
    for i, (bp, dp) in enumerate(zip(base_person, donor_person)):
        if mode == "normal" or bp:
            plan.append(Source("base", i))
        elif mode == "swap":
            plan.append(Source("donor", i) if not dp else Source("donor_mean"))
        else:  # fill_check
            plan.append(Source("base", i) if not dp else Source("base_mean"))
    return plan


def plan_counts(plan: Sequence[Source]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in plan:
        out[s.kind] = out.get(s.kind, 0) + 1
    return out


# --- tokens ------------------------------------------------------------------
@dataclass
class Tokens:
    """One image's tokens at the encoder's output: ``main`` [n, d] and any deepstack layers."""

    main: Any
    deepstack: List[Any] = field(default_factory=list)
    shape: Optional[Tuple[int, ...]] = None      # main's shape as the model returned it

    @property
    def n(self) -> int:
        return int(self.main.shape[0])


def _mean(t: Any, rows: Sequence[int]) -> Any:
    import torch
    idx = torch.tensor(list(rows), device=t.device, dtype=torch.long)
    return t.index_select(0, idx).float().mean(0).to(t.dtype)


def _apply_one(base: Any, donor: Any, plan: Sequence[Source], base_bg: Sequence[int], donor_bg: Sequence[int]) -> Any:
    out = base.clone()
    b_mean = _mean(base, base_bg) if any(s.kind == "base_mean" for s in plan) else None
    d_mean = _mean(donor, donor_bg) if any(s.kind == "donor_mean" for s in plan) else None
    for i, s in enumerate(plan):
        if s.kind == "donor":
            out[i] = donor[s.index]
        elif s.kind == "donor_mean":
            out[i] = d_mean
        elif s.kind == "base_mean":
            out[i] = b_mean
    return out


def apply_plan(base: Tokens, donor: Tokens, plan: Sequence[Source],
               base_person: Sequence[bool], donor_person: Sequence[bool]) -> Tokens:
    """The spliced tokens; the means are over each image's own background cells."""
    if base.n != len(plan) or donor.n != len(plan):
        raise ValueError(f"plan has {len(plan)} tokens, images have {base.n} and {donor.n}")
    base_bg = [i for i, p in enumerate(base_person) if not p]
    donor_bg = [i for i, p in enumerate(donor_person) if not p]
    main = _apply_one(base.main, donor.main, plan, base_bg, donor_bg)
    ds = [_apply_one(b, d, plan, base_bg, donor_bg) for b, d in zip(base.deepstack, donor.deepstack)]
    return Tokens(main=main, deepstack=ds, shape=base.shape)


# --- the model ---------------------------------------------------------------
def _resolve(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class VisionTap:
    """Reads and overrides the tokens a model's vision function returns."""

    def __init__(self, model: Any, family: str) -> None:
        if family not in VISION_FUNCTIONS:
            raise ValueError(f"no vision function known for {family!r}; add it to VISION_FUNCTIONS")
        path, name = VISION_FUNCTIONS[family]
        self.model, self.family = model, family
        self.owner = _resolve(model, path)
        self.name = name
        self.original = getattr(self.owner, name)

    @staticmethod
    def _read(out: Any) -> Tokens:
        main = out.pooler_output if hasattr(out, "pooler_output") else out
        shape = tuple(main.shape)
        flat = main.reshape(-1, shape[-1])
        ds = list(getattr(out, "deepstack_features", None) or [])
        return Tokens(main=flat.detach().clone(), deepstack=[d.reshape(-1, d.shape[-1]).detach().clone() for d in ds],
                      shape=shape)

    @staticmethod
    def _write(out: Any, tokens: Tokens) -> Any:
        if hasattr(out, "pooler_output"):
            ref = out.pooler_output
            out.pooler_output = tokens.main.to(ref.device, ref.dtype).reshape(ref.shape)
            if tokens.deepstack and getattr(out, "deepstack_features", None):
                out.deepstack_features = [t.to(d.device, d.dtype).reshape(d.shape)
                                          for t, d in zip(tokens.deepstack, out.deepstack_features)]
            return out
        return tokens.main.to(out.device, out.dtype).reshape(out.shape)

    @contextlib.contextmanager
    def _wrapped(self, fn) -> Iterator[None]:
        setattr(self.owner, self.name, fn)
        try:
            yield
        finally:
            setattr(self.owner, self.name, self.original)

    def capture(self, encoded: Dict[str, Any]) -> Tokens:
        """Run one forward pass on ``encoded`` and return the image tokens the encoder produced."""
        import torch
        got: List[Tokens] = []
        orig = self.original

        def spy(*args, **kwargs):
            out = orig(*args, **kwargs)
            got.append(self._read(out))
            return out

        with self._wrapped(spy), torch.no_grad():
            self.model(**encoded)
        if len(got) != 1:
            raise RuntimeError(f"expected one call to the vision function, saw {len(got)}")
        return got[0]

    @contextlib.contextmanager
    def override(self, tokens: Tokens) -> Iterator[None]:
        """While active, the vision function returns ``tokens`` (its own output is computed and
        discarded, so every shape and side field stays as the model expects)."""
        orig = self.original

        def swapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            ref = out.pooler_output if hasattr(out, "pooler_output") else out
            if ref.reshape(-1, ref.shape[-1]).shape[0] != tokens.n:
                raise RuntimeError(f"override has {tokens.n} tokens, the image has {ref.reshape(-1, ref.shape[-1]).shape[0]}")
            return self._write(out, tokens)

        with self._wrapped(swapped):
            yield
