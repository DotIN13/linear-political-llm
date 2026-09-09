"""The generation build every task repeats: conditions, schemes, transcript, meta.

A pilot owns its prompt, its questions and its dependent variable. Everything
between those -- the photos/no_photos branch, the chat/agentic transcript, the
per-trial order, the probe points and the ``meta`` dict -- is identical for all
of them, so it lives here once instead of in fourteen copies that can drift.

``build_base`` is a byte-for-byte reimplementation of the old
``GenerationSurface.build`` (``bench/surfaces/shared/surface.py``); the parity
tests build the same trial through both and compare, which is what keeps it so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from bench_v2.helpers.transcript import build_scheme_messages
from bench_v2.types import Conversation, Item, ProbePoint, Trial

CONDITIONS = ("photos", "no_photos")
SCHEMES = ("chat", "agentic")
CONDITION_DESC = {
    "photos": "the persona's photos, delivered per conversation scheme",
    "no_photos": "the question on its own, no persona framing -- once per question",
}

PROBE_POINTS = [
    ProbePoint(name="s_pre", kind="prefix_end", reduce="last"),
    ProbePoint(name="s_gen", kind="generated_tokens", reduce="mean"),
    ProbePoint(name="s_img", kind="image_tokens", reduce="mean"),
]

QuestionFn = Callable[[str, list[int] | None, str], str]
OrderFn = Callable[[Item, int], list[int]]
MetaExtra = Callable[[dict[str, Any], list[int] | None, str], dict[str, Any]]


def build_base(
    *,
    surface: str,
    item: Item,
    condition: str,
    variant: dict[str, Any] | None = None,
    seed: int = 42,
    question_fn: QuestionFn,
    max_new_tokens: int,
    family: str = "generation",
    prefill: str | None = None,
    judge: str | None = None,
    randomizes_per_item: bool = False,
    item_order_fn: OrderFn | None = None,
    meta_extra: MetaExtra | None = None,
    conditions: Sequence[str] = CONDITIONS,
    schemes: Sequence[str] = SCHEMES,
) -> Trial:
    """Build one generation trial, exactly as the bench surface did."""
    if condition not in conditions:
        raise ValueError(f"unknown condition {condition!r}; expected {tuple(conditions)}")
    variant = dict(variant or {"scheme": "chat"})
    scheme = str(variant.get("scheme", "chat"))
    qid = str(variant.get("question", "q0"))
    variant["question"] = qid
    if "rep" in variant:
        variant["rep"] = int(variant["rep"])
    attribution = str(variant.get("attribution", "shown"))
    variant["attribution"] = attribution
    clause = str(variant.get("clause", "bare"))
    variant["clause"] = clause

    pinned = variant.get("order")
    if pinned is not None:
        order: list[int] | None = [int(x) for x in pinned]
    elif randomizes_per_item and item_order_fn is not None:
        order = item_order_fn(item, seed)
    else:
        order = None
    if order is not None:
        variant["order"] = order

    with_images = condition == "photos"
    image_paths = list(item.image_paths) if with_images else []
    question = question_fn(qid, order, attribution)
    if condition == "no_photos":
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        tools = None
        n_files = 0
    else:
        n_files = len(item.image_paths) or 3
        messages, tools = build_scheme_messages(scheme, image_paths, question, n_files, clause)

    meta: dict[str, Any] = {
        "family": family,
        "scheme": scheme,
        "clause": clause,
        "question_id": qid,
        "prefill": prefill,
        "question": question,
        "tools": tools,
        "prefix_n_messages": len(messages) - 1,
        "condition_desc": CONDITION_DESC[condition],
        "n_images": len(image_paths),
        "n_files": 0 if condition == "no_photos" else n_files,
        "item_invariant": condition == "no_photos" and not randomizes_per_item,
        "scheme_invariant": condition == "no_photos",
        "judge": judge,
    }
    if meta_extra is not None:
        meta.update(meta_extra(variant, order, qid))

    return Trial(
        surface=surface,
        item_id=item.item_id,
        condition=condition,
        conversation=Conversation(messages=messages, images=image_paths),
        candidates=[],
        probe_points=list(PROBE_POINTS),
        max_new_tokens=max_new_tokens,
        variant=variant,
        meta=meta,
    )
