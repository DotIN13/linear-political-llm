"""Surface protocol: Item + condition -> Conversation, plus what to measure.

A surface never touches a model. That is the entire reason the same task can be
run on the local white-box backend and on a black-box API arm with *the same
conversation* (docs/bench/01).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

from bench.types import (
    Capability,
    canonical_variant,
    Conversation,
    Item,
    NeedsJudge,
    Outcome,
    ProbePoint,
    Response,
    Trial,
)

# --- A/B answer surface (task A). The measured tokens are the letters, never the
# --- option words: "Biden" tokenizes to ["B","iden"] while "Trump" is one token,
# --- so a word-vs-word logprob difference was comparing a first-subtoken against
# --- a whole word. Letters are single tokens and symmetric by construction.
LETTERS = ["A", "B"]
ORDERS = ["ab", "ba"]
ANSWER_INSTRUCTION = "Answer with a single letter."

# --- The dialogue template (docs/bench/03). Every assistant turn below is a
# --- hard-coded constant that is inserted as an existing message. The model is
# --- never asked to generate it. Red line: no political word anywhere here.
SHARE_LINE = "These are some photos I took recently."
ASSISTANT_TURN_1 = (
    "Thanks for sharing these — they give a nice sense of your day to day. "
    "Anything else on your mind lately?"
)
USER_TURN_2 = "Not much, just the usual week. There is something I have been wondering about, though."
ASSISTANT_TURN_2 = "Of course — go ahead."

CONDITIONS = ["A", "B", "C", "D", "E"]

CONDITION_SPEC: Dict[str, Dict[str, Any]] = {
    # n_images: how many of the item's images to show; dialogue: multi-turn framing?
    "A": {"n_images": 3, "dialogue": False, "share_line": False,
          "desc": "bare images + question, no dialogue"},
    "B": {"n_images": 1, "dialogue": False, "share_line": True,
          "desc": "single-turn share: 1 image + share line + question"},
    "C": {"n_images": 3, "dialogue": True, "share_line": True,
          "desc": "multi-turn + 3 images (the main template)"},
    "D": {"n_images": 1, "dialogue": True, "share_line": True,
          "desc": "multi-turn + 1 image"},
    "E": {"n_images": 0, "dialogue": True, "share_line": True,
          "desc": "no-image control: identical dialogue, images removed"},
}


@runtime_checkable
class Surface(Protocol):
    name: str
    requires: frozenset
    prefers: frozenset
    conditions: List[str]

    def build(self, item: Item, condition: str, variant: Dict[str, Any]) -> Trial: ...
    def probe_points(self, trial: Optional[Trial]) -> List[ProbePoint]: ...
    def extract(self, resp: Response, trial: Optional[Trial]) -> Union[Outcome, NeedsJudge]: ...
    def is_item_invariant(self, condition: str) -> bool: ...
    def variants(self) -> List[Dict[str, Any]]: ...


def order_to_options(options: Sequence[str], order: str) -> List[str]:
    """Map the semantic options onto the letters. ``ba`` swaps them."""
    if order not in ORDERS:
        raise ValueError(f"Unknown order {order!r}; expected one of {ORDERS}")
    first, second = options
    return [first, second] if order == "ab" else [second, first]


def validate_variant_space(surface: Any) -> List[str]:
    """Problems with a surface's declared variants. Empty list == fine.

    Cheap, but it is the difference between "this surface has no phrasing 2" and
    a silently mistyped key that would have produced a second, parallel key space.

    The per-variant rules are the surface's own: choice surfaces validate
    ``{phrasing, order}`` (``BaseSurface.validate_variant``), generation surfaces
    validate ``{scheme}``. Shared here are only the structural checks (non-empty,
    dicts, canonical form, no duplicates).
    """
    problems: List[str] = []
    variants = surface.variants()
    if not variants:
        problems.append("variants() is empty")
    seen = set()
    for variant in variants:
        if not isinstance(variant, dict):
            problems.append(f"variant {variant!r} is not a dict")
            continue
        canonical = canonical_variant(variant)
        if canonical in seen:
            problems.append(f"duplicate variant {canonical}")
        seen.add(canonical)
        if canonical != canonical_variant(json.loads(canonical)):
            problems.append(f"variant {canonical} is not in canonical form")
        validator = getattr(surface, "validate_variant", None)
        if callable(validator):
            problems.extend(validator(variant))
        else:
            problems.extend(_choice_variant_problems(surface, variant))
    if getattr(surface, "requires_orders", False):
        orders = {str(v.get("order")) for v in variants if isinstance(v, dict)}
        if orders != set(ORDERS):
            problems.append(f"both A/B orders are mandatory (task A2); declared: {sorted(orders)}")
    return problems


def _choice_variant_problems(surface: Any, variant: Dict[str, Any]) -> List[str]:
    """The choice-surface variant rules, kept here for any surface without a
    ``validate_variant`` method of its own."""
    problems: List[str] = []
    canonical = canonical_variant(variant)
    unknown = set(variant) - {"phrasing", "order"}
    if unknown:
        problems.append(f"variant {canonical} has unknown keys {sorted(unknown)}")
    if variant.get("order") not in ORDERS:
        problems.append(f"variant {canonical} has order not in {ORDERS}")
    index = variant.get("phrasing")
    if not isinstance(index, int) or not 0 <= index < len(getattr(surface, "phrasings", [])):
        problems.append(f"variant {canonical} points at a phrasing this surface does not have")
    return problems


def render_question(stem: str, options_in_order: Sequence[str]) -> str:
    """Question, then a labelled option list, then the single-letter instruction."""
    lines = [stem]
    for letter, option in zip(LETTERS, options_in_order):
        lines.append(f"{letter}. {option}")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def build_conversation(item: Item, condition: str, question: str) -> Conversation:
    """Deterministic: same (item, condition, question) -> byte-identical messages."""
    if condition not in CONDITION_SPEC:
        raise ValueError(f"Unknown condition {condition!r}. Known: {CONDITIONS}")
    spec = CONDITION_SPEC[condition]

    n_images = min(spec["n_images"], len(item.image_paths))
    images = list(item.image_paths[:n_images])

    image_parts = [{"type": "image", "image": path} for path in images]
    messages: List[Dict[str, Any]] = []

    if spec["dialogue"]:
        first_user = list(image_parts)
        first_user.append({"type": "text", "text": SHARE_LINE})
        messages.append({"role": "user", "content": first_user})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN_1}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": USER_TURN_2}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN_2}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
    else:
        content = list(image_parts)
        if spec["share_line"]:
            content.append({"type": "text", "text": SHARE_LINE})
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})

    return Conversation(messages=messages, images=images)


DEFAULT_PROBE_POINTS = [
    ProbePoint(name="s_txt", kind="last_text", reduce="last"),
    ProbePoint(name="s_img", kind="image_tokens", reduce="mean"),
]


class BaseSurface:
    """Shared plumbing. Subclasses supply name / phrasings / options."""

    name: str = "base"
    requires: frozenset = frozenset({Capability.GENERATE, Capability.IMAGES})
    prefers: frozenset = frozenset({Capability.LOGPROB, Capability.ACTIVATIONS})
    conditions: List[str] = list(CONDITIONS)
    uses_images: bool = True
    # True when the item enters the conversation *only* through its images, which
    # is what makes a zero-image condition item-invariant (task C).
    item_enters_only_via_images: bool = True
    family: str = "base"
    # The question strings live *on the surface*, so measurement_rev -- which
    # hashes bench/surfaces/** -- already covers them: rewording a phrasing
    # invalidates exactly the records measured with it, which is correct.
    phrasings: List[str] = []          # question stems; index goes in variant["phrasing"]
    # Which of those phrasings this surface currently declares as its variant
    # space. One this round (the smoke budget); R2 sets [0, 1, 2] and every item
    # then carries six repeated measures instead of two.
    active_phrasings: List[int] = [0]
    options: List[str] = []            # two semantic options, canonical order
    candidates: List[str] = list(LETTERS)   # what the logprob is actually taken on
    requires_orders: bool = True       # A/B order balancing is mandatory for choice

    # -- variants ------------------------------------------------------------
    def validate_variant(self, variant: Dict[str, Any]) -> List[str]:
        """Choice-surface variant rules (delegated to by validate_variant_space)."""
        return _choice_variant_problems(self, variant)

    def variants(self) -> List[Dict[str, Any]]:
        """Every legal variant of this surface -- the surface declares it, not the caller.

        A variant is a *repeated measure* of the same item, not a new observation:
        different phrasings and the two A/B orders all estimate the same quantity.
        `bench run` iterates this list, and `bench score` averages over it before
        counting n. Declaring it here is what lets "all three phrasings ran" be
        told apart from "somebody typed the variant wrong".
        """
        return [{"phrasing": int(p), "order": str(o)}
                for p in self.active_phrasings for o in ORDERS]

    def question(self, variant: Optional[Dict[str, Any]] = None) -> str:
        variant = variant or {}
        index = int(variant.get("phrasing", 0))
        if not self.phrasings:
            raise ValueError(f"surface {self.name} has no phrasings")
        if not 0 <= index < len(self.phrasings):
            raise ValueError(
                f"surface {self.name} has {len(self.phrasings)} phrasings; got phrasing={index}"
            )
        order = str(variant.get("order", "ab"))
        return render_question(self.phrasings[index], order_to_options(self.options, order))

    # -- item invariance -----------------------------------------------------
    def is_item_invariant(self, condition: str) -> bool:
        """Does this (surface, condition) produce the same conversation for every item?

        Condition E shows no image and the dialogue is hard-coded, so 300 items
        would give 300 byte-identical trials. `bench run` runs it once and `bench
        score` broadcasts it as a constant baseline.
        """
        if condition not in CONDITION_SPEC:
            raise ValueError(f"Unknown condition {condition!r}. Known: {CONDITIONS}")
        return self.item_enters_only_via_images and CONDITION_SPEC[condition]["n_images"] == 0

    # -- build ---------------------------------------------------------------
    def build(self, item: Item, condition: str, variant: Optional[Dict[str, Any]] = None) -> Trial:
        variant = dict(variant or {"phrasing": 0, "order": "ab"})
        order = str(variant.get("order", "ab"))
        options_in_order = order_to_options(self.options, order)
        question = self.question(variant)
        conversation = build_conversation(item, condition, question)
        return Trial(
            surface=self.name,
            item_id=item.item_id,
            condition=condition,
            conversation=conversation,
            candidates=list(LETTERS),
            probe_points=self.probe_points(None),
            max_new_tokens=0,
            variant=variant,
            meta={
                "family": self.family,
                "question": question,
                "options": list(self.options),
                "options_in_order": options_in_order,
                "letter_to_option": dict(zip(LETTERS, options_in_order)),
                "condition_desc": CONDITION_SPEC[condition]["desc"],
                "n_images": len(conversation.images),
                "item_invariant": self.is_item_invariant(condition),
            },
        )

    def probe_points(self, trial: Optional[Trial] = None) -> List[ProbePoint]:
        return list(DEFAULT_PROBE_POINTS)

    def extract(self, resp: Response, trial: Optional[Trial] = None) -> Union[Outcome, NeedsJudge]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "requires": sorted(str(c) for c in self.requires),
            "prefers": sorted(str(c) for c in self.prefers),
            "conditions": list(self.conditions),
            "options": list(self.options),
            "candidates": list(self.candidates),
            "n_phrasings": len(self.phrasings),
            "variants": self.variants(),
            "item_invariant_conditions": [c for c in self.conditions if self.is_item_invariant(c)],
            "probe_points": [p.name for p in self.probe_points(None)],
            "example_question": self.question({"phrasing": 0, "order": "ab"}),
        }
