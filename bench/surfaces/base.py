"""Surface protocol: Item + condition -> Conversation, plus what to measure.

A surface never touches a model. That is the entire reason the same task can be
run on the local white-box backend and on a black-box API arm with *the same
conversation* (docs/bench/01).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from bench.types import (
    Capability,
    Conversation,
    Item,
    NeedsJudge,
    Outcome,
    ProbePoint,
    Response,
    Trial,
)

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

    def build(self, item: Item, condition: str) -> Trial: ...
    def probe_points(self, trial: Optional[Trial]) -> List[ProbePoint]: ...
    def extract(self, resp: Response) -> Union[Outcome, NeedsJudge]: ...


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
    """Shared plumbing. Subclasses supply name/question/candidates."""

    name: str = "base"
    requires: frozenset = frozenset({Capability.GENERATE, Capability.IMAGES})
    prefers: frozenset = frozenset({Capability.LOGPROB, Capability.ACTIVATIONS})
    conditions: List[str] = list(CONDITIONS)
    uses_images: bool = True
    family: str = "base"
    question: str = ""
    candidates: List[str] = []

    def build(self, item: Item, condition: str) -> Trial:
        conversation = build_conversation(item, condition, self.question)
        return Trial(
            surface=self.name,
            item_id=item.item_id,
            condition=condition,
            conversation=conversation,
            candidates=list(self.candidates),
            probe_points=self.probe_points(None),
            max_new_tokens=0,
            meta={
                "family": self.family,
                "question": self.question,
                "condition_desc": CONDITION_SPEC[condition]["desc"],
                "n_images": len(conversation.images),
            },
        )

    def probe_points(self, trial: Optional[Trial] = None) -> List[ProbePoint]:
        return list(DEFAULT_PROBE_POINTS)

    def extract(self, resp: Response) -> Union[Outcome, NeedsJudge]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "requires": sorted(str(c) for c in self.requires),
            "prefers": sorted(str(c) for c in self.prefers),
            "conditions": list(self.conditions),
            "candidates": list(self.candidates),
            "probe_points": [p.name for p in self.probe_points(None)],
        }
