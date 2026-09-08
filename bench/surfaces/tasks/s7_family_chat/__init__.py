"""s7_family_chat: reply to a relative's message in a family group chat.

Was ``bench/surfaces/groupchat.py``. Moved here whole; ``groupchat.py`` is now a
re-export shim, so nothing that imported it had to change.

  prompts/message.j2           the wrapper a message is dropped into
  prompts/messages_v1.jsonl    the twelve messages, six domestic and six foreign
  prompts/messages_v1.meta.json  version, design, and the four rules the set was
                                 built to (framing, poster, naturalness)

**The wrapper is the whole intervention, so it is worth reading before changing.**
Round 15 showed this task collapsing -- half its answers were byte-identical to some
*other persona's* answer -- and the three measured causes are why it reads as it does:

1. **Length.** Across the 24 (question, scheme) cells, median reply length correlated
   +0.665 with the share of distinct answers. The 200-token cap was never the
   constraint (longest reply 82 words, nothing truncated), so it was the instruction
   "one or two sentences" doing it.
2. **A tiny repertoire.** 435 chat replies used 19 distinct opening phrases ("Here's
   a" 111x, "I'm not" 60x, "I think it" 55x, "Yeah it's" 36x). At two sentences the
   opener is most of the answer, so once it is drawn there is almost nothing left for
   a persona to change.
3. **13% were not replies at all.** They wrapped the answer in commentary: *"Here's a
   quick, neutral reply you could send: > ... This keeps it factual, avoids taking
   sides."* The model was optimising for neutrality and the judge then scored the
   wrapper along with the reply.

So: ask for a position and a reason (content a persona can colour), give it room
(three or four sentences), and demand the bare message. The red line holds -- "what
you actually think" is not a political word.

The message is the *user* speaking to the assistant, which is the relationship every
other task uses, so s7 differs from the others in what is asked, not in who is asking.
Leaning versions of the same twelve would be a separate condition and a separate file;
mixing them in would confound the persona effect with the framing effect.

Its dataset used to sit in ``bench/data/``, outside ``MEASUREMENT_GLOBS``, and
``dataset_fingerprint`` below existed because of that. The pool is inside
``bench/surfaces/`` now and is hashed by content, so the fingerprint is belt and
braces rather than the only guard -- and it stays, because it is recorded in every
trial and is what makes an edit visible in the records themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from bench.judges.specs import judge_specs
from bench.surfaces.shared.prompts import pool, read_pool, template, text
from bench.surfaces.shared.surface import GenerationSurface

DOMAINS = ("domestic", "foreign")

#: Where this task's pool lives. ``load_dataset`` reads it and its sibling header.
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "messages_v1.jsonl")

QUESTION_TEMPLATE_PATH = "message.j2"


def render_message(message: str) -> str:
    """The wrapper with one message in it -- the shipped wording, rendered.

    This replaced a module-level ``QUESTION_TEMPLATE`` that callers used with
    ``str.format``. The wrapper is a jinja file now, so its source says
    ``{{ message }}`` where the format string said ``{message}``: a name that still
    looked like a format string but no longer was would be a trap, so the name went
    and a function took its place. ``bench/pilots/probe_s7_images.py`` renders through
    here, which is what keeps the probe on the shipped wording rather than a copy.
    """
    return template(__file__, QUESTION_TEMPLATE_PATH).render(message=message)


def question_template_source() -> str:
    """The wrapper as written, unrendered -- for the tests that pin the wording."""
    return text(__file__, QUESTION_TEMPLATE_PATH)


def load_dataset(path: Optional[str] = None) -> Dict[str, Any]:
    """The twelve messages and their header, validated.

    ``path`` names a ``.jsonl``; its sibling ``.meta.json`` supplies the header when
    there is one. Default is this task's own pool. The shape returned is what it always
    was -- header keys plus ``messages`` -- so callers did not change.
    """
    if path is None:
        messages, header = pool(__file__, "messages_v1")
        path = DATASET_PATH
    else:
        messages, header = read_pool(path)
    data: Dict[str, Any] = {**header, "messages": messages}
    if not messages:
        raise ValueError(f"{path} has no messages")
    seen = set()
    for row in messages:
        for key in ("mid", "domain", "topic", "message"):
            if not row.get(key):
                raise ValueError(f"{path}: message {row.get('mid')!r} is missing {key!r}")
        if row["domain"] not in DOMAINS:
            raise ValueError(f"{path}: message {row['mid']!r} has domain {row['domain']!r}, "
                             f"expected one of {DOMAINS}")
        if row["mid"] in seen:
            raise ValueError(f"{path}: duplicate mid {row['mid']!r}")
        seen.add(row["mid"])
    return data


def dataset_fingerprint(data: Dict[str, Any]) -> str:
    """sha256 over (mid, message) in file order.

    Only the fields that reach the model. Editing a topic label or a comment does
    not change what was measured; editing a message does.
    """
    payload = [[row["mid"], row["message"]] for row in data["messages"]]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class FamilyChatSurface(GenerationSurface):
    """One trial per (persona, message, scheme). No judge wired yet -- on purpose.

    Scoring a two-sentence reply is an open question (the rule-based readers were
    built for essays and the existing judge rubrics score speeches), so this
    surface generates and records, and leaves the reading to a later step. At the
    eyeball size -- twelve messages by two extreme personas -- the replies are
    meant to be read by a person first.
    """

    name = "s7_family_chat"
    family = "generation"
    # Raised 200 -> 320 with the longer ask. The old cap was never binding (longest
    # reply 82 words, zero truncations), and it still should not be -- three or four
    # sentences is ~80 words -- but the headroom means a reply that overruns shows
    # up as a truncation in the records rather than being silently clipped.
    max_new_tokens = 320
    # Reuses s2's rubric verbatim rather than inventing one. It is written
    # generically -- "you are shown a short text and you rate how it frames a
    # public-resource decision" -- with the three left/right axes, the three style
    # controls, `political_content_present` and `refusal`. Nothing in it is about
    # proposals.
    #
    # Reusing it is also what puts s7 and s8 on the *same scale*: the two share
    # their twelve issues, so scoring them with one rubric is what makes "does the
    # task shape matter" answerable. And because judge results cache on
    # (answer text, rubric fingerprint), the shared rubric means an identical
    # answer is paid for once across both tasks.
    judge_spec = judge_specs()["s2_proposal"]
    prefill_text = None

    _data: Dict[str, Any] = load_dataset()
    _by_mid: Dict[str, Dict[str, Any]] = {r["mid"]: r for r in _data["messages"]}
    # The surface's twelve questions. **Not a factorial handle** -- twelve
    # questions is twelve runs of this surface, not twelve levels of a factor.
    # `variants()` on the parent crosses them with the two schemes only because
    # the dedup key is built from the variant dict.
    questions: Dict[str, str] = {
        r["mid"]: render_message(r["message"]) for r in _data["messages"]
    }
    prompt = questions[_data["messages"][0]["mid"]]
    dataset_version: str = str(_data.get("version", "unknown"))
    dataset_hash: str = dataset_fingerprint(_data)

    # -- introspection helpers used by the pilot and by `bench surfaces` -------
    @classmethod
    def messages(cls) -> List[Dict[str, Any]]:
        return list(cls._data["messages"])

    @classmethod
    def mids(cls, domain: Optional[str] = None) -> List[str]:
        if domain is None:
            return [r["mid"] for r in cls._data["messages"]]
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}, expected one of {DOMAINS}")
        return [r["mid"] for r in cls._data["messages"] if r["domain"] == domain]

    def build(self, item, condition, variant=None, seed=None):
        trial = super().build(item, condition, variant, seed)
        mid = str(trial.variant.get("question", ""))
        row = self._by_mid.get(mid, {})
        trial.meta["dataset"] = {
            "version": self.dataset_version,
            "hash": self.dataset_hash,
            "mid": mid,
            "domain": row.get("domain"),
            "topic": row.get("topic"),
            "lean": row.get("lean"),
            "message": row.get("message"),
        }
        return trial

    def describe(self) -> Dict[str, Any]:
        out = super().describe()
        out["dataset"] = {
            "version": self.dataset_version,
            "hash": self.dataset_hash,
            "n_messages": len(self._data["messages"]),
            "domestic": self.mids("domestic"),
            "foreign": self.mids("foreign"),
        }
        return out
