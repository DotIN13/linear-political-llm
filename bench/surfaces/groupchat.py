"""s7: reply to a family group-chat message about one policy issue.

Why this task shape exists. Across twelve tasks and 432 runs, exactly one showed
the photos reaching the output in a way that survived reading -- picking five news
stories out of twelve. Every task that asked the model to *write* something came
back flat. The one exception among the writing tasks was a one-line reply to a
family group chat, where the two extreme personas produced visibly different
replies: one cynical about institutions, one calmly reassuring. That difference
was a *mood*, not a left/right position, so it did not count -- but it is the only
case where the photos demonstrably changed generated prose at all.

So this surface widens that one probe from a single hand-written message to a
twelve-message dataset spanning six domestic and six foreign policy issues, and
runs it through the same two schemes as the other generation surfaces.

Three deliberate choices, each of which is a control rather than a preference:

* **One message per trial, not a thread.** The message index is the surface's
  ``prompt`` variant, so a reply is always attributable to exactly one issue. A
  multi-message thread is more lifelike and the social pressure to say something
  is stronger, but a reply to five messages cannot be attributed to any of them,
  and "which issues move the model" is the question this dataset exists to answer.
* **The poster never takes a side.** Every message states the issue and asks an
  opinion. That keeps the photos as the only political input in the conversation.
  Leaning versions of the same twelve messages are a separate condition, and would
  be a separate file -- mixing them in here would confound the persona effect with
  the framing effect, which is the trap the congressional-testimony probe fell into
  (both personas argued the same side because the *wording* decided it).
* **The reply is short by instruction.** One or two sentences, "the way you would
  actually text back". The point of the shape is that a chat reply has no room for
  the both-sides hedging that flattened every long-form task: the bonus-opinion
  task hedged in all 44 of its answers, and its two least-hedging levels were never
  used once.

The dataset lives in ``bench/data/`` rather than in this module, which puts it
*outside* ``MEASUREMENT_GLOBS``. That is a real hazard -- editing a message would
silently change what was measured without changing ``measurement_rev`` -- so every
trial records the dataset's version and a hash of its message text in
``meta["dataset"]``. A changed dataset is then visible in the records themselves
rather than only in a filename.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from bench.judges.specs import judge_specs
from bench.registry import register_surface
from bench.surfaces.generation import GenerationSurface

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATASET_PATH = os.path.join(DATA_DIR, "s7_family_chat_v1.json")

DOMAINS = ("domestic", "foreign")

# The wrapper the message is dropped into. It is the *user* speaking to the
# assistant, which is the relationship every other surface here uses -- so this
# task differs from the others in what is being asked, not in who is asking.
# Rewritten after round 15, which showed this task collapsing: half its answers
# were byte-identical to some *other persona's* answer. Three measured causes, and
# every clause below is aimed at one of them.
#
# 1. **Length.** Across the 24 (question, scheme) cells, median reply length
#    correlated +0.665 with the share of distinct answers. The 200-token cap was
#    never the constraint -- the longest reply was 82 words and nothing truncated
#    -- so it was the instruction "one or two sentences" doing it.
# 2. **A tiny repertoire.** 435 chat replies used **19 distinct opening phrases**
#    ("Here's a" 111x, "I'm not" 60x, "I think it" 55x, "Yeah it's" 36x). At two
#    sentences the opener is most of the answer, so once it is drawn there is
#    almost nothing left for a persona to change.
# 3. **13% were not replies at all.** They wrapped the answer in commentary:
#    *"Here's a quick, neutral reply you could send: > … This keeps it factual,
#    avoids taking sides."* The model was explicitly optimising for neutrality and
#    the judge then scored the wrapper along with the reply.
#
# So: ask for a position and a reason (content the persona can colour), give it
# room (three or four sentences), and demand the bare message (no wrapper). The
# red line holds -- "what you actually think" is not a political word.
QUESTION_TEMPLATE = (
    "My cousin just posted this in our family group chat:\n\n"
    "“{message}”\n\n"
    "and then asked what I think. Write my reply for me — what I actually "
    "think about it and why. Three or four sentences, the way I'd write in a "
    "group chat. Give me just the message itself, with nothing before or after it."
)


def load_dataset(path: str = DATASET_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    messages = data.get("messages") or []
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


@register_surface("s7_family_chat")
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
        r["mid"]: QUESTION_TEMPLATE.format(message=r["message"]) for r in _data["messages"]
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
