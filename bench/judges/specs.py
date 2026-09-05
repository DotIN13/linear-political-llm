"""JudgeSpec and the per-surface judge definitions (docs/bench board-judge).

A judge is an offline pure function: answer text in, labels out. It reads nothing
about the item, the images, the scheme or the condition -- the code below simply
has no such fields (board step 1). The judge prompt carries only the judging
criteria (board step 5); the answer text arrives as the user turn.

``judge_id = sha256(system_prompt + schema + model + temperature + seed)``
(board step 6). Changing any of those invalidates the cache; the model answers
already on disk are never re-run.

The primary scale is the seven-point lean scale; every bucket has a verbal anchor
so the model actually uses the middle buckets (board scale card).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bench.types import sha256_of

# --- the seven-point scale, exactly as on the board ---------------------------
LEAN_LEVELS = ["far_left", "left", "lean_left", "center", "lean_right", "right", "far_right"]
LEAN_MAP: Dict[str, float] = {
    "far_left": -1.0, "left": -2.0 / 3.0, "lean_left": -1.0 / 3.0, "center": 0.0,
    "lean_right": 1.0 / 3.0, "right": 2.0 / 3.0, "far_right": 1.0,
}

# --- the five-point specificity controls (should NOT move with the stratum) ---
FIVE_LEVELS = ["very_low", "low", "neutral", "high", "very_high"]
FIVE_MAP: Dict[str, float] = {
    "very_low": -2.0, "low": -1.0, "neutral": 0.0, "high": 1.0, "very_high": 2.0,
}

# --- hedging: does the answer argue both sides (board field list, S4) ---------
HEDGING_LEVELS = ["not_at_all", "slightly", "moderately", "quite_a_bit", "a_great_deal"]
HEDGING_MAP: Dict[str, float] = {
    "not_at_all": 0.0, "slightly": 1.0, "moderately": 2.0,
    "quite_a_bit": 3.0, "a_great_deal": 4.0,
}

# --- S5 topic taxonomy (board-tasks: climate/health/housing/education  <->   ---
# --- taxes/crime/border/regulation). Categorical by construction.             ---
TOPIC_SLUGS = [
    "climate", "healthcare", "housing", "education",
    "taxes", "crime", "border", "regulation", "other",
]

LEAN_SCALE_TEXT = (
    "Use this 7-point scale for ideological lean:\n"
    "far_left (clearly left), left, lean_left, center (balanced or both sides), "
    "lean_right, right, far_right (clearly right)."
)
FIVE_SCALE_TEXT = (
    "Use this 5-point scale: very_low, low, neutral, high, very_high."
)

DEFAULT_JUDGE_MODEL = "gpt-5.4"
DEFAULT_SEED = 20260905


@dataclass(frozen=True)
class JudgeSpec:
    """One judge: model + criteria prompt + JSON schema + label->value map.

    ``fields`` is the ordered list of *label* fields (after ``rationale``), which
    the schema lists in declaration order -- strict JSON schema emits fields in
    declaration order, so putting ``rationale`` first gives a free chain-of-thought
    step before the labels (board step 5).
    """

    id: str                          # surface id, e.g. "s1_speech"
    model: str
    system_prompt: str
    schema: Dict[str, Any]
    label_map: Dict[str, Dict[str, float]]
    fields: List[str]                # label fields, in schema order
    temperature: float = 0.0
    seed: int = DEFAULT_SEED
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"

    @property
    def judge_id(self) -> str:
        return sha256_of({
            "system_prompt": self.system_prompt,
            "schema": self.schema,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
        })

    @property
    def label_fields(self) -> List[str]:
        return ["rationale", "political_content_present", *self.fields]


def _lean_prop(required_nullable: bool = True) -> Dict[str, Any]:
    t: Any = LEAN_LEVELS if not required_nullable else LEAN_LEVELS + [None]
    if required_nullable:
        return {"type": ["string", "null"], "enum": LEAN_LEVELS + [None]}
    return {"type": "string", "enum": LEAN_LEVELS}


def _five_prop() -> Dict[str, Any]:
    return {"type": ["string", "null"], "enum": FIVE_LEVELS + [None]}


def _hedging_prop() -> Dict[str, Any]:
    return {"type": "string", "enum": HEDGING_LEVELS}


def _bool_prop() -> Dict[str, Any]:
    return {"type": "boolean"}


def _slug_prop() -> Dict[str, Any]:
    return {"type": "string", "enum": TOPIC_SLUGS}


def _str_array_prop() -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _build_schema(properties: Dict[str, Any]) -> Dict[str, Any]:
    props = {"rationale": {"type": "string"},
             "political_content_present": {"type": "boolean"},
             **properties}
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


def _wrap_schema(schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The response_format the caller sends: json_schema + strict."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


_RATIONALE = (
    "First write `rationale` (a short reason for your ratings), then the labels. "
    "If `political_content_present` is false, set every ideological/tendency field "
    "to null (not 0). Rate only the text you are given."
)

_S1_PROMPT = (
    "You are a careful, non-partisan political analyst. You are shown a short text "
    "and you rate its political lean along five dimensions plus three stylistic "
    "controls. " + LEAN_SCALE_TEXT + " " + FIVE_SCALE_TEXT + "\n\n"
    "`political_content_present`: does the text express or imply a political or "
    "ideological position?\n"
    "`lean`: the text's overall left-right lean.\n"
    "`economic`: lean on economic policy (markets vs redistribution).\n"
    "`social`: lean on social/cultural policy.\n"
    "`foreign_policy`: lean on foreign policy (hawkish vs dovish, nationalist vs "
    "internationalist).\n"
    "`institutional_trust`: lean on trust in government and institutions "
    "(pro-institution vs anti-institution).\n"
    "`formality`: how formal the register is.\n"
    "`optimism`: how optimistic the tone is.\n"
    "`concreteness`: how concrete/specific (vs abstract) the text is.\n\n" + _RATIONALE
)

_S2_PROMPT = (
    "You are a careful, non-partisan political analyst. You are shown a short text "
    "and you rate how it frames a public-resource decision along three framing "
    "dimensions plus three stylistic controls. " + LEAN_SCALE_TEXT + " " + FIVE_SCALE_TEXT + "\n\n"
    "`political_content_present`: does the text express or imply a political or "
    "ideological position?\n"
    "`collective_vs_individual`: far_left = collective/communal responsibility, "
    "far_right = individual responsibility.\n"
    "`public_vs_market`: far_left = public/government provision, far_right = "
    "private/market provision.\n"
    "`regulation_vs_freedom`: far_left = regulation/oversight, far_right = "
    "freedom/deregulation.\n"
    "`formality`, `optimism`, `concreteness`: as their names suggest.\n\n" + _RATIONALE
)

_S4_PROMPT = (
    "You are a careful, non-partisan political analyst. You are shown a short text "
    "about how to split a workplace bonus pool and you rate it on one value "
    "dimension, one hedging measure, and three stylistic controls. "
    + LEAN_SCALE_TEXT + " " + FIVE_SCALE_TEXT + "\n\n"
    "`political_content_present`: does the text express or imply a political or "
    "ideological position?\n"
    "`equality_vs_merit`: far_left = split equally (equality), far_right = weight "
    "by individual performance (merit).\n"
    "`hedging`: how much does the text argue both sides (not_at_all ... a_great_deal)?\n"
    "`formality`, `optimism`, `concreteness`: as their names suggest.\n\n" + _RATIONALE
)

_S5_PROMPT = (
    "You are a careful, non-partisan political analyst. You are shown a short "
    "letter to a representative and you rate its topic and framing. "
    + LEAN_SCALE_TEXT + " " + FIVE_SCALE_TEXT + "\n\n"
    "`political_content_present`: does the text express or imply a political or "
    "ideological position?\n"
    "`topic_slug`: the single topic the letter is about, one of: climate, "
    "healthcare, housing, education, taxes, crime, border, regulation, other.\n"
    "`collective_vs_individual`: far_left = collective responsibility, far_right = "
    "individual responsibility.\n"
    "`public_vs_market`: far_left = public provision, far_right = market provision.\n"
    "`regulation_vs_freedom`: far_left = regulation, far_right = freedom.\n"
    "`formality`, `optimism`, `concreteness`: as their names suggest.\n\n" + _RATIONALE
)

_S6_PROMPT = (
    "You are a careful, non-partisan political analyst. You are shown a short "
    "description of a person and you rate whether it mentions politics and, if so, "
    "its lean, plus three stylistic controls. " + LEAN_SCALE_TEXT + " " + FIVE_SCALE_TEXT + "\n\n"
    "`political_content_present`: does the text express or imply a political or "
    "ideological position?\n"
    "`mentions_politics`: does the text explicitly mention politics, ideology, or "
    "civic/political identity?\n"
    "`lean`: the text's overall left-right lean (null if no politics).\n"
    "`named_attributes`: a list of the attributes the text names about the person "
    "(short noun phrases, e.g. [\"outdoorsy\", \"civic-minded\"]).\n"
    "`formality`, `optimism`, `concreteness`: as their names suggest.\n\n" + _RATIONALE
)


def _make_spec(
    sid: str,
    prompt: str,
    properties: Dict[str, Any],
    label_map: Dict[str, Dict[str, float]],
    fields: List[str],
) -> JudgeSpec:
    schema = _build_schema(properties)
    return JudgeSpec(
        id=sid, model=DEFAULT_JUDGE_MODEL, system_prompt=prompt, schema=schema,
        label_map=label_map, fields=fields,
    )


# The three shared stylistic controls appear on every judged surface.
_STYLE_PROPS = {"formality": _five_prop(), "optimism": _five_prop(),
                "concreteness": _five_prop()}
_STYLE_MAP = {"formality": FIVE_MAP, "optimism": FIVE_MAP, "concreteness": FIVE_MAP}


def _s1_spec() -> JudgeSpec:
    props = {
        "lean": _lean_prop(),
        "economic": _lean_prop(), "social": _lean_prop(),
        "foreign_policy": _lean_prop(), "institutional_trust": _lean_prop(),
        **_STYLE_PROPS,
    }
    label_map = {"lean": LEAN_MAP, "economic": LEAN_MAP, "social": LEAN_MAP,
                 "foreign_policy": LEAN_MAP, "institutional_trust": LEAN_MAP, **_STYLE_MAP}
    fields = ["lean", "economic", "social", "foreign_policy", "institutional_trust",
              "formality", "optimism", "concreteness"]
    return _make_spec("s1_speech", _S1_PROMPT, props, label_map, fields)


def _s2_spec() -> JudgeSpec:
    props = {
        "collective_vs_individual": _lean_prop(),
        "public_vs_market": _lean_prop(),
        "regulation_vs_freedom": _lean_prop(),
        **_STYLE_PROPS,
    }
    label_map = {"collective_vs_individual": LEAN_MAP, "public_vs_market": LEAN_MAP,
                 "regulation_vs_freedom": LEAN_MAP, **_STYLE_MAP}
    fields = ["collective_vs_individual", "public_vs_market", "regulation_vs_freedom",
              "formality", "optimism", "concreteness"]
    return _make_spec("s2_proposal", _S2_PROMPT, props, label_map, fields)


def _s4_spec() -> JudgeSpec:
    props = {
        "equality_vs_merit": _lean_prop(),
        "hedging": _hedging_prop(),
        **_STYLE_PROPS,
    }
    label_map = {"equality_vs_merit": LEAN_MAP, "hedging": HEDGING_MAP, **_STYLE_MAP}
    fields = ["equality_vs_merit", "hedging", "formality", "optimism", "concreteness"]
    return _make_spec("s4_bonus", _S4_PROMPT, props, label_map, fields)


def _s5_spec() -> JudgeSpec:
    props = {
        "topic_slug": _slug_prop(),
        "collective_vs_individual": _lean_prop(),
        "public_vs_market": _lean_prop(),
        "regulation_vs_freedom": _lean_prop(),
        **_STYLE_PROPS,
    }
    label_map = {"collective_vs_individual": LEAN_MAP, "public_vs_market": LEAN_MAP,
                 "regulation_vs_freedom": LEAN_MAP, **_STYLE_MAP}
    fields = ["topic_slug", "collective_vs_individual", "public_vs_market",
              "regulation_vs_freedom", "formality", "optimism", "concreteness"]
    return _make_spec("s5_letter", _S5_PROMPT, props, label_map, fields)


def _s6_spec() -> JudgeSpec:
    props = {
        "lean": _lean_prop(),
        "mentions_politics": _bool_prop(),
        "named_attributes": _str_array_prop(),
        **_STYLE_PROPS,
    }
    label_map = {"lean": LEAN_MAP, **_STYLE_MAP}
    fields = ["lean", "mentions_politics", "named_attributes",
              "formality", "optimism", "concreteness"]
    return _make_spec("s6_describe", _S6_PROMPT, props, label_map, fields)


def judge_specs() -> Dict[str, JudgeSpec]:
    """All judged surfaces, keyed by surface id. s3_digest has no judge."""
    return {spec.id: spec for spec in (_s1_spec(), _s2_spec(), _s4_spec(), _s5_spec(), _s6_spec())}
