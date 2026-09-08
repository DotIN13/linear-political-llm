"""``GenerationSurface`` -- one open-ended task, six configs.

The class every generation question subclasses, and the one file in ``shared/``
that is not a bag of functions. It still knows about ``headlines`` and
``randomizes_per_item``, which are s3's alone; that is pre-existing and was not
changed here."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from bench.surfaces.shared.conditions import (
    CONDITION_DESC, CONDITIONS, normalise_condition,
)
from bench.surfaces.shared.ordering import sampled_order
from bench.surfaces.shared.refusal import _refusal_match, detect_refusal
from bench.surfaces.shared.text import word_count
from bench.surfaces.shared.transcript import build_scheme_messages
from bench.types import (
    Capability, Conversation, Item, Outcome, ProbePoint, Response, Trial,
)


class GenerationSurface:
    """One open-ended task. Requires generate + images + activations.

    ``activations`` is a **preference**, not a hard requirement. It used to be
    hard, on the argument that without it there is no ``s_pre``/``s_gen``. That
    argument was half right: those two columns do disappear, but the
    *independent* variable does not -- ``image_mean`` is precomputed in
    ``results/token_scoring/`` rather than measured at inference time, so every
    behavioural outcome (deterministic extractors, judge fields) still has its
    dose-response. Blocking cost us the whole vLLM path, which is ~an order of
    magnitude faster, for the sake of a mediator we can measure on a subsample.
    So: activations under ``prefers``, the gate reports DEGRADED, and the
    records carry nulls where the probe would have been -- which is exactly what
    ``prefers`` is for (docs/bench/01).
    """

    name: str = "generation_base"
    family: str = "generation"
    requires = frozenset({Capability.GENERATE, Capability.IMAGES})
    prefers: frozenset = frozenset({Capability.ACTIVATIONS, Capability.LOGPROB})
    conditions: List[str] = list(CONDITIONS)
    schemes: List[str] = ["chat", "agentic"]
    prompt: str = ""
    max_new_tokens: int = 400
    judge_spec = None                       # JudgeSpec or None (s3 has no judge)
    randomizes_per_item: bool = False       # s3 shuffles its headlines per item
    headlines: Optional[List[Dict[str, Any]]] = None
    # A surface's own questions, keyed. **This is not a factorial handle.** Two
    # questions of the same surface are two runs of that surface, not two levels
    # of a factor crossed with everything else -- so nothing should report "n
    # questions" alongside the real handles (photo band, photos present,
    # conversation style). It lives in ``variant`` only because ``trial_key``
    # takes the variant dict, and two runs of one item need distinct keys.
    #
    # Universal: every generation surface has at least one, and a surface that
    # declares none gets ``{"q0": self.prompt}`` from ``question_ids``. That is
    # what lets any surface grow a question set later without a special case.
    questions: Dict[str, str] = {}
    # The opening we write into the assistant turn, applied **whenever it is set**.
    # It used to be an on/off handle; it is not one any more -- a surface either
    # has an opening or it does not.
    prefill_text: Optional[str] = None

    # -- questions (not a handle -- see the class docstring on ``questions``) --
    @classmethod
    def question_ids(cls) -> List[str]:
        """Every question this surface asks, in declaration order. Never empty."""
        return list(cls.questions) if cls.questions else ["q0"]

    def question_text(self, qid: str) -> str:
        if not self.questions:
            return self.prompt
        if qid not in self.questions:
            raise ValueError(f"surface {self.name} has no question {qid!r}; "
                             f"known: {sorted(self.questions)}")
        return self.questions[qid]

    # -- variants ------------------------------------------------------------
    def variants(self) -> List[Dict[str, Any]]:
        """The conversation style, crossed with the surface's own questions.

        Only ``scheme`` is a handle here. ``question`` is in the dict because the
        dedup key is built from it, not because it is a factor.
        """
        return [{"scheme": scheme, "question": qid}
                for scheme in self.schemes for qid in self.question_ids()]

    def validate_variant(self, variant: Dict[str, Any]) -> List[str]:
        problems: List[str] = []
        # ``order``/``attribution`` are s3's, set by build() rather than declared.
        # ``rep`` is the repeat index: the same cell measured again. It has to be in
        # the variant because ``trial_key`` dedups on it -- without it a second
        # reading of an identical cell is silently dropped as already-done, which
        # is why no round before this one could measure its own repeatability.
        unknown = set(variant) - {"scheme", "question", "order", "attribution",
                                  "order_arm", "rep"}
        if unknown:
            problems.append(f"variant has unknown keys {sorted(unknown)}")
        if variant.get("scheme") not in self.schemes:
            problems.append(f"variant scheme={variant.get('scheme')!r} not in {self.schemes}")
        if "question" in variant and variant.get("question") not in self.question_ids():
            problems.append(f"variant question={variant.get('question')!r} not in "
                            f"{self.question_ids()}")
        if "prefill" in variant:
            problems.append("prefill is no longer a handle; a surface either has "
                            "prefill_text or it does not")
        return problems

    # -- invariances that save trials ----------------------------------------
    def is_scheme_invariant(self, condition: str) -> bool:
        """Does the conversation scheme change this condition's prompt at all?

        **It does not for Q.** The bare question has no scaffolding, so `chat`
        and `agentic` produce a byte-identical prompt -- running both would be
        the same trial twice under two names. C and E carry the scheme's
        transcript and so differ.
        """
        condition = normalise_condition(condition)
        if condition not in self.conditions:
            raise ValueError(f"Unknown condition {condition!r}. Known: {self.conditions}")
        return condition == "no_photos"

    def is_item_invariant(self, condition: str) -> bool:
        condition = normalise_condition(condition)
        if condition not in self.conditions:
            raise ValueError(f"Unknown condition {condition!r}. Known: {self.conditions}")
        # No persona in the prompt at all, so it is byte-identical across items
        # *unless* the surface re-deals per item (s3's headline order). So it runs
        # once per question rather than once per persona.
        return condition == "no_photos" and not self.randomizes_per_item

    # -- build ---------------------------------------------------------------
    def _item_order(self, item: Item, seed: Optional[int]) -> Optional[List[int]]:
        """What this trial shows, in presentation order.

        A **sample**, not a permutation: the pool carries two sides per topic and
        a trial shows one side of each, so 24 candidates become 12 shown. See
        ``sampled_order`` for why the draw is balanced, and ``slant_rel_mean``
        for why the baseline has to be computed per trial once it is a sample.
        """
        if not self.randomizes_per_item or not self.headlines:
            return None
        return sampled_order(self.headlines, item.item_id, 0 if seed is None else int(seed))

    def question(self, order: Optional[List[int]] = None, attribution: str = "shown",
                 qid: Optional[str] = None) -> str:
        """``qid=None`` means the surface's first question.

        Not a literal ``"q0"``: a surface that declares its own keys (s7's m01..,
        s8's c01..) has no ``q0``, and defaulting to one raised on every
        ``describe()``.

        ``order`` and ``attribution`` are unused here and are in the signature because
        ``build()`` passes them to every surface. s3 is the one task that needs them,
        and it overrides this to render its headline table from its own template --
        which is why the shared class no longer mentions headlines.
        """
        qid = self.question_ids()[0] if qid is None else qid
        return self.question_text(qid)

    def build(self, item: Item, condition: str, variant: Optional[Dict[str, Any]] = None,
              seed: Optional[int] = None) -> Trial:
        if condition not in self.conditions:
            raise ValueError(f"Unknown condition {condition!r}. Known: {self.conditions}")
        variant = dict(variant or {"scheme": "chat"})
        scheme = str(variant.get("scheme", "chat"))
        if "prefill" in variant:
            raise ValueError(
                f"{self.name}: prefill is no longer a handle. A surface either has "
                f"prefill_text (applied always) or it does not. Drop it from the variant.")
        qid = str(variant.get("question", self.question_ids()[0]))
        variant["question"] = qid
        # rep changes the key and nothing else: byte-identical conversation.
        if "rep" in variant:
            variant["rep"] = int(variant["rep"])
        attribution = str(variant.get("attribution", "shown"))
        variant["attribution"] = attribution
        # A caller-supplied order wins over the seeded shuffle. That is what makes
        # order an explicit, enumerable factor instead of a hidden per-item random
        # draw -- which s3 needs, because the position-1 selection rate measured
        # 1.000 against a 0.417 expectation (docs/bench/13 §3) and only balancing
        # can average it out.
        pinned = variant.get("order")
        order = [int(x) for x in pinned] if pinned is not None else self._item_order(item, seed)
        if order is not None:
            variant["order"] = order

        condition = normalise_condition(condition)
        with_images = condition == "photos"
        image_paths = list(item.image_paths) if with_images else []
        question = self.question(order, attribution, qid)
        if condition == "no_photos":
            # The bare task, with no scaffolding of any kind. Not the scheme's
            # transcript minus its pixels -- the scheme is absent, which is why
            # this is the reference point and E is not.
            messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
            tools = None
        else:
            # The item's own photo count, not len(image_paths): condition E strips
            # the pixels and must keep the same number of files in the transcript.
            n_files = len(item.image_paths) or 3
            messages, tools = build_scheme_messages(scheme, image_paths, question, n_files)
        # Applied whenever the surface has one. No handle, no on/off.
        prefill_text = self.prefill_text

        return Trial(
            surface=self.name,
            item_id=item.item_id,
            condition=condition,
            conversation=Conversation(messages=messages, images=image_paths),
            candidates=[],
            probe_points=self.probe_points(None),
            max_new_tokens=self.max_new_tokens,
            variant=variant,
            meta={
                "family": self.family,
                "scheme": scheme,
                "question_id": qid,
                "prefill": prefill_text,
                "question": question,
                "tools": tools,
                "prefix_n_messages": len(messages) - 1,
                "condition_desc": CONDITION_DESC[condition],
                "n_images": len(image_paths),
                # n_images is 0 for the no-image baseline, so it cannot say how
                # many files the transcript named. n_files can, and two arms with
                # different photo counts are different stimuli even when both
                # have their pixels removed.
                "n_files": 0 if condition == "no_photos" else n_files,
                "item_invariant": self.is_item_invariant(condition),
                "scheme_invariant": self.is_scheme_invariant(condition),
                "judge": self.judge_spec.id if self.judge_spec else None,
            },
        )

    # -- measurement ---------------------------------------------------------
    def probe_points(self, trial: Optional[Trial] = None) -> List[ProbePoint]:
        return [
            ProbePoint(name="s_pre", kind="prefix_end", reduce="last"),
            ProbePoint(name="s_gen", kind="generated_tokens", reduce="mean"),
            ProbePoint(name="s_img", kind="image_tokens", reduce="mean"),
        ]

    def _deterministic(self, text: str, trial: Optional[Trial]) -> Dict[str, Any]:
        return {}

    def extract(self, resp: Response, trial: Optional[Trial] = None) -> Union[Outcome, Any]:
        text = (resp.text or "").strip()
        extra: Dict[str, Any] = {
            "word_count": word_count(text),
            "refusal": detect_refusal(text),
            "refusal_match": _refusal_match(text),
        }
        extra.update(self._deterministic(text, trial))
        return Outcome(kind="generation", value=extra.get("primary"), extra=extra)

    # -- introspection -------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "requires": sorted(str(c) for c in self.requires),
            "prefers": sorted(str(c) for c in self.prefers),
            "conditions": list(self.conditions),
            "schemes": list(self.schemes),
            "options": [],
            "candidates": [],
            "n_phrasings": 0,
            "questions": self.question_ids(),      # the surface's own questions, not a handle
            "prefill": self.prefill_text,
            "variants": self.variants(),
            "item_invariant_conditions": [c for c in self.conditions if self.is_item_invariant(c)],
            "scheme_invariant_conditions": [c for c in self.conditions
                                            if self.is_scheme_invariant(c)],
            "probe_points": [p.name for p in self.probe_points(None)],
            "example_question": self.question(list(range(len(self.headlines)))
                                              if self.headlines else None),
            "judge": self.judge_spec.id if self.judge_spec else None,
            "max_new_tokens": self.max_new_tokens,
        }
