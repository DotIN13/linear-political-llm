"""The two conditions, their historical letters, and the one that was removed.

Shared: ``photos``/``no_photos`` is the treatment, identical for every question.
Note that ``bench/surfaces/base.py`` has its *own* five-condition set and its own
``normalise_condition`` for the multiple-choice family -- a genuinely different
design, not a duplicate of this one."""

from __future__ import annotations


# Two conditions, named for what they are.
#
#   photos      the persona is shown: its photos delivered per conversation scheme
#   no_photos   **the task on its own** -- no share line, no memory directories,
#               no filenames, no tool calls, no scripted small talk. Just the
#               question. This is the reference point for "what does the model say
#               if you simply ask it", and it runs **once per question**: with no
#               persona in it, it is identical for every persona, and with no
#               scaffolding it is identical for both conversation schemes.
#
# There used to be a third, `E`, which kept the entire transcript and withheld
# only the image bytes -- the model was told these were photos of where someone
# lives, "opened" three files and got back a filename and nothing else. It is
# removed. It answered a narrower question (the pixels, holding the framing
# constant) at the cost of being a strange stimulus that is nobody's default
# behaviour, and building an experiment on two baselines that get confused for
# each other is worse than having the narrower one at all.
#
# The old single-letter names are gone too. `C` still resolves, because a dozen
# historical pilots pass it and it means exactly `photos`; `E` raises, because
# silently turning it into `no_photos` would swap one stimulus for a different
# one without anybody noticing.
CONDITIONS = ["photos", "no_photos"]
# The letters the generation surfaces used to use. `C` was the three-photos
# in-conversation template and `Q` was the bare question, added and renamed the
# same day.
# The letters the *generation* surfaces used. The multiple-choice surfaces keep
# their own A-E set (`bench/surfaces/base.py`), which is a genuinely different
# five-way design -- photo count crossed with whether there is a conversation --
# and is not renamed here.
CONDITION_ALIASES = {"C": "photos", "Q": "no_photos"}
REMOVED_CONDITIONS = {
    "E": ("condition 'E' (full transcript, image bytes withheld) was removed. It is "
          "not the same stimulus as 'no_photos', which is the bare question, so it "
          "cannot be aliased. Use 'no_photos' if you want the baseline, or restore "
          "E deliberately if you specifically want the pixels-only contrast."),
}


def normalise_condition(condition: str) -> str:
    """Accept the historical letters, refuse the one that changed meaning."""
    if condition in REMOVED_CONDITIONS:
        raise ValueError(REMOVED_CONDITIONS[condition])
    return CONDITION_ALIASES.get(condition, condition)


CONDITION_DESC = {
    "photos": "the persona's photos, delivered per conversation scheme",
    "no_photos": "the question on its own, no persona framing -- once per question",
}
