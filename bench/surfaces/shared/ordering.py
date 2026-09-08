"""Seeded orders over a pool of items.

Only s3 has a pool today, but ``GenerationSurface._item_order`` -- shared code --
is what calls ``sampled_order``, so this is machinery the base class needs rather
than machinery one question owns."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Sequence


def _order_seed(item_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{item_id}|{seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def shuffled_order(headlines: Sequence[Any], item_id: str, seed: int) -> List[int]:
    """**No production caller.** `sampled_order` replaced it; only two test
    modules still reach for it, and they are what keeps it here -- they pin the
    twelve-of-twelve ordering the design used before the pool grew to 24.
    Delete it and those go with it; that is a decision about whether the old
    scheme is worth documenting, not a tidy-up.

    A per-(item, seed) permutation of the headline indices.

    Seeding by ``(item_id, seed)`` rather than ``item_id`` alone decouples the
    order effect from the item effect: two seeds for one item get two orders.
    """
    rng = random.Random(_order_seed(item_id, seed))
    order = list(range(len(headlines)))
    rng.shuffle(order)
    return order


def sampled_order(headlines: Sequence[Any], item_id: str, seed: int,
                  per_topic: int = 1) -> List[int]:
    """Show one version of every topic: a stratified sample, then shuffled.

    The pool carries each topic twice -- one left-of-centre outlet and one
    right-of-centre outlet covering the same story -- and this draws **one of
    the two per topic**, so a trial shows every topic exactly once and the only
    thing the draw varies is which side's coverage of it appears.

    Why stratify rather than take any 12 of 24. An unconstrained draw would show
    some topics twice and others not at all, so the topic mix would vary trial to
    trial and become a second source of variance on top of the slant. Holding
    all topics present every trial makes the shown *set of topics* a constant and
    the shown *slant* the only thing that moves.

    **This moves the topic/slant decoupling from within a trial to across
    trials.** In the twelve-headline design both sides of a story were on screen
    together, so choosing one over the other held topic exactly fixed -- strong,
    but it also showed the model two versions of the same story, which no real
    feed does. Here the decoupling comes from randomising which side is shown,
    which is a weaker guarantee per trial and an equally valid one in aggregate.
    It also means **the per-trial baseline is not a constant**: see
    ``slant_rel_mean`` in ``extract_picks``.

    **The draw is balanced, not independent.** Half the topics show their left
    side and half their right, assigned at random -- because drawing each topic
    independently lets the deal lean, and one seed on the six-topic pool dealt
    six right-side stories out of six. A lopsided deal inflates or masks a slant
    preference that was never there.

    Returns positions into ``headlines``, in presentation order.
    """
    rng = random.Random(_order_seed(item_id, seed))
    by_topic: Dict[Any, List[int]] = {}
    for i, h in enumerate(headlines):
        by_topic.setdefault(h["topic"], []).append(i)
    topics = sorted(by_topic)                      # sorted: draw order is not file order

    if per_topic == 1 and all(len(by_topic[t]) == 2 for t in topics):
        # **Balanced draw**: half the topics show their left-side coverage and
        # half their right-side, assigned at random. Drawing each topic's side
        # independently would let the deal itself lean -- on the six-topic pool
        # one seed dealt six right-side stories out of six -- and a lopsided deal
        # inflates or masks a slant preference that is not there. Balancing costs
        # nothing and removes that variance at the source.
        # `len(topics) // 2` sent the leftover topic on an odd pool to the else
        # branch below, which always takes "right" -- so a 5-topic pool dealt 3
        # right in 400 of 400 seeds. A constant, not a lean, and exactly the
        # failure the balanced draw exists to prevent. The pool is 12 topics
        # today so the floor was exact and this never bit; it would have bitten
        # silently the day anyone added or dropped a topic.
        #
        # The odd topic now goes to a side chosen by the same rng, so the deal is
        # balanced to within the one topic that cannot be split, in a direction
        # that varies by seed instead of always being "right".
        half = len(topics) // 2
        if len(topics) % 2 and rng.random() < 0.5:
            half += 1
        left_topics = set(rng.sample(topics, half))
        chosen = [next(i for i in by_topic[t]
                       if headlines[i]["side"] == ("left" if t in left_topics else "right"))
                  for t in topics]
    else:
        chosen = []
        for topic in topics:
            pool = by_topic[topic]
            chosen.extend(rng.sample(pool, min(per_topic, len(pool))))
    rng.shuffle(chosen)
    return chosen
