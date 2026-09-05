"""bench CLI. One subcommand, one job (docs/bench/02).

    surfaces  adaptors  check  sample  run  score      -- implemented
    judge     report                                   -- stubs this round
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from bench import registry
from bench.adaptors.base import check_candidates, check_capabilities
from bench.store import (
    RunStore, git_rev, measurement_inputs, measurement_rev, read_items, trial_key, write_items,
)
from bench.surfaces.base import validate_variant_space
from bench.types import (
    BASELINE_ITEM_ID, Item, NeedsJudge, Outcome, baseline_item, canonical_variant,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
def cmd_surfaces(args: argparse.Namespace) -> int:
    registry.load_all()
    rows = []
    for name in registry.surface_names():
        rows.append(registry.get_surface(name)().describe())
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    print(f"{len(rows)} surfaces\n")
    for row in rows:
        print(f"  {row['name']:<16} family={row['family']}")
        print(f"      requires     : {{{', '.join(row['requires'])}}}")
        print(f"      prefers      : {{{', '.join(row['prefers'])}}}")
        print(f"      conditions   : {','.join(row['conditions'])} "
              f"(item-invariant: {','.join(row['item_invariant_conditions']) or '-'})")
        if row.get("family") == "generation":
            print(f"      schemes      : {','.join(row['schemes'])}  "
                  f"judge={row.get('judge') or '-'}  max_new_tokens={row.get('max_new_tokens')}")
        else:
            print(f"      options      : {row['options']}  measured as {row['candidates']}")
            print(f"      phrasings    : {row['n_phrasings']}  variants/item: {len(row['variants'])}")
        print(f"      probe_points : {row['probe_points']}")
        for line in row["example_question"].splitlines():
            print(f"        | {line}")
    return 0


def cmd_adaptors(args: argparse.Namespace) -> int:
    registry.load_all()
    rows = []
    for name in registry.adaptor_names():
        cls = registry.get_adaptor(name)
        rows.append({"name": cls.name,
                     "capabilities": sorted(str(c) for c in cls.capabilities),
                     "class": f"{cls.__module__}.{cls.__qualname__}"})
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    all_caps = ["generate", "logprob", "activations", "steer", "images", "session"]
    width = max(len(r["name"]) for r in rows)
    print(f"{'adaptor'.ljust(width)}  " + "  ".join(c[:11].ljust(11) for c in all_caps))
    for row in rows:
        marks = ["yes".ljust(11) if c in row["capabilities"] else "-".ljust(11) for c in all_caps]
        print(f"{row['name'].ljust(width)}  " + "  ".join(marks))
    print()
    for row in rows:
        print(f"  {row['name']}: {{{', '.join(row['capabilities'])}}}  <- {row['class']}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Capability gate + candidate gate. Both run before anything is queued."""
    registry.load_all()
    names = registry.surface_names() if args.surface == "all" else _split_list(args.surface)
    adaptor_cls = registry.get_adaptor(args.adaptor)

    # The capability gate needs only the class; the candidate gate needs an
    # instance, because it does one real forward pass.
    adaptor = None
    if args.candidates:
        kwargs: Dict[str, Any] = {"seed": args.seed}
        if args.model:
            kwargs["model"] = args.model
        if args.probe and args.adaptor == "local_hf":
            kwargs["probe"] = args.probe
        adaptor = adaptor_cls(**kwargs)

    items = None
    if args.items:
        items = [Item.from_dict(r) for r in read_items(_abs(args.items))]

    payload: List[Dict[str, Any]] = []
    ok = True
    for name in names:
        surface = registry.get_surface(name)()
        cap = check_capabilities(surface, adaptor or adaptor_cls)
        entry: Dict[str, Any] = {"capabilities": cap.to_dict()}
        if not args.json:
            print(cap.render())
        ok = ok and cap.ok

        # variant space: declared by the surface, checked for canonical form
        problems = validate_variant_space(surface)
        entry["variants"] = {"declared": surface.variants(),
                             "canonical": [canonical_variant(v) for v in surface.variants()],
                             "problems": problems}
        if not args.json:
            mark = "OK" if not problems else "x "
            print(f"  [{mark}] variants ({len(surface.variants())}): "
                  f"{', '.join(canonical_variant(v) for v in surface.variants())}")
            for problem in problems:
                print(f"      {problem}")
        ok = ok and not problems

        if args.candidates and cap.ok:
            item = items[0] if items else baseline_item()
            condition = args.condition if items else "E"
            cand = check_candidates(surface, adaptor, item=item,
                                    condition=condition,
                                    variant={"phrasing": args.phrasing, "order": args.order})
            entry["candidates"] = cand.to_dict()
            if not args.json:
                print(cand.render())
            ok = ok and cand.ok
        if not args.json:
            print()
        payload.append(entry)

    if args.json:
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2, sort_keys=True))
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
def cmd_sample(args: argparse.Namespace) -> int:
    from bench import sample as S

    stats_path = _abs(args.stats)
    cache_path = _abs(args.lvis_cache)
    if not os.path.exists(cache_path):
        S.build_lvis_meta_cache(_abs(args.lvis_json), cache_path)

    print(f"[sample] stats: {stats_path}")
    rows = S.load_stats(stats_path, limit=args.limit)
    print(f"[sample] loaded {len(rows)} scored images")
    meta = S.load_lvis_meta(cache_path)
    print(f"[sample] loaded LVIS meta for {len(meta)} images")

    filters = S.parse_filters(args.filters)
    kept, dropped = S.apply_filters(rows, meta, filters)
    print(f"[sample] filters {args.filters!r}"
          f"{'  (integrity only -- content filters are annotations now)' if not args.filters else ''}")
    for reason, count in sorted(dropped.items()):
        print(f"           dropped {count:>7}  {reason}")
    print(f"[sample] pool after filters: {len(kept)} / {len(rows)} scored images")

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    items, profile = S.make_items(
        kept, bins=args.bins, per_bin=args.per_bin,
        images_per_item=args.images_per_item, splits=splits, seed=args.seed,
    )

    # Stimuli are never overwritten in place (docs/bench/03): a new selection rule
    # means a new file name, so old runs keep meaning something.
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_dir = _abs(args.out)
    os.makedirs(out_dir, exist_ok=True)
    for split, records in items.items():
        path = os.path.join(out_dir, f"{split}{suffix}.jsonl")
        n = write_items(path, records)
        print(f"[sample] wrote {n:>6} items -> {path}")
    profile_path = os.path.join(out_dir, f"decile_profile{suffix}.csv")
    S.write_decile_profile(profile, profile_path)
    manifest_path = os.path.join(out_dir, f"sample_manifest{suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"command": sys.argv, "code_rev": git_rev(ROOT_DIR),
                   "filters": args.filters, "stats": stats_path,
                   "n_scored_images": len(rows),
                   "dropped": dict(dropped),
                   "pool_after_filters": len(kept), "profile": profile},
                  handle, indent=2, sort_keys=True, default=str)
    print(f"[sample] wrote {profile_path}")
    print("\nstratum profile (frequencies are reported, never used to select):")
    for row in profile["strata"]:
        cats = ", ".join(f"{n}({c})" for n, c in row["top_categories"][:5])
        print(f"  s{row['stratum']}  n={row['n_images_selected']:>5}  "
              f"mean={row['image_mean_mean']:+.3f}  "
              f"[{row['image_mean_min']:+.3f},{row['image_mean_max']:+.3f}]  "
              f"n_obj_med={row['n_objects_median']:>4}  "
              f"person={row['share_with_person']:.2f}  text={row['share_with_text_cat']:.2f}  "
              f"top: {cats}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    registry.load_all()
    started = time.time()
    code_rev = git_rev(ROOT_DIR)

    surface_names = _split_list(args.surface)
    conditions = _split_list(args.conditions)
    surfaces = {name: registry.get_surface(name)() for name in surface_names}

    # The surface owns its variant space; --variant only narrows it.
    constraints = parse_constraints(args.variant)
    variant_space: Dict[str, List[Dict[str, Any]]] = {}
    for name, surface in surfaces.items():
        problems = validate_variant_space(surface)
        if problems:
            print(f"refusing to run: {name} has an invalid variant space: {problems}",
                  file=sys.stderr)
            return 2
        selected = [v for v in surface.variants() if matches(v, constraints)]
        if not selected:
            print(f"refusing to run: --variant {args.variant} selects none of {name}'s variants "
                  f"({[canonical_variant(v) for v in surface.variants()]})", file=sys.stderr)
            return 2
        variant_space[name] = selected

    adaptor_cls = registry.get_adaptor(args.adaptor)
    adaptor_kwargs: Dict[str, Any] = {"seed": args.seed}
    if args.model:
        adaptor_kwargs["model"] = args.model
    if args.probe and args.adaptor == "local_hf":
        adaptor_kwargs["probe"] = args.probe
    adaptor = adaptor_cls(**adaptor_kwargs)

    # The dedup key is pinned to what can change a measurement, not to git HEAD:
    # editing the README must not invalidate 48k forward passes (task D).
    probe_weights = adaptor.describe().get("probe_weights")
    top_k = getattr(adaptor, "top_k", None)
    note = f"top_k={top_k}" if top_k is not None else ""
    rev = measurement_rev(ROOT_DIR, extra_files=[probe_weights] if probe_weights else [], note=note)
    print(f"[run] code_rev={code_rev}  measurement_rev={rev}  (the key uses measurement_rev)")

    # capability gate, before anything expensive happens
    gate_reports = []
    for name, surface in surfaces.items():
        report = check_capabilities(surface, adaptor)
        gate_reports.append(report.to_dict())
        print(report.render())
        if not report.ok:
            print(f"\nrefusing to run: {name} x {args.adaptor} is blocked", file=sys.stderr)
            return 2
    print()

    items = [Item.from_dict(r) for r in read_items(_abs(args.items))]
    if args.limit:
        items = items[: args.limit]

    # Plan the work first, so the count printed is the count actually run.
    plan: List[Any] = []                    # (surface_name, condition, variant, item)
    n_baseline = 0
    for name, surface in surfaces.items():
        for condition in conditions:
            invariant = surface.is_item_invariant(condition)
            for variant in variant_space[name]:
                if invariant:
                    # One record per (surface, condition, variant): the conversation
                    # is byte-identical for every item, so 300 items would be 300
                    # copies of one number (task C).
                    plan.append((name, condition, variant, baseline_item()))
                    n_baseline += 1
                else:
                    plan.extend((name, condition, variant, item) for item in items)
    # is_item_invariant() is a claim, and acting on it deletes trials -- so check
    # it against real items instead of trusting it. A future surface that puts
    # anything item-specific into the question would otherwise silently collapse
    # every item onto one baseline record.
    for name, surface in surfaces.items():
        for condition in conditions:
            if not surface.is_item_invariant(condition) or len(items) < 2:
                continue
            variant = variant_space[name][0]
            probes = [items[0], items[-1], baseline_item()]
            shas = {surface.build(i, condition, variant).conversation.sha for i in probes}
            if len(shas) != 1:
                print(f"refusing to run: {name} declares condition {condition} item-invariant, "
                      f"but different items build different conversations ({len(shas)} shas)",
                      file=sys.stderr)
                return 2

    n_variants = {name: len(v) for name, v in variant_space.items()}
    print(f"[run] {len(items)} items x {len(surfaces)} surfaces x {len(conditions)} conditions "
          f"x variants {n_variants} -> {len(plan)} trials "
          f"({n_baseline} of them item-invariant baselines)")

    run_dir = _abs(args.out)
    store = RunStore(run_dir=run_dir, conversations_dir=_abs(args.conversations))
    print(f"[run] resume: {store.n_done} trials already in {store.trials_path}")

    model_id = getattr(adaptor, "model", args.model or "")
    manifest_extra = {"gate": gate_reports, "items": _abs(args.items),
                      "surfaces": surface_names, "conditions": conditions,
                      "variant_space": {n: v for n, v in variant_space.items()},
                      "measurement_inputs": measurement_inputs(
                          ROOT_DIR, extra_files=[probe_weights] if probe_weights else []),
                      "n_planned": len(plan)}
    store.write_manifest(sys.argv, code_rev, adaptor.describe(), started,
                         extra=manifest_extra, measurement_rev=rev)

    adaptor.setup()

    # candidate gate: single-token letters + the model actually answering with one
    candidate_reports = []
    if args.candidate_gate:
        probe_item = next((i for i in items if i.image_paths), None) or baseline_item()
        for name, surface in surfaces.items():
            condition = next((c for c in conditions if not surface.is_item_invariant(c)),
                             conditions[0])
            item = probe_item if not surface.is_item_invariant(condition) else baseline_item()
            cand = check_candidates(surface, adaptor, item=item, condition=condition,
                                    variant=variant_space[name][0])
            candidate_reports.append(cand.to_dict())
            print(cand.render())
            if not cand.ok:
                print(f"\nrefusing to run: {name} fails the candidate gate "
                      f"(rerun with --no-candidate-gate to override)", file=sys.stderr)
                store.write_manifest(sys.argv, code_rev, adaptor.describe(), started,
                                     finished_at=time.time(),
                                     extra={**manifest_extra, "candidate_gate": candidate_reports},
                                     measurement_rev=rev)
                adaptor.teardown()
                return 2
        manifest_extra["candidate_gate"] = candidate_reports
        print()

    n_new = n_skip = n_err = 0
    try:
        for name, condition, variant, item in plan:
            surface = surfaces[name]
            trial = surface.build(item, condition, variant)
            # Build before keying: a surface may enrich the variant inside build()
            # (s3_digest adds its per-item headline order), and that order must be
            # part of the identity or two orders would collide on one key.
            key = trial_key(name, item.item_id, condition, trial.variant, adaptor.name,
                            model_id, args.seed, rev)
            if store.has(key):
                n_skip += 1
                continue
            conversation_sha = store.put_conversation(trial.conversation)
            response = adaptor.run(trial)
            if response.error:
                n_err += 1
                print(f"  ! {name}/{item.item_id}/{condition}/{trial.variant_key}: "
                      f"{response.error}", file=sys.stderr)

            extracted = surface.extract(response, trial) if not response.error else None
            outcome = extracted.to_dict() if isinstance(extracted, Outcome) else None
            needs_judge = (
                {"reason": extracted.reason, "hint": extracted.judge_hint}
                if isinstance(extracted, NeedsJudge) else None
            )

            record = {
                "trial_key": key,
                "run_id": os.path.basename(run_dir.rstrip("/")),
                "code_rev": code_rev,              # provenance
                "measurement_rev": rev,            # identity
                "surface": name,
                "surface_family": surface.family,
                "condition": condition,
                "variant": trial.variant,
                "item_id": item.item_id,
                "is_baseline": item.item_id == BASELINE_ITEM_ID,
                "stratum": item.stratum,
                "primary_iv": item.primary_iv,
                "split": item.split,
                "images": trial.conversation.images,
                "image_scores": item.image_scores,
                "image_mean": item.image_mean,     # covariate, not the IV
                "options": trial.meta.get("options"),
                "letter_to_option": trial.meta.get("letter_to_option"),
                "adaptor": adaptor.name,
                "model": model_id,
                "seed": args.seed,
                "conversation_sha": conversation_sha,
                "response": response.to_dict(),
                "probe": response.probe,
                "outcome": outcome,
                "needs_judge": needs_judge,
                "judge": None,
                "timing": {"ms": response.timing_ms},
                "cost_usd": response.cost_usd,
                "error": response.error,
                "covariates": item.covariates,
            }
            store.append(record)
            n_new += 1
            if n_new % 10 == 0:
                rate = n_new / max(time.time() - started, 1e-6)
                print(f"  [{n_new}/{len(plan)}] {rate:.2f} trials/s", flush=True)
    finally:
        store.write_manifest(sys.argv, code_rev, adaptor.describe(), started,
                             finished_at=time.time(),
                             extra={**manifest_extra,
                                    "n_new": n_new, "n_skipped": n_skip, "n_errors": n_err},
                             measurement_rev=rev)
        store.close()
        adaptor.teardown()

    print(f"\n[run] new={n_new} skipped={n_skip} errors={n_err}")
    print(f"[run] {store.trials_path}")
    return 0 if n_err == 0 else 1


# --------------------------------------------------------------------------- #
def cmd_score(args: argparse.Namespace) -> int:
    """Deterministic aggregation. No statistics beyond means -- that is `report`.

    The unit of analysis is the **item**, never the row. Variants (phrasings, the
    two A/B orders) are repeated measures of the same item: with 3 phrasings x 2
    orders, counting rows would inflate n sixfold and every p value with it. So:

      1. fold the two orders of one (surface, item, condition, phrasing) into one
         value and one diagnostic, ``position_bias``;
      2. average over the remaining variants -> one number per item;
      3. only then take mean / sd / n, with n counting items. ``n_rows`` is
         printed beside ``n_items`` so any future inflation is visible.

    ``--by variant`` splits the variants back apart. That is a diagnostic view,
    not the main analysis.
    """
    store = RunStore(run_dir=_abs(args.run), conversations_dir=_abs(args.conversations))
    rows = list(store.read())
    if not rows:
        print(f"no trials in {store.trials_path}", file=sys.stderr)
        return 1

    constraints = parse_constraints(args.filter)
    if constraints:
        kept = [r for r in rows if matches(r, constraints)]
        print(f"filter   : {args.filter} -> {len(kept)}/{len(rows)} rows")
        rows = kept
        if not rows:
            print("no rows left after --filter", file=sys.stderr)
            return 1

    print(f"run      : {args.run}")
    print(f"trials   : {len(rows)}")
    manifest = os.path.join(_abs(args.run), "manifest.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as handle:
            payload = json.load(handle)
        print(f"code_rev : {payload.get('code_rev')}  (provenance only)")
        print(f"meas_rev : {payload.get('measurement_rev')}  (dedup key)")
        print(f"adaptor  : {payload.get('adaptor', {}).get('name')} / "
              f"{payload.get('adaptor', {}).get('model')} / "
              f"probe={payload.get('adaptor', {}).get('probe')}")
    print(f"unit     : item ({'split by variant -- diagnostic view' if args.by == 'variant' else 'variants averaged as repeated measures'})")
    print()

    # -- 1. fold the two A/B orders -----------------------------------------
    pairs: Dict[Any, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant = row.get("variant") or {}
        key = (row.get("surface"), row.get("condition"), row.get("item_id"),
               variant.get("phrasing"))
        pairs[key][str(variant.get("order"))] = row

    folded: List[Dict[str, Any]] = []
    for (surface, condition, item_id, phrasing), by_order in pairs.items():
        values = {o: r.get("outcome", {}).get("value")
                  for o, r in by_order.items()
                  if r.get("outcome") and r["outcome"].get("value") is not None}
        any_row = next(iter(by_order.values()))
        # position_bias = ab reading minus ba reading. Both are already oriented
        # onto options[0], so what is left is the model's preference for whichever
        # option sits at letter A -- content-independent, and free.
        bias = (values["ab"] - values["ba"]) if ("ab" in values and "ba" in values) else None
        for order, row in by_order.items():
            folded.append({
                "surface": surface, "condition": condition, "item_id": item_id,
                "phrasing": phrasing, "order": order,
                "variant_key": canonical_variant(row.get("variant") or {}),
                "stratum": _stratum_of(any_row),
                "is_baseline": bool(any_row.get("is_baseline")),
                "outcome": values.get(order),
                "pair_mean": _mean(list(values.values())),
                "position_bias": bias,
                "s_txt": (row.get("probe") or {}).get("s_txt"),
                "s_img": (row.get("probe") or {}).get("s_img"),
            })

    # -- 2. average the variants -> one value per item ----------------------
    by_variant = args.by == "variant"
    units: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for cell in folded:
        key = (cell["surface"], cell["condition"], cell["item_id"],
               cell["variant_key"] if by_variant else "")
        units[key].append(cell)

    items_agg: List[Dict[str, Any]] = []
    for (surface, condition, item_id, variant_key), cells in units.items():
        items_agg.append({
            "surface": surface, "condition": condition, "item_id": item_id,
            "variant_key": variant_key,
            "stratum": cells[0]["stratum"], "is_baseline": cells[0]["is_baseline"],
            "n_rows": len(cells),
            "n_variants": len({c["variant_key"] for c in cells}),
            "outcome": _mean([c["outcome"] for c in cells if c["outcome"] is not None]),
            # one bias per phrasing (the ab/ba pair), then averaged
            "position_bias": _mean(list({c["phrasing"]: c["position_bias"] for c in cells
                                         if c["position_bias"] is not None}.values())),
            "s_txt": _mean([c["s_txt"] for c in cells if c["s_txt"] is not None]),
            "s_img": _mean([c["s_img"] for c in cells if c["s_img"] is not None]),
        })

    # -- 3. broadcast the item-invariant baseline ---------------------------
    baseline: Dict[Any, float] = {}
    for unit in items_agg:
        if unit["is_baseline"] and unit["outcome"] is not None:
            baseline[(unit["surface"], unit["variant_key"])] = unit["outcome"]
    for unit in items_agg:
        base = baseline.get((unit["surface"], unit["variant_key"]))
        unit["baseline"] = base
        unit["outcome_minus_baseline"] = (
            None if (base is None or unit["outcome"] is None or unit["is_baseline"])
            else unit["outcome"] - base
        )

    n_with = sum(1 for u in items_agg if not u["is_baseline"] and u["baseline"] is not None)
    n_without = sum(1 for u in items_agg if not u["is_baseline"] and u["baseline"] is None)
    print(f"baselines: {len(baseline)} item-invariant cell(s) stored once and broadcast; "
          f"{n_with} image items got one, {n_without} did not")
    for (surface, variant_key), value in sorted(baseline.items(), key=lambda kv: str(kv[0])):
        print(f"           {surface} {variant_key or '(variants averaged)'}: "
              f"baseline outcome = {value:+.4f}")
    print()

    # -- the table -----------------------------------------------------------
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for unit in items_agg:
        groups[(unit["surface"], unit["condition"], unit["stratum"],
                unit["variant_key"] if by_variant else "")].append(unit)

    variant_col = f"{'variant':<26}" if by_variant else ""
    header = (f"{'surface':<16}{'cond':<6}{variant_col}{'strat':>6}{'n_items':>8}{'n_rows':>7}"
              f"{'outcome_mean':>14}{'outcome_sd':>12}{'minus_base':>12}{'pos_bias':>10}"
              f"{'s_txt_mean':>12}{'s_img_mean':>12}")
    print(header)
    print("-" * len(header))
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), str(k[3]), _num(k[2]))):
        surface, condition, stratum, variant_key = key
        bucket = groups[key]
        outcomes = [u["outcome"] for u in bucket if u["outcome"] is not None]
        minus = [u["outcome_minus_baseline"] for u in bucket
                 if u["outcome_minus_baseline"] is not None]
        bias = [u["position_bias"] for u in bucket if u["position_bias"] is not None]
        s_txt = [u["s_txt"] for u in bucket if u["s_txt"] is not None]
        s_img = [u["s_img"] for u in bucket if u["s_img"] is not None]
        label = "base" if bucket[0]["is_baseline"] else _fmt_int(stratum)
        cell = f"{variant_key:<26}" if by_variant else ""
        print(f"{str(surface):<16}{str(condition):<6}{cell}{label:>6}"
              f"{len(bucket):>8}{sum(u['n_rows'] for u in bucket):>7}"
              f"{_fmt(_mean(outcomes)):>14}{_fmt(_sd(outcomes)):>12}"
              f"{_fmt(_mean(minus)):>12}{_fmt(_mean(bias)):>10}"
              f"{_fmt(_mean(s_txt)):>12}{_fmt(_mean(s_img)):>12}")

    # -- manipulation check: does the stratum actually move s_img? -----------
    print()
    print("manipulation check  stratum -> s_img  (ordinal; the stratum is the primary IV)")
    checked = False
    for surface, condition in sorted({(u["surface"], u["condition"]) for u in items_agg
                                      if not u["is_baseline"]}, key=str):
        points = [(u["stratum"], u["s_img"]) for u in items_agg
                  if u["surface"] == surface and u["condition"] == condition
                  and not u["is_baseline"] and u["s_img"] is not None
                  and isinstance(u["stratum"], int)]
        if len(points) < 3:
            continue
        checked = True
        per_stratum: Dict[int, List[float]] = defaultdict(list)
        for stratum, value in points:
            per_stratum[stratum].append(value)
        means = [(s, sum(v) / len(v)) for s, v in sorted(per_stratum.items())]
        steps = sum(1 for a, b in zip(means, means[1:]) if b[1] > a[1])
        rho = _spearman([p[0] for p in points], [p[1] for p in points])
        print(f"  {surface:<16} {condition}  n_items={len(points):>4} strata={len(means):>3}  "
              f"spearman(stratum, s_img)={_fmt(rho)}  "
              f"increasing steps {steps}/{max(len(means) - 1, 0)}")
        print("      " + "  ".join(f"s{s}:{v:+.3f}" for s, v in means))
    if not checked:
        print("  (no image-bearing condition with >=3 items in this run)")

    # completeness
    n_logprob = sum(1 for r in rows if r.get("response", {}).get("logprobs"))
    n_stxt = sum(1 for r in rows if r.get("probe") and r["probe"].get("s_txt") is not None)
    n_simg = sum(1 for r in rows if r.get("probe") and r["probe"].get("s_img") is not None)
    n_paired = sum(1 for c in folded if c["position_bias"] is not None) // 2
    print()
    print(f"rows                    : {len(rows)}   analysis units (items): {len(items_agg)}")
    print(f"records with logprobs   : {n_logprob}/{len(rows)}")
    print(f"records with probe.s_txt: {n_stxt}/{len(rows)}")
    print(f"records with probe.s_img: {n_simg}/{len(rows)}  "
          f"(null by construction in the no-image condition E)")
    print(f"complete ab/ba pairs    : {n_paired}  (only these have a position_bias)")
    print(f"records with errors     : {sum(1 for r in rows if r.get('error'))}/{len(rows)}")
    return 0


def _stratum_of(row: Dict[str, Any]) -> Any:
    value = row.get("stratum", row.get("decile"))
    return value


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, ties averaged. No scipy in this repo (docs/bench/02)."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def cmd_judge(args: argparse.Namespace) -> int:
    """The offline judge step: answer text in, labels out (board-judge).

    Reads already-written ``runs/*.jsonl``, judges every answer whose surface has
    a JudgeSpec, shuffles the order first (board step 2), caches on
    ``(response_hash, judge_id)`` (board step 6), and appends results to a
    separate ``judges.jsonl`` -- trials stay append-only.
    """
    import random

    from bench.judges import JudgeCache, JudgeCaller, judge_specs, response_hash

    registry.load_all()
    specs = judge_specs()
    run_dir = _abs(args.run)
    store = RunStore(run_dir=run_dir, conversations_dir=_abs(args.conversations))
    rows = list(store.read())
    if not rows:
        print(f"no trials in {store.trials_path}", file=sys.stderr)
        return 1

    surfaces = _split_list(args.surface) if args.surface else sorted(specs)
    judged = [r for r in rows
              if r.get("surface") in surfaces
              and r["surface"] in specs
              and (r.get("response") or {}).get("text")]
    if not judged:
        print("no generated answers to judge", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    order = list(judged)
    rng.shuffle(order)
    print(f"judging {len(order)} answers over surfaces {surfaces} (shuffled, seed={args.seed})")

    cache_path = _abs(args.cache) if args.cache else os.path.join(_abs("judge_cache"), "judge.sqlite")
    out_path = args.out if args.out else os.path.join(run_dir, "judges.jsonl")

    seen: set = set()
    n_new = n_cache = n_err = 0
    with JudgeCache(cache_path) as cache:
        with open(out_path, "a", encoding="utf-8") as handle:
            for i, row in enumerate(order):
                spec = specs[row["surface"]]
                if args.model:
                    spec = _with_model(spec, args.model)
                text = row["response"]["text"]
                rhash = response_hash(text)
                cached = cache.get(rhash, spec.judge_id)
                if cached is not None:
                    result = {"cached": True, **cached}
                    n_cache += 1
                else:
                    try:
                        result = JudgeCaller(spec).call(text)
                        cache.put(rhash, spec.judge_id, result)
                        result = {"cached": False, **result}
                        n_new += 1
                    except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
                        result = {"error": f"{type(exc).__name__}: {exc}", "cached": False}
                        n_err += 1
                result.update({"trial_key": row["trial_key"], "surface": row["surface"],
                               "response_hash": rhash})
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  [{i + 1}/{len(order)}] new={n_new} cache={n_cache} err={n_err}",
                          flush=True)

    print(f"[judge] new={n_new} cached={n_cache} errors={n_err} -> {out_path}")
    return 0 if n_err == 0 else 1


def _with_model(spec: Any, model: str) -> Any:
    """Same judge, different model: judge_id changes by construction."""
    return type(spec)(
        id=spec.id, model=model, system_prompt=spec.system_prompt, schema=spec.schema,
        label_map=spec.label_map, fields=spec.fields, temperature=spec.temperature,
        seed=spec.seed, base_url=spec.base_url, api_key_env=spec.api_key_env,
    )


def cmd_report(args: argparse.Namespace) -> int:
    print("bench report: not implemented yet "
          "(planned: dose-response by decile, placebo direction, mediation; needs a stats dep)")
    return 0


# --------------------------------------------------------------------------- #
def parse_constraints(specs: Optional[Sequence[str]]) -> List[Any]:
    """['variant.phrasing=0', 'condition=C'] -> [(path, value), ...].

    A dotted path and a string comparison. Deliberately not a query language:
    the point is only to pull one variant, or one condition, back out of a run.
    """
    out: List[Any] = []
    for spec in specs or []:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"filter {part!r} must look like path=value")
            path, value = part.split("=", 1)
            out.append((path.strip(), value.strip()))
    return out


def dotted_get(payload: Any, path: str) -> Any:
    for key in path.split("."):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
        else:
            return None
    return payload


def matches(payload: Dict[str, Any], constraints: Sequence[Any], prefix: str = "") -> bool:
    for path, value in constraints:
        lookup = path[len(prefix):] if prefix and path.startswith(prefix) else path
        actual = dotted_get(payload, lookup)
        if actual is None or str(actual) != value:
            return False
    return True


def _split_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _sd(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.4f}"


def _fmt_int(value: Any) -> str:
    return "-" if value is None else str(value)


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="stimulus -> surface -> adaptor -> store")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("surfaces", help="list surfaces and their requires")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_surfaces)

    p = sub.add_parser("adaptors", help="list adaptors and their capabilities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_adaptors)

    p = sub.add_parser("check", help="capability gate + candidate gate for surface x adaptor")
    p.add_argument("--surface", required=True, help="comma separated, or 'all'")
    p.add_argument("--adaptor", required=True)
    p.add_argument("--candidates", dest="candidates", action="store_true", default=True,
                   help="check the candidate tokens too (default: on)")
    p.add_argument("--no-candidates", dest="candidates", action="store_false",
                   help="capability gate only; does not load the model")
    p.add_argument("--items", default=None,
                   help="probe with the first item of this file instead of a synthetic no-image one")
    p.add_argument("--condition", default="C", help="condition used with --items")
    p.add_argument("--phrasing", type=int, default=0)
    p.add_argument("--order", default="ab", choices=["ab", "ba"])
    p.add_argument("--model", default=None)
    p.add_argument("--probe", default="combined_ideology_headwise_linear")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("sample", help="build frozen stimulus items from the LVIS score CSV")
    from bench.sample import DEFAULT_FILTERS, DEFAULT_LVIS_CACHE, DEFAULT_LVIS_JSON, DEFAULT_STATS_CSV
    p.add_argument("--pool", default="lvis", choices=["lvis"])
    p.add_argument("--stats", default=DEFAULT_STATS_CSV)
    p.add_argument("--lvis-json", default=DEFAULT_LVIS_JSON)
    p.add_argument("--lvis-cache", default=DEFAULT_LVIS_CACHE)
    p.add_argument("--strata", default="image_mean", choices=["image_mean"])
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--per-bin", type=int, default=400, help="images per decile")
    p.add_argument("--images-per-item", type=int, default=3)
    p.add_argument("--filters", default=DEFAULT_FILTERS)
    p.add_argument("--split", default="explore,confirm")
    p.add_argument("--suffix", default="v2",
                   help="written as items/<split>_<suffix>.jsonl; items are never overwritten")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only read the first N CSV rows")
    p.add_argument("--out", default="items/")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("run", help="the only step that touches a GPU or the network")
    p.add_argument("--items", required=True)
    p.add_argument("--surface", required=True, help="comma separated")
    p.add_argument("--conditions", default="C")
    p.add_argument("--variant", action="append", default=None,
                   help="narrow the surface's declared variant space, e.g. --variant phrasing=0 "
                        "(repeatable; without it every declared variant runs)")
    p.add_argument("--candidate-gate", dest="candidate_gate", action="store_true", default=True)
    p.add_argument("--no-candidate-gate", dest="candidate_gate", action="store_false",
                   help="skip the single-token/argmax pre-flight check")
    p.add_argument("--adaptor", default="local_hf")
    p.add_argument("--model", default=None)
    p.add_argument("--probe", default="combined_ideology_headwise_linear")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="first N items only")
    p.add_argument("--out", required=True)
    p.add_argument("--conversations", default="conversations")
    p.add_argument("--resume", action="store_true",
                   help="no-op: resuming is always on, the store dedups on trial_key")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="deterministic aggregation over a run")
    p.add_argument("--run", required=True)
    p.add_argument("--conversations", default="conversations")
    p.add_argument("--filter", action="append", default=None,
                   help="dotted-path row filter, e.g. --filter variant.phrasing=0 (repeatable)")
    p.add_argument("--by", default="item", choices=["item", "variant"],
                   help="item (default): variants are repeated measures and are averaged. "
                        "variant: split them apart -- diagnostic, not the main analysis")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("judge", help="offline judge over a run's generated answers")
    p.add_argument("--run", required=True)
    p.add_argument("--conversations", default="conversations")
    p.add_argument("--surface", default=None,
                   help="comma-separated subset (default: every judged surface present)")
    p.add_argument("--model", default=None, help="override the judge model")
    p.add_argument("--cache", default=None, help="sqlite cache path (default judge_cache/judge.sqlite)")
    p.add_argument("--out", default=None, help="output jsonl (default <run>/judges.jsonl)")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed (board step 2)")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="(stub)")
    p.add_argument("--run")
    p.add_argument("--by", default="decile")
    p.add_argument("--plot", action="store_true")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
