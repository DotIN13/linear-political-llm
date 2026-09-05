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
from bench.adaptors.base import check_capabilities
from bench.store import RunStore, git_rev, read_items, trial_key, write_items
from bench.types import Item, NeedsJudge, Outcome

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
        print(f"      conditions   : {','.join(row['conditions'])}")
        print(f"      candidates   : {row['candidates']}")
        print(f"      probe_points : {row['probe_points']}")
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
    registry.load_all()
    surface = registry.get_surface(args.surface)()
    adaptor_cls = registry.get_adaptor(args.adaptor)
    report = check_capabilities(surface, adaptor_cls)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.render())
    return 0 if report.ok else 2


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
    print(f"[sample] filters {args.filters}")
    for reason, count in sorted(dropped.items()):
        print(f"           dropped {count:>7}  {reason}")
    print(f"[sample] pool after filters: {len(kept)}")

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    items, profile = S.make_items(
        kept, bins=args.bins, per_bin=args.per_bin,
        images_per_item=args.images_per_item, splits=splits, seed=args.seed,
    )

    out_dir = _abs(args.out)
    os.makedirs(out_dir, exist_ok=True)
    for split, records in items.items():
        path = os.path.join(out_dir, f"{split}.jsonl")
        n = write_items(path, records)
        print(f"[sample] wrote {n:>6} items -> {path}")
    profile_path = os.path.join(out_dir, "decile_profile.csv")
    S.write_decile_profile(profile, profile_path)
    with open(os.path.join(out_dir, "sample_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"command": sys.argv, "code_rev": git_rev(ROOT_DIR),
                   "filters": args.filters, "stats": stats_path,
                   "pool_after_filters": len(kept), "profile": profile},
                  handle, indent=2, sort_keys=True, default=str)
    print(f"[sample] wrote {profile_path}")
    print("\ndecile profile:")
    for row in profile["deciles"]:
        cats = ", ".join(f"{n}({c})" for n, c in row["top_categories"][:5])
        print(f"  d{row['decile']}  n={row['n_images_selected']:>5}  "
              f"mean={row['image_mean_mean']:+.3f}  "
              f"[{row['image_mean_min']:+.3f},{row['image_mean_max']:+.3f}]  "
              f"n_obj_med={row['n_objects_median']:>4}  top: {cats}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    registry.load_all()
    started = time.time()
    code_rev = git_rev(ROOT_DIR)

    surface_names = _split_list(args.surface)
    conditions = _split_list(args.conditions)
    surfaces = {name: registry.get_surface(name)() for name in surface_names}

    adaptor_cls = registry.get_adaptor(args.adaptor)
    adaptor_kwargs: Dict[str, Any] = {"seed": args.seed}
    if args.model:
        adaptor_kwargs["model"] = args.model
    if args.probe and args.adaptor == "local_hf":
        adaptor_kwargs["probe"] = args.probe
    adaptor = adaptor_cls(**adaptor_kwargs)

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
    print(f"[run] {len(items)} items x {len(surfaces)} surfaces x {len(conditions)} conditions "
          f"= {len(items) * len(surfaces) * len(conditions)} trials")

    run_dir = _abs(args.out)
    store = RunStore(run_dir=run_dir, conversations_dir=_abs(args.conversations))
    print(f"[run] resume: {store.n_done} trials already in {store.trials_path}")

    model_id = getattr(adaptor, "model", args.model or "")
    store.write_manifest(sys.argv, code_rev, adaptor.describe(), started,
                         extra={"gate": gate_reports, "items": _abs(args.items),
                                "surfaces": surface_names, "conditions": conditions})

    adaptor.setup()
    n_new = n_skip = n_err = 0
    try:
        for item in items:
            for condition in conditions:
                for name, surface in surfaces.items():
                    key = trial_key(name, item.item_id, condition, adaptor.name,
                                    model_id, args.seed, code_rev)
                    if store.has(key):
                        n_skip += 1
                        continue
                    trial = surface.build(item, condition)
                    conversation_sha = store.put_conversation(trial.conversation)
                    response = adaptor.run(trial)
                    if response.error:
                        n_err += 1
                        print(f"  ! {name}/{item.item_id}/{condition}: {response.error}", file=sys.stderr)

                    extracted = surface.extract(response) if not response.error else None
                    outcome = extracted.to_dict() if isinstance(extracted, Outcome) else None
                    needs_judge = (
                        {"reason": extracted.reason, "hint": extracted.judge_hint}
                        if isinstance(extracted, NeedsJudge) else None
                    )

                    record = {
                        "trial_key": key,
                        "run_id": os.path.basename(run_dir.rstrip("/")),
                        "code_rev": code_rev,
                        "surface": name,
                        "surface_family": surface.family,
                        "condition": condition,
                        "item_id": item.item_id,
                        "decile": item.decile,
                        "split": item.split,
                        "images": trial.conversation.images,
                        "image_scores": item.image_scores,
                        "image_mean": item.image_mean,
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
                        print(f"  [{n_new}] {rate:.2f} trials/s", flush=True)
    finally:
        store.write_manifest(sys.argv, code_rev, adaptor.describe(), started,
                             finished_at=time.time(),
                             extra={"gate": gate_reports, "items": _abs(args.items),
                                    "surfaces": surface_names, "conditions": conditions,
                                    "n_new": n_new, "n_skipped": n_skip, "n_errors": n_err})
        store.close()
        adaptor.teardown()

    print(f"\n[run] new={n_new} skipped={n_skip} errors={n_err}")
    print(f"[run] {store.trials_path}")
    return 0 if n_err == 0 else 1


# --------------------------------------------------------------------------- #
def cmd_score(args: argparse.Namespace) -> int:
    store = RunStore(run_dir=_abs(args.run), conversations_dir=_abs(args.conversations))
    rows = list(store.read())
    if not rows:
        print(f"no trials in {store.trials_path}", file=sys.stderr)
        return 1

    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("surface"), row.get("condition"), row.get("decile"))].append(row)

    print(f"run      : {args.run}")
    print(f"trials   : {len(rows)}")
    manifest = os.path.join(_abs(args.run), "manifest.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as handle:
            payload = json.load(handle)
        print(f"code_rev : {payload.get('code_rev')}")
        print(f"adaptor  : {payload.get('adaptor', {}).get('name')} / "
              f"{payload.get('adaptor', {}).get('model')} / "
              f"probe={payload.get('adaptor', {}).get('probe')}")
    print()

    header = (f"{'surface':<16}{'cond':<6}{'dec':>4}{'n':>5}"
              f"{'outcome_mean':>14}{'outcome_sd':>12}{'s_txt_mean':>12}{'s_img_mean':>12}")
    print(header)
    print("-" * len(header))
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), _num(k[2]))):
        surface, condition, decile = key
        bucket = groups[key]
        outcomes = [r["outcome"]["value"] for r in bucket
                    if r.get("outcome") and r["outcome"].get("value") is not None]
        s_txt = [r["probe"]["s_txt"] for r in bucket
                 if r.get("probe") and r["probe"].get("s_txt") is not None]
        s_img = [r["probe"]["s_img"] for r in bucket
                 if r.get("probe") and r["probe"].get("s_img") is not None]
        print(f"{str(surface):<16}{str(condition):<6}{_fmt_int(decile):>4}{len(bucket):>5}"
              f"{_fmt(_mean(outcomes)):>14}{_fmt(_sd(outcomes)):>12}"
              f"{_fmt(_mean(s_txt)):>12}{_fmt(_mean(s_img)):>12}")

    # completeness, which is what acceptance item 5 is actually asking about
    n_logprob = sum(1 for r in rows if r.get("response", {}).get("logprobs"))
    n_stxt = sum(1 for r in rows
                 if r.get("probe") and r["probe"].get("s_txt") is not None)
    n_simg = sum(1 for r in rows
                 if r.get("probe") and r["probe"].get("s_img") is not None)
    print()
    print(f"records with logprobs   : {n_logprob}/{len(rows)}")
    print(f"records with probe.s_txt: {n_stxt}/{len(rows)}")
    print(f"records with probe.s_img: {n_simg}/{len(rows)}  "
          f"(null by construction in the no-image condition E)")
    print(f"records with errors     : {sum(1 for r in rows if r.get('error'))}/{len(rows)}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    print("bench judge: not implemented yet "
          "(planned: judges/speech_lean.py + judges/rewrite_bias.py, cached by judge_id)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print("bench report: not implemented yet "
          "(planned: dose-response by decile, placebo direction, mediation; needs a stats dep)")
    return 0


# --------------------------------------------------------------------------- #
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

    p = sub.add_parser("check", help="capability gate for one surface x adaptor pair")
    p.add_argument("--surface", required=True)
    p.add_argument("--adaptor", required=True)
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only read the first N CSV rows")
    p.add_argument("--out", default="items/")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("run", help="the only step that touches a GPU or the network")
    p.add_argument("--items", required=True)
    p.add_argument("--surface", required=True, help="comma separated")
    p.add_argument("--conditions", default="C")
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
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("judge", help="(stub)")
    p.add_argument("--run")
    p.add_argument("--surface")
    p.add_argument("--judge")
    p.add_argument("--adaptor")
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
