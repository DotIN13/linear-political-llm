"""Build ``statements_v2.jsonl``: all 240 statements of the paper's pool.

statements_v1 is eighteen hand-picked texts, three per domain (see
``build_statements.py``). This file is the whole pool, 40 lawmakers x 6 domains,
so a run reads every statement the paper used.

The eighteen v1 picks keep their v1 ``sid`` (``st01l`` ...) and their hand
``text_lean``, so their trials compare directly with v1-v3. The other 222 get
``st<domain><nn>`` in file order and ``text_lean: null``: nobody has read them
for lean, and the author's DW-NOMINATE score and party are what they carry.

    python -m bench_v2.tasks.s16_bias_detect.build_statements_all
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict

from bench_v2.tasks.s16_bias_detect.build_statements import (
    DOMAINS, MEMBERS, PICKS, STATEMENTS, HERE, clean, fullname)

OUT = HERE / "statements_v2.jsonl"


def main() -> int:
    nominate: dict[str, tuple[float, str]] = {}
    with MEMBERS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["nominate_dim1"]:
                nominate.setdefault(fullname(row["bioname"]),
                                    (float(row["nominate_dim1"]), row["party_code"]))
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    with STATEMENTS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_domain[row["domain"]].append(row)

    picked = {(domain, name): lean for (domain, lean), (name, _) in PICKS.items()}
    rows = []
    for d, domain in enumerate(DOMAINS, start=1):
        for n, row in enumerate(by_domain[domain], start=1):
            name = row["name"]
            lean = picked.get((domain, name))
            dim1, party = nominate.get(name, (None, None))
            rows.append({
                "sid": f"st{d:02d}{lean[0]}" if lean else f"st{d:02d}_{n:02d}",
                "domain": domain,
                "text_lean": lean,
                "author": name,
                "author_party": {"100": "D", "200": "R"}.get(party, party),
                "author_nominate_dim1": dim1,
                "statement": clean(row["response"]),
            })
    if len(rows) != 240 or len({r["sid"] for r in rows}) != 240:
        raise ValueError(f"expected 240 unique statements, got {len(rows)}")
    if sum(1 for r in rows if r["text_lean"]) != len(PICKS):
        raise ValueError("not every v1 pick was found in the pool")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    meta = {
        "version": "statements_v2",
        "n_items": len(rows),
        "source": "data/policy_statements_240.csv (huggingface.co/datasets/DotIN13/political-statements)",
        "source_sha256": hashlib.sha256(STATEMENTS.read_bytes()).hexdigest()[:16],
        "paper": "Zhang, Probing Political Ideology in Large Language Models, Findings of EMNLP 2025, 23349-23360",
        "rule": "every statement in the pool; the 18 statements_v1 picks keep their sid and hand text_lean, the rest have text_lean null",
        "unmatched_authors": sorted({r["author"] for r in rows if r["author_nominate_dim1"] is None}),
        "built_by": "bench_v2/tasks/s16_bias_detect/build_statements_all.py",
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} statements to {OUT}; unmatched authors: {meta['unmatched_authors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
