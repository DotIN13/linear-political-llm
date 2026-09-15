"""Render a run's transcripts as one self-contained HTML board.

A generalization of ``transcript_html.py`` for the tasks whose prompt variant is
a *question* (s7_family_chat's twelve messages, s8_letter_answered's twelve
concerns) rather than s3's shuffled-headline list. It reads
``transcripts.jsonl`` plus (optionally) the items file and ``trials.jsonl``, and
lays the personas out as a three-column board -- **low | mid | high** -- one
persona per cell, row *i* lining up the *i*-th persona of each bucket.

Each persona cell shows its images and, for every question, the model's answer
under each scheme x persona arm, with the judge's ``lean`` chip when available.
The whole thing is embedded (downscaled images as base64, all records as JSON),
so the page opens with no cluster access and no sibling assets; filtering is
client-side and instant.

    python -m bench_v2.oneoffs.transcript_board \
        --transcripts runs/bench_v2/s7_family_chat/v2/transcripts.jsonl \
        --trials runs/bench_v2/s7_family_chat/v2/trials.jsonl \
        --items items/explore_bucket_v1.jsonl \
        --out /tmp/s7v2_transcripts.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUCKETS = ("low", "mid", "high")


def thumb_b64(path: str, max_width: int, quality: int) -> str | None:
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - fall back to the full file
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            print(f"[board] skip image {path}: {exc}", file=sys.stderr)
            return None
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, max(1, int(img.height * ratio))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        print(f"[board] skip image {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def bucket_of(item_id: str, meta: dict[str, Any]) -> str:
    cov = meta.get("covariates") or {}
    b = meta.get("bucket", cov.get("bucket"))
    if b in BUCKETS:
        return str(b)
    for name, abbrev in (("low", "_lo_"), ("mid", "_mid_"), ("high", "_hi_")):
        if abbrev in item_id:
            return name
    return "mid"


def lean_chip(row: dict[str, Any]) -> str | None:
    label = (((row.get("judge") or {}).get("labels")) or {}).get("lean")
    return str(label) if label else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="transcripts.jsonl -> self-contained HTML board")
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--items", default="")
    ap.add_argument("--trials", default="", help="trials.jsonl; joins judge lean by trial_key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all personas per bucket")
    ap.add_argument("--max-width", type=int, default=340)
    ap.add_argument("--quality", type=int, default=74)
    ap.add_argument("--title", default="transcripts")
    args = ap.parse_args(argv)

    trows = [json.loads(line) for line in Path(args.transcripts).read_text(
        encoding="utf-8").splitlines() if line.strip()]

    item_meta: dict[str, dict[str, Any]] = {}
    if args.items and Path(args.items).exists():
        for line in Path(args.items).read_text(encoding="utf-8").splitlines():
            if line.strip():
                blob = json.loads(line)
                item_meta[blob["item_id"]] = blob

    trials: dict[str, dict[str, Any]] = {}
    if args.trials and Path(args.trials).exists():
        for line in Path(args.trials).read_text(encoding="utf-8").splitlines():
            if line.strip():
                blob = json.loads(line)
                trials[str(blob.get("trial_key"))] = blob
        print(f"[board] judge rows for {len(trials)} trials", file=sys.stderr)

    # --- collect the questions (prompt variants), in a stable order -----------
    questions: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    persona_rows: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in trows:
        variant = row.get("variant") or {}
        mid = str(variant.get("question", "q0"))
        if mid not in questions:
            ds = ((trials.get(str(row.get("trial_key"))) or {}).get("meta") or {}).get("dataset") or {}
            questions[mid] = {
                "mid": mid,
                "topic": ds.get("topic"),
                "domain": ds.get("domain"),
                "concern": ds.get("concern") or ds.get("message"),
            }
        persona_rows.setdefault(str(row["item_id"]), []).append(row)

    # --- bucket -> ordered personas ------------------------------------------
    by_bucket: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for item_id in persona_rows:
        by_bucket[bucket_of(item_id, item_meta.get(item_id, {}))].append(item_id)
    by_bucket = {b: sorted(ids) for b, ids in by_bucket.items()}
    if args.max_items:
        by_bucket = {b: ids[: args.max_items] for b, ids in by_bucket.items()}
    keep = {i for ids in by_bucket.values() for i in ids}

    # --- embed each unique image once ----------------------------------------
    img_payload: dict[str, str] = {}
    img_ids: dict[str, str] = {}
    for item_id in keep:
        for row in persona_rows[item_id]:
            for path in row.get("images") or []:
                if path in img_ids:
                    continue
                data = thumb_b64(path, args.max_width, args.quality)
                if data is None:
                    continue
                ident = f"img_{len(img_ids)}"
                img_ids[path] = ident
                img_payload[ident] = data

    # --- assemble the JSON payload -------------------------------------------
    personas_payload: dict[str, dict[str, Any]] = {}
    for item_id in keep:
        meta = item_meta.get(item_id, {})
        cov = meta.get("covariates") or {}
        group = persona_rows[item_id]
        paths = group[0].get("images") or []
        scores = meta.get("image_scores") or []
        personas_payload[item_id] = {
            "id": item_id,
            "bucket": bucket_of(item_id, meta),
            "cats": ", ".join(cov.get("categories") or []),
            "images": [{"ident": img_ids.get(p), "name": Path(str(p)).name,
                        "score": (scores[i] if i < len(scores) else None)}
                       for i, p in enumerate(paths)],
        }

    records = []
    for item_id in keep:
        for row in persona_rows[item_id]:
            variant = row.get("variant") or {}
            trial = trials.get(str(row.get("trial_key"))) or {}
            ds = (trial.get("meta") or {}).get("dataset") or {}
            records.append({
                "item": item_id,
                "mid": str(variant.get("question", "q0")),
                "scheme": str(variant.get("scheme", "?")),
                "clause": str(variant.get("clause", "?")),
                "lean": lean_chip(trial),
                "response": (row.get("response_text") or "").strip(),
                "error": row.get("error"),
                "topic": ds.get("topic"),
            })

    payload = {
        "buckets": BUCKETS,
        "byBucket": {b: by_bucket[b] for b in BUCKETS},
        "personas": personas_payload,
        "questions": list(questions.values()),
        "records": records,
        "images": img_payload,
        "schemes": sorted({r["scheme"] for r in records}),
        "clauses": sorted({r["clause"] for r in records}),
    }

    n_personas = sum(len(v) for v in by_bucket.values())
    n_rows = max((len(v) for v in by_bucket.values()), default=0)
    doc = _DOC.replace("/*__PAYLOAD__*/null", json.dumps(payload, ensure_ascii=False))
    doc = doc.replace("__TITLE__", html.escape(args.title))
    doc = doc.replace("__COUNTS__",
                      f"{n_personas} personas "
                      f"({len(by_bucket['low'])} low / {len(by_bucket['mid'])} mid / "
                      f"{len(by_bucket['high'])} high) &middot; {len(records)} transcripts "
                      f"&middot; {len(questions)} questions &middot; {len(img_ids)} images "
                      f"&middot; {n_rows} rows")
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"[board] wrote {args.out} ({n_personas} personas, {len(records)} transcripts, "
          f"{len(img_ids)} images, {Path(args.out).stat().st_size / 1e6:.1f} MB)")
    return 0


_DOC = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --fg:#1b1f24; --muted:#6a737d; --line:#e1e4e8; --bg:#f6f8fa; --card:#fff;
        --left:#dbeafe; --right:#fee2e2; }
* { box-sizing:border-box; }
body { margin:0; font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:var(--fg); background:var(--bg); }
header { position:sticky; top:0; z-index:20; background:#fff; border-bottom:1px solid var(--line);
  padding:10px 16px; }
h1 { font-size:16px; margin:0 0 3px; }
.sub { color:var(--muted); font-size:12px; }
.filterbar { margin-top:8px; display:flex; flex-wrap:wrap; gap:14px; font-size:12px; align-items:center; }
.filterbar label { color:var(--muted); margin-right:4px; }
select { font:inherit; padding:2px 4px; }
main { padding:14px; }
.colheads { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;
  position:sticky; top:112px; z-index:15; background:var(--bg); padding:6px 0; }
.colhead { font-weight:600; text-transform:uppercase; letter-spacing:.06em; font-size:12px;
  color:#364152; background:#eef2f6; border:1px solid var(--line); border-radius:8px; padding:5px 10px; }
.board { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; align-items:start; }
.cell { border:1px solid var(--line); border-radius:10px; padding:10px; background:var(--card);
  min-width:0; }
.cell h3 { font-size:12.5px; margin:0 0 4px; word-break:break-all; }
.badge { font-size:10px; padding:1px 6px; border-radius:9px; background:#eef2f6; color:#364152;
  margin-left:6px; }
.cats { color:var(--muted); font-size:11px; margin-bottom:8px; min-height:15px; }
.images { display:flex; gap:6px; margin-bottom:8px; }
.images figure { margin:0; width:33%; }
.images img { width:100%; border-radius:5px; border:1px solid var(--line); display:block; }
.images figcaption { font-size:10px; color:var(--muted); margin-top:2px; }
.qblock { border-top:1px dashed var(--line); padding-top:6px; margin-top:6px; }
.qhead { font-size:11px; color:#364152; cursor:pointer; display:flex; gap:6px; align-items:baseline; }
.qhead .qtext { color:var(--muted); font-weight:400; }
.qtoggle { color:var(--muted); font-size:10px; }
.answers { margin-top:5px; display:flex; flex-direction:column; gap:5px; }
.ans { border:1px solid var(--line); border-radius:7px; padding:5px 7px; background:#fcfdff; }
.ans .meta { display:flex; gap:8px; align-items:center; font-size:9.5px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); margin-bottom:3px; }
.ans .tag { background:#eef2f6; border-radius:6px; padding:1px 5px; }
.ans .tag.memory { background:#efe7fb; }
.ans .lean { margin-left:auto; border-radius:6px; padding:1px 6px; font-variant-numeric:tabular-nums; }
.ans .lean.neg { background:var(--left); color:#1a3a8f; }
.ans .lean.pos { background:var(--right); color:#8f1a1a; }
.ans .lean.zero { background:#eef2f6; color:#364152; }
.ans .lean.none { background:#f3f4f6; color:var(--muted); }
.ans .resp { white-space:pre-wrap; }
.ans .err { color:#b3261e; }
.hidden { display:none !important; }
@media (max-width:980px) {
  .board, .colheads { grid-template-columns:1fr; }
  .colheads .colhead:nth-child(2), .colheads .colhead:nth-child(3) { display:none; }
}
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">__COUNTS__</div>
  <div class="filterbar">
    <span><label>scheme</label><select id="f-scheme"></select></span>
    <span><label>variant</label><select id="f-clause"></select></span>
    <span><label>question</label><select id="f-mid"></select></span>
    <span><label>bucket</label><select id="f-bucket"></select></span>
    <span id="count" class="sub"></span>
  </div>
</header>
<main>
<div class="colheads">
  <div class="colhead">low <span class="sub" id="c-low"></span></div>
  <div class="colhead">mid <span class="sub" id="c-mid"></span></div>
  <div class="colhead">high <span class="sub" id="c-high"></span></div>
</div>
<div class="board" id="board"></div>
</main>
<script>
const DATA = /*__PAYLOAD__*/null;

function esc(s) { return (s == null ? '' : String(s)); }
function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}
function leanClass(l) {
  if (l == null) return 'none';
  if (l === 'center') return 'zero';
  return /left/.test(l) ? 'neg' : 'pos';
}
const LEAN = { far_left:-1, left:-2/3, lean_left:-1/3, center:0,
               lean_right:1/3, right:2/3, far_right:1 };

// index records by item -> mid -> scheme|clause
const byItem = new Map();
for (const r of DATA.records) {
  if (!byItem.has(r.item)) byItem.set(r.item, new Map());
  const m = byItem.get(r.item);
  if (!m.has(r.mid)) m.set(r.mid, new Map());
  m.get(r.mid).set(r.scheme + '|' + r.clause, r);
}

function fillSelect(sel, values, allLabel) {
  const o0 = document.createElement('option'); o0.value = ''; o0.textContent = allLabel;
  sel.appendChild(o0);
  for (const v of values) { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }
}
fillSelect(document.getElementById('f-scheme'), DATA.schemes, 'all');
fillSelect(document.getElementById('f-clause'), DATA.clauses, 'all');
fillSelect(document.getElementById('f-mid'), DATA.questions.map(q => q.mid), 'all');

const bucketSel = document.getElementById('f-bucket');
fillSelect(bucketSel, DATA.buckets, 'all');

const qById = new Map(DATA.questions.map(q => [q.mid, q]));

function buildBoard() {
  const board = document.getElementById('board');
  board.innerHTML = '';
  const fScheme = document.getElementById('f-scheme').value;
  const fClause = document.getElementById('f-clause').value;
  const fMid = document.getElementById('f-mid').value;
  const fBucket = document.getElementById('f-bucket').value;

  let shown = 0, personas = 0, transcripts = 0;
  for (const b of DATA.buckets) {
    document.getElementById('c-' + b).textContent = DATA.byBucket[b].length;
  }
  const maxRows = Math.max(...DATA.buckets.map(b => DATA.byBucket[b].length));
  for (let i = 0; i < maxRows; i++) {
    for (const b of DATA.buckets) {
      const id = DATA.byBucket[b][i];
      if (!id) { board.appendChild(el('div', 'cell')); continue; }
      if (fBucket && fBucket !== b) { continue; }
      const p = DATA.personas[id];
      const cell = el('div', 'cell');
      const h = el('h3', null, id); h.appendChild(el('span', 'badge', b)); cell.appendChild(h);
      cell.appendChild(el('div', 'cats', p.cats || ''));
      const imgs = el('div', 'images');
      for (const im of p.images) {
        const fig = el('figure');
        const img = el('img');
        if (im.ident && DATA.images) { /* images fetched from global */ }
        if (im.ident) img.dataset.ident = im.ident;
        img.alt = im.name;
        fig.appendChild(img);
        const cap = im.score == null ? im.name : (im.score >= 0 ? '+' : '') + im.score.toFixed(2);
        fig.appendChild(el('figcaption', null, cap));
        imgs.appendChild(fig);
      }
      cell.appendChild(imgs);

      const mids = fMid ? [fMid] : DATA.questions.map(q => q.mid);
      for (const mid of mids) {
        const arms = byItem.get(id) && byItem.get(id).get(mid);
        if (!arms) continue;
        const q = qById.get(mid) || {mid, topic:'', concern:''};
        const block = el('div', 'qblock');
        const qh = el('div', 'qhead');
        qh.appendChild(el('span', 'qtoggle', '\u25b8'));
        qh.appendChild(el('span', null, mid + (q.topic ? ' / ' + q.topic : '') + (q.domain ? ' \u00b7 ' + q.domain : '')));
        if (q.concern) qh.appendChild(el('span', 'qtext', '\u201c' + q.concern + '\u201d'));
        block.appendChild(qh);
        const answers = el('div', 'answers');
        let local = 0;
        for (const scheme of DATA.schemes) {
          for (const clause of DATA.clauses) {
            const r = arms.get(scheme + '|' + clause);
            if (!r) continue;
            if (fScheme && scheme !== fScheme) continue;
            if (fClause && clause !== fClause) continue;
            const a = el('div', 'ans');
            const meta = el('div', 'meta');
            meta.appendChild(el('span', 'tag', scheme));
            meta.appendChild(el('span', 'tag' + (clause === 'memory' ? ' memory' : ''), clause));
            const l = el('span', 'lean ' + leanClass(r.lean), r.lean == null ? 'lean: –' : r.lean);
            meta.appendChild(l);
            a.appendChild(meta);
            a.appendChild(el('div', 'resp' + (r.error ? ' err' : ''), r.error || r.response || '(empty)'));
            answers.appendChild(a);
            local++; transcripts++; shown++;
          }
        }
        if (local === 0) continue;
        block.appendChild(answers);
        qh.addEventListener('click', () => {
          answers.classList.toggle('hidden');
          qh.querySelector('.qtoggle').textContent = answers.classList.contains('hidden') ? '\u25b8' : '\u25be';
        });
        cell.appendChild(block);
      }
      personas++;
      board.appendChild(cell);
    }
  }
  document.getElementById('count').textContent =
    shown + ' transcripts shown \u00b7 ' + personas + ' personas';
}
// images are assigned after layout so the base64 map can live in one place
const IMGS = DATA.images || {};
function assignImages() {
  document.querySelectorAll('img[data-ident]').forEach(img => {
    const d = IMGS[img.dataset.ident];
    if (d) img.src = d;
  });
}
function rebuild() { buildBoard(); assignImages(); }
['f-scheme','f-clause','f-mid','f-bucket'].forEach(id =>
  document.getElementById(id).addEventListener('change', rebuild));
rebuild();
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
