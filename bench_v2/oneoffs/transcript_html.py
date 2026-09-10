"""Render a run's transcript log as a single self-contained HTML file.

Reads ``transcripts.jsonl`` (as written by ``bench_v2.helpers.run``) plus the
items file the run used, and lays the personas out as a three-column board --
**low | mid | high**, one persona per cell, so row *i* lines up the *i*-th
persona of each bucket. Each persona's images and its four variant transcripts
are embedded in the cell.

Everything is embedded (downscaled images as base64, assigned once via JS), so
the page opens on a laptop with no cluster access and no sibling asset files.

    python -m bench_v2.oneoffs.transcript_html \
        --transcripts runs/bench_v2/s3_digest/v2/transcripts.jsonl \
        --items items/explore_extreme_v1.jsonl \
        --out /tmp/s3v2_transcripts.html
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
        raw = Path(path).read_bytes()
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
        print(f"[html] skip image {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def part_html(part: dict[str, Any], img_ids: dict[str, str]) -> str:
    kind = part.get("type")
    if kind == "image":
        src = part.get("image")
        ident = img_ids.get(src)
        if ident is None:
            return f'<span class="missing">[image missing: {esc(src)}]</span>'
        return f'<img class="imb" data-img="{ident}" alt="{esc(Path(str(src)).name)}">'
    if kind == "text":
        return f'<div class="text">{esc(part.get("text", ""))}</div>'
    return ""


def bubble(role: str, body: str, kind: str = "") -> str:
    return (f'<div class="bubble {esc(role)} {esc(kind)}">'
            f'<div class="role">{esc(role)}</div><div class="body">{body}</div></div>')


def render_transcript(row: dict[str, Any], img_ids: dict[str, str]) -> str:
    variant = row.get("variant") or {}
    scheme = str(variant.get("scheme", "?"))
    clause = str(variant.get("clause", "?"))
    out = [f'<article class="transcript" data-scheme="{esc(scheme)}" '
           f'data-clause="{esc(clause)}">']
    out.append(f'<h4>{esc(scheme)} · {"memory" if clause == "memory" else "no memory"}</h4>')
    if row.get("prefill"):
        out.append(bubble("assistant",
                          f'<pre class="resp">{esc(row["prefill"])}</pre>', kind="prefill"))
    for message in row.get("messages") or []:
        role = str(message.get("role", ""))
        content = message.get("content")
        if isinstance(content, list):
            body = "".join(part_html(p, img_ids) for p in content)
        else:
            body = f'<div class="text">{esc(content)}</div>'
        out.append(bubble(role, body))
    if row.get("error"):
        out.append(bubble("error", f'<div class="text">{esc(row["error"])}</div>'))
    else:
        out.append(bubble("assistant", f'<pre class="resp">{esc((row.get("response_text") or "").strip())}</pre>',
                          kind="response"))
    out.append('</article>')
    return "".join(out)


def render_cell(item_id: str, group: list[dict[str, Any]], meta: dict[str, Any],
                img_ids: dict[str, str]) -> str:
    cov = meta.get("covariates") or {}
    bucket = meta.get("bucket", cov.get("bucket", "?"))
    scores = meta.get("image_scores") or []
    extremes = cov.get("image_extreme") or []
    cats = ", ".join(cov.get("categories") or [])
    out = [f'<div class="cell" data-bucket="{esc(bucket)}">']
    out.append(f'<h3>{esc(item_id)}<span class="badge">{esc(bucket)}</span></h3>')
    out.append(f'<div class="cats">{esc(cats)}</div>')
    out.append('<div class="images">')
    paths = group[0].get("images") or []
    for i, path in enumerate(paths):
        ident = img_ids.get(path)
        cap = f"{i + 1}"
        if i < len(scores):
            cap += f" · μ{scores[i]:+.2f}"
        if i < len(extremes):
            cap += f" · e{extremes[i]:+.2f}"
        if ident is None:
            out.append(f'<figure><div class="missing">missing</div>'
                       f'<figcaption>{esc(cap)}</figcaption></figure>')
        else:
            out.append(f'<figure><img class="imb" data-img="{ident}" '
                       f'alt="{esc(Path(path).name)}">'
                       f'<figcaption>{esc(cap)}</figcaption></figure>')
    out.append('</div>')

    group.sort(key=lambda r: (str((r.get("variant") or {}).get("scheme")),
                              str((r.get("variant") or {}).get("clause"))))
    out.append('<div class="transcripts">')
    for row in group:
        out.append(render_transcript(row, img_ids))
    out.append('</div></div>')
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="transcripts.jsonl -> self-contained HTML board")
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--items", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all (per bucket)")
    parser.add_argument("--max-width", type=int, default=420)
    parser.add_argument("--quality", type=int, default=76)
    parser.add_argument("--title", default="s3_digest transcripts")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.transcripts).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    item_meta: dict[str, dict[str, Any]] = {}
    if args.items and Path(args.items).exists():
        for line in Path(args.items).read_text(encoding="utf-8").splitlines():
            if line.strip():
                blob = json.loads(line)
                item_meta[blob["item_id"]] = blob

    by_item: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)

    # bucket -> ordered personas (row i lines up the i-th persona of each bucket)
    by_bucket: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for item_id in by_item:
        meta = item_meta.get(item_id, {})
        bucket = meta.get("bucket", (meta.get("covariates") or {}).get("bucket"))
        if bucket not in by_bucket:
            bucket = "low"
        by_bucket[bucket].append(item_id)
    if args.max_items:
        by_bucket = {b: ids[: args.max_items] for b, ids in by_bucket.items()}

    # embed each unique image once
    img_payload: dict[str, str] = {}
    img_ids: dict[str, str] = {}
    for item_id, group in by_item.items():
        for row in group:
            for path in row.get("images") or []:
                if path in img_ids:          # dedup by PATH (img_payload is keyed by ident)
                    continue
                data = thumb_b64(path, args.max_width, args.quality)
                if data is None:
                    continue
                ident = f"img_{len(img_ids)}"
                img_ids[path] = ident
                img_payload[ident] = data

    n_personas = sum(len(v) for v in by_bucket.values())
    n_transcripts = sum(len(by_item[i]) for i in by_item)
    n_rows = max((len(v) for v in by_bucket.values()), default=0)

    out: list[str] = []
    out.append(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(args.title)}</title>
<style>
:root {{ --fg:#1b1f24; --muted:#6a737d; --line:#e1e4e8; --bg:#f6f8fa; --card:#fff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:var(--fg); background:var(--bg); }}
header {{ position:sticky; top:0; z-index:20; background:#fff; border-bottom:1px solid var(--line);
  padding:10px 16px; }}
h1 {{ font-size:16px; margin:0 0 3px; }}
.sub {{ color:var(--muted); font-size:12px; }}
.filterbar {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:14px; font-size:12px; align-items:center; }}
.filterbar label {{ color:var(--muted); margin-right:4px; }}
main {{ padding:14px; }}
.colheads {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;
  position:sticky; top:78px; z-index:15; background:var(--bg); padding:6px 0; }}
.colhead {{ font-weight:600; text-transform:uppercase; letter-spacing:.06em; font-size:12px;
  color:#364152; background:#eef2f6; border:1px solid var(--line); border-radius:8px;
  padding:5px 10px; }}
.board {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; align-items:start; }}
.cell {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:var(--card);
  min-width:0; }}
.cell h3 {{ font-size:12.5px; margin:0 0 4px; word-break:break-all; }}
.badge {{ font-size:10px; padding:1px 6px; border-radius:9px; background:#eef2f6; color:#364152;
  margin-left:6px; }}
.cats {{ color:var(--muted); font-size:11px; margin-bottom:8px; min-height:15px; }}
.images {{ display:flex; gap:6px; margin-bottom:8px; }}
.images figure {{ margin:0; width:33%; }}
.images img {{ width:100%; border-radius:5px; border:1px solid var(--line); display:block; }}
.images figcaption {{ font-size:10px; color:var(--muted); margin-top:2px; }}
.transcripts {{ display:flex; flex-direction:column; gap:8px; }}
.transcript {{ border:1px solid var(--line); border-radius:7px; padding:7px; background:#fcfdff; }}
.transcript h4 {{ font-size:10px; margin:0 0 5px; color:#364152; text-transform:uppercase;
  letter-spacing:.05em; }}
.bubble {{ border-radius:7px; padding:6px 7px; margin-bottom:5px; font-size:11.5px; }}
.bubble.user {{ background:#eef4ff; }}
.bubble.assistant {{ background:#f0fff4; }}
.bubble.assistant.response {{ background:#eafff1; border:1px solid #b7efc5; }}
.bubble.assistant.prefill {{ background:#fff8e6; border:1px solid #f0e0a8; }}
.bubble.error {{ background:#fff1f0; border:1px solid #f3b8b3; }}
.bubble .role {{ font-size:9px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
  margin-bottom:2px; }}
.bubble .body img {{ max-width:70px; border-radius:4px; vertical-align:middle; margin:1px 2px 1px 0; }}
.text {{ white-space:pre-wrap; }}
pre.resp {{ white-space:pre-wrap; word-wrap:break-word; margin:0; font:inherit; }}
.missing {{ color:#b3261e; font-size:10px; }}
.hidden {{ display:none !important; }}
@media (max-width:900px) {{
  .board, .colheads {{ grid-template-columns:1fr; }}
  .colheads .colhead:nth-child(2), .colheads .colhead:nth-child(3) {{ display:none; }}
}}
</style></head><body>
<header>
  <h1>{esc(args.title)}</h1>
  <div class="sub">{n_personas} personas ({len(by_bucket['low'])} low / {len(by_bucket['mid'])} mid / {len(by_bucket['high'])} high)
    &middot; {n_transcripts} transcripts &middot; {len(img_ids)} images &middot; {n_rows} rows</div>
  <div class="filterbar">
    <span><label>scheme</label>
      <select id="f-scheme"><option value="">all</option>
      <option>chat</option><option>agentic</option></select></span>
    <span><label>variant</label>
      <select id="f-clause"><option value="">all</option>
      <option>bare</option><option>memory</option></select></span>
    <span id="count" class="sub"></span>
  </div>
</header>
<main>
<div class="colheads">
  <div class="colhead">low <span class="sub">{len(by_bucket['low'])}</span></div>
  <div class="colhead">mid <span class="sub">{len(by_bucket['mid'])}</span></div>
  <div class="colhead">high <span class="sub">{len(by_bucket['high'])}</span></div>
</div>
<div class="board">
""")

    for i in range(n_rows):
        for bucket in BUCKETS:
            ids = by_bucket[bucket]
            if i < len(ids):
                item_id = ids[i]
                out.append(render_cell(item_id, by_item[item_id], item_meta.get(item_id, {}), img_ids))
            else:
                out.append(f'<div class="cell" data-bucket="{bucket}"></div>')

    out.append("""</div></main>
<script>
const IMGS = /*__IMAGES__*/{};
document.querySelectorAll('img.imb').forEach(function(el) {
  var d = IMGS[el.dataset.img];
  if (d) { el.src = d; }
});
function apply() {
  var s = document.getElementById('f-scheme').value;
  var c = document.getElementById('f-clause').value;
  var shown = 0;
  document.querySelectorAll('div.cell').forEach(function(cell) {
    cell.querySelectorAll('article.transcript').forEach(function(t) {
      var ok = (s === '' || t.dataset.scheme === s) && (c === '' || t.dataset.clause === c);
      t.classList.toggle('hidden', !ok);
      if (ok) { shown++; }
    });
  });
  document.getElementById('count').textContent = shown + ' transcripts shown';
}
['f-scheme','f-clause'].forEach(function(id) {
  document.getElementById(id).addEventListener('change', apply);
});
apply();
</script>
</body></html>""")

    doc = "".join(out).replace("/*__IMAGES__*/{}", json.dumps(img_payload))
    Path(args.out).write_text(doc, encoding="utf-8")
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"[html] wrote {args.out} ({n_personas} personas in {n_rows} rows x 3 cols, "
          f"{n_transcripts} transcripts, {len(img_ids)} images, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
