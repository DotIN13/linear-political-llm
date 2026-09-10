"""Render a run's transcript log as a single self-contained HTML file.

Reads ``transcripts.jsonl`` (as written by ``bench_v2.helpers.run``) plus the
items file the run used, groups the four variants of each persona, and embeds
downscaled copies of that persona's images as base64 -- so the page opens on a
laptop with no cluster access and no sibling asset files.

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


def part_html(part: dict[str, Any], img_ids: dict[str, str], images: list[str]) -> str:
    kind = part.get("type")
    if kind == "image":
        src = part.get("image")
        ident = img_ids.get(src)
        if ident is None:
            return f'<span class="missing">[image missing: {esc(src)}]</span>'
        return (f'<img class="imb" data-img="{ident}" '
                f'alt="{esc(Path(str(src)).name)}">')
    if kind == "text":
        return f'<div class="text">{esc(part.get("text", ""))}</div>'
    return ""


def bubble(role: str, body: str, kind: str = "") -> str:
    return (f'<div class="bubble {esc(role)} {esc(kind)}">'
            f'<div class="role">{esc(role)}</div><div class="body">{body}</div></div>')


def render_response(text: str) -> str:
    return f'<pre class="resp">{esc(text.strip())}</pre>'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="transcripts.jsonl -> self-contained HTML")
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--items", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--max-width", type=int, default=480)
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

    # group transcripts by item, keep variant order stable
    by_item: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    if args.max_items:
        by_item = OrderedDict(list(by_item.items())[: args.max_items])

    # embed each unique image once
    img_payload: dict[str, str] = {}
    img_ids: dict[str, str] = {}
    for item_id, group in by_item.items():
        for row in group:
            for path in row.get("images") or []:
                if path in img_payload:
                    continue
                data = thumb_b64(path, args.max_width, args.quality)
                if data is None:
                    continue
                ident = f"img_{len(img_ids)}"
                img_ids[path] = ident
                img_payload[ident] = data

    n_items = len(by_item)
    n_transcripts = sum(len(g) for g in by_item.values())
    out = []
    out.append(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(args.title)}</title>
<style>
:root {{ --fg:#1b1f24; --muted:#6a737d; --line:#e1e4e8; --bg:#f6f8fa; --card:#fff; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:var(--fg); background:var(--bg); }}
header {{ position: sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line);
  padding:12px 18px; }}
h1 {{ font-size:17px; margin:0 0 4px; }}
.sub {{ color:var(--muted); font-size:12px; }}
.filterbar {{ margin-top:10px; display:flex; flex-wrap:wrap; gap:14px; font-size:12px; }}
.filterbar label {{ color:var(--muted); margin-right:4px; }}
main {{ padding:18px; max-width:1100px; margin:0 auto; }}
.item {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin-bottom:18px; }}
.item h2 {{ font-size:15px; margin:0 0 6px; }}
.badge {{ font-size:11px; padding:1px 7px; border-radius:10px; background:#eef2f6;
  color:#364152; margin-left:6px; vertical-align:middle; }}
.cats {{ color:var(--muted); font-size:12px; margin-bottom:10px; }}
.images {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
.images figure {{ margin:0; width:210px; }}
.images img {{ width:100%; border-radius:6px; border:1px solid var(--line); display:block; }}
.images figcaption {{ font-size:11px; color:var(--muted); margin-top:3px; }}
.transcripts {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
@media (max-width:860px) {{ .transcripts {{ grid-template-columns:1fr; }} }}
.transcript {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fcfdff; }}
.transcript h3 {{ font-size:12px; margin:0 0 8px; color:#364152; text-transform:uppercase;
  letter-spacing:.04em; }}
.bubble {{ border-radius:8px; padding:7px 9px; margin-bottom:7px; font-size:12.5px; }}
.bubble.user {{ background:#eef4ff; }}
.bubble.assistant {{ background:#f0fff4; }}
.bubble.assistant.response {{ background:#eafff1; border:1px solid #b7efc5; }}
.bubble .role {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--muted); margin-bottom:3px; }}
.bubble .body img {{ max-width:96px; border-radius:4px; vertical-align:middle; margin:2px 3px 2px 0; }}
.text {{ white-space:pre-wrap; }}
pre.resp {{ white-space:pre-wrap; word-wrap:break-word; margin:0; font:inherit; }}
.missing {{ color:#b3261e; font-size:11px; }}
.lightbox {{ position:fixed; inset:0; background:rgba(0,0,0,.82); display:none;
  align-items:center; justify-content:center; z-index:100; }}
.lightbox:target {{ display:flex; }}
.lightbox img {{ max-width:92vw; max-height:92vh; }}
.hidden {{ display:none !important; }}
</style></head><body>
<header>
  <h1>{esc(args.title)}</h1>
  <div class="sub">{n_items} personas &times; {n_transcripts} transcripts &middot;
    {len(img_ids)} images embedded &middot; qwen3-vl-8b-instruct / vLLM</div>
  <div class="filterbar">
    <span><label>bucket</label>
      <select id="f-bucket"><option value="">all</option>
      <option>low</option><option>mid</option><option>high</option></select></span>
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
""")

    for item_id, group in by_item.items():
        meta = item_meta.get(item_id, {})
        cov = meta.get("covariates") or {}
        bucket = meta.get("bucket", cov.get("bucket", "?"))
        scores = meta.get("image_scores") or []
        extremes = cov.get("image_extreme") or []
        cats = ", ".join(cov.get("categories") or [])
        out.append(f'<section class="item" data-bucket="{esc(bucket)}">')
        out.append(f'<h2>{esc(item_id)}'
                   f'<span class="badge">{esc(bucket)}</span>'
                   f'<span class="badge">{len(group)} transcripts</span></h2>')
        out.append(f'<div class="cats">categories: {esc(cats)}</div>')
        out.append('<div class="images">')
        paths = (group[0].get("images") or [])
        for i, path in enumerate(paths):
            ident = img_ids.get(path)
            cap = f"image {i + 1}"
            if i < len(scores):
                cap += f" · mean {scores[i]:+.2f}"
            if i < len(extremes):
                cap += f" · extreme {extremes[i]:+.2f}"
            if ident is None:
                out.append(f'<figure><div class="missing">missing image</div>'
                           f'<figcaption>{esc(cap)}</figcaption></figure>')
            else:
                out.append(f'<figure><img class="imb" data-img="{ident}" '
                           f'alt="{esc(Path(path).name)}">'
                           f'<figcaption>{esc(cap)}</figcaption></figure>')
        out.append('</div>')

        out.append('<div class="transcripts">')
        group.sort(key=lambda r: (str((r.get("variant") or {}).get("scheme")),
                                  str((r.get("variant") or {}).get("clause"))))
        for row in group:
            variant = row.get("variant") or {}
            scheme = str(variant.get("scheme", "?"))
            clause = str(variant.get("clause", "?"))
            out.append(f'<article class="transcript" data-scheme="{esc(scheme)}" '
                       f'data-clause="{esc(clause)}">')
            out.append(f'<h3>{esc(scheme)} · {"with memory" if clause == "memory" else "no memory"}</h3>')
            if row.get("prefill"):
                out.append(f'<div class="bubble assistant"><div class="role">prefill</div>'
                           f'<div class="body"><pre class="resp">{esc(row["prefill"])}</pre></div></div>')
            for message in row.get("messages") or []:
                role = str(message.get("role", ""))
                content = message.get("content")
                if isinstance(content, list):
                    body = "".join(part_html(p, img_ids, paths) for p in content)
                else:
                    body = f'<div class="text">{esc(content)}</div>'
                out.append(bubble(role, body))
            if row.get("error"):
                out.append(bubble("error", f'<div class="text">{esc(row["error"])}</div>'))
            else:
                out.append(bubble("assistant", render_response(row.get("response_text") or ""),
                                  kind="response"))
            out.append('</article>')
        out.append('</div></section>')

    out.append("""</main>
<script>
const IMGS = /*__IMAGES__*/{};
document.querySelectorAll('img.imb').forEach(function(el) {
  var d = IMGS[el.dataset.img];
  if (d) { el.src = d; }
});
function apply() {
  var b = document.getElementById('f-bucket').value;
  var s = document.getElementById('f-scheme').value;
  var c = document.getElementById('f-clause').value;
  var shown = 0;
  document.querySelectorAll('section.item').forEach(function(sec) {
    var itemVisible = (b === '' || sec.dataset.bucket === b);
    var any = false;
    sec.querySelectorAll('article.transcript').forEach(function(t) {
      var ok = itemVisible && (s === '' || t.dataset.scheme === s)
               && (c === '' || t.dataset.clause === c);
      t.classList.toggle('hidden', !ok);
      if (ok) { any = true; shown++; }
    });
    sec.classList.toggle('hidden', !any);
  });
  document.getElementById('count').textContent = shown + ' transcripts shown';
}
['f-bucket','f-scheme','f-clause'].forEach(function(id) {
  document.getElementById(id).addEventListener('change', apply);
});
apply();
</script>
</body></html>""")

    doc = "".join(out).replace("/*__IMAGES__*/{}", json.dumps(img_payload))
    Path(args.out).write_text(doc, encoding="utf-8")
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"[html] wrote {args.out} ({n_items} personas, {n_transcripts} transcripts, "
          f"{len(img_ids)} images, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
