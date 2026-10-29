"""face_swap stimuli: every person on every background, from EasyPortrait.

Five people whose own patches read most liberal and five most conservative, crossed
with five backgrounds read most liberal and five most conservative (the background
patches of other photos), give 100 composites; each person on their own background
adds 10 checks. Scores are Qwen3-VL's combined-ideology headwise probe on the image's
own tokens (``image_fg_mean`` and ``image_bg_mean`` in the EasyPortrait token CSV),
read only to pick the extremes.

Every image goes through the same edit, the checks included, so the editing cannot
explain a difference between cells:

1. the background photo's own person is cut out (its EasyPortrait mask, grown by a few
   pixels) and the hole filled by OpenCV inpainting;
2. the new person is cut out with its own mask, scaled so its height matches the
   person it replaces, aligned on the bottom centre of that person, and pasted with a
   soft edge;
3. the result is saved with its longer side at 800 pixels, as the token scoring did.

    python -m bench_llm.datasets.face_swap.build select     # writes selection.json beside this file
    python -m bench_llm.datasets.face_swap.build build      # writes data/face_swap/images/ and manifest.csv
    python -m bench_llm.datasets.face_swap.build jsonl      # writes data/probes/face_swap_score_data.jsonl
    python -m bench_llm.datasets.face_swap.build build --version v2   # data/face_swap_v2/
    python -m bench_llm.datasets.face_swap.build jsonl --version v2   # data/probes/face_swap_v2_score_data.jsonl
    python -m bench_llm.datasets.face_swap.build frames                # data/face_swap_v2/frames/ and masks/

``frames`` writes the 20 source photos themselves, unedited, in v2's frame, with their person
masks: what the token-swap task (``bench_llm/tasks/face_swap/v2``) encodes and splices.

v2 keeps each person exactly the same in all 11 of their images. v1 scaled the person to the
height of whoever they replaced, so the same face appeared at different sizes and places on
different backgrounds, and size travelled with the background. In v2 every image is one fixed
frame (600 x 800); the person keeps the size and position they had in their own photo, cropped
to that frame, and only the pixels around them change.

``jsonl`` writes the composites in the same record format and with the same prompt as
``data/probes/easyportrait_score_data.jsonl``, so ``scripts/probes/token_scoring.py`` scores them
exactly as it scored EasyPortrait (``DATASETS=face_swap`` in run_token_scoring_qwen3_gemma4.sh).

``select`` needs the token CSV and ``data/probes/easyportrait_score_data.jsonl``; ``build``
needs the 20 source photos and their masks under ``--source`` (default: the repo's
``datasets/EasyPortrait/data``), laid out as ``images/<split>/<uuid>.jpg`` and
``annotations/<split>/<uuid>.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CSV = "results/token_scoring/qwen3_vl/easyportrait/prompt_token_fg_bg_stats_combined_ideology_headwise_linear.csv"
INDEX = "data/probes/easyportrait_score_data.jsonl"
SOURCE = "datasets/EasyPortrait/data"
OUT = "data/face_swap"
N_SIDE = 5            # people (and backgrounds) per end
MAX_SIDE = 800
GROW = 9              # pixels the cut-out person is grown by before inpainting, at 800 px
FEATHER = 2           # soft edge of the pasted person, in pixels at 800 px
# photos looked at and left out, with the reason
EXCLUDE = {
    "5569ae0e-5029-48ae-8808-25961e5c46a8": "holds a latte cup, which the person mask leaves out: a hole in every composite, and a confound for the latte question",
}


# --- select ------------------------------------------------------------------
def usable(r: Dict[str, str]) -> bool:
    """A typical portrait: tall frame, the person between a third and two thirds of it,
    every image token present."""
    try:
        h, w, fg = float(r["image_h"]), float(r["image_w"]), float(r["fg_ratio"])
    except (KeyError, ValueError):
        return False
    return (r.get("image_token_mismatch") == "0" and 1.2 <= h / w <= 1.45 and 0.3 <= fg <= 0.65
            and r.get("image_fg_mean") not in ("", None) and r.get("image_bg_mean") not in ("", None))


def select(csv_path: Path, index_path: Path) -> Dict[str, Any]:
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if usable(r)]
    where = {}
    for line in open(index_path, encoding="utf-8"):
        d = json.loads(line)
        where[d["id"]] = d["image_path"]
    rows = [r for r in rows if r["record_id"] in where]

    def pick(key: str, taken: set) -> Dict[str, List[Dict[str, Any]]]:
        ranked = sorted(rows, key=lambda r: float(r[key]))
        out = {"liberal": [], "conservative": []}
        for side, seq in (("liberal", ranked), ("conservative", ranked[::-1])):
            for r in seq:
                if len(out[side]) == N_SIDE:
                    break
                if r["record_id"] in taken or Path(where[r["record_id"]]).stem in EXCLUDE:
                    continue
                taken.add(r["record_id"])
                split, name = Path(where[r["record_id"]]).parts[-2:]
                out[side].append({"record_id": r["record_id"], "uuid": Path(name).stem, "split": split,
                                  "score": float(r[key]), "fg_mean": float(r["image_fg_mean"]),
                                  "bg_mean": float(r["image_bg_mean"])})
        return out

    taken: set = set()
    people = pick("image_fg_mean", taken)
    backgrounds = pick("image_bg_mean", taken)
    return {"csv": CSV, "probe": "qwen3_vl combined_ideology_headwise_linear, image-token means",
            "pool": len(rows), "people": people, "backgrounds": backgrounds}


# --- build -------------------------------------------------------------------
def load(source: Path, item: Dict[str, Any]):
    import cv2
    img = cv2.imread(str(source / "images" / item["split"] / f"{item['uuid']}.jpg"), cv2.IMREAD_COLOR)
    ann = cv2.imread(str(source / "annotations" / item["split"] / f"{item['uuid']}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or ann is None:
        raise FileNotFoundError(f"missing photo or mask for {item['uuid']} under {source}")
    if ann.shape != img.shape[:2]:
        ann = cv2.resize(ann, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # to 800 px on the longer side, so every edit happens at the size the model sees
    s = MAX_SIDE / max(img.shape[:2])
    size = (round(img.shape[1] * s), round(img.shape[0] * s))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.resize(ann, size, interpolation=cv2.INTER_NEAREST) > 0


def bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def plate(img: np.ndarray, person: np.ndarray) -> np.ndarray:
    """The background with its own person removed and the hole filled."""
    import cv2
    hole = cv2.dilate(person.astype(np.uint8), np.ones((2 * GROW + 1, 2 * GROW + 1), np.uint8))
    # inpaint at half size, which is faster and smoother over a large hole, then paste back
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, hs, 7, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[hole > 0] = filled[hole > 0]
    return out


def composite(person_img, person_mask, bg_plate, bg_mask) -> np.ndarray:
    """The person scaled to the height of the person they replace, bottom centres aligned."""
    import cv2
    px0, py0, px1, py1 = bbox(person_mask)
    bx0, by0, bx1, by1 = bbox(bg_mask)
    s = (by1 - by0) / (py1 - py0)
    cut = person_img[py0:py1, px0:px1]
    m = person_mask[py0:py1, px0:px1].astype(np.float32)
    w, h = max(1, round(cut.shape[1] * s)), max(1, round(cut.shape[0] * s))
    cut = cv2.resize(cut, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (2 * FEATHER + 1, 2 * FEATHER + 1), 0)[..., None]
    H, W = bg_plate.shape[:2]
    x0 = round((bx0 + bx1) / 2 - w / 2)
    y0 = by1 - h
    out = bg_plate.astype(np.float32)
    # clip the pasted person to the frame
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
    if dx1 > dx0 and dy1 > dy0:
        c = cut[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0].astype(np.float32)
        a = m[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        out[dy0:dy1, dx0:dx1] = a * c + (1 - a) * out[dy0:dy1, dx0:dx1]
    return np.clip(out, 0, 255).astype(np.uint8)


def build(sel: Dict[str, Any], source: Path, out_dir: Path) -> List[Dict[str, Any]]:
    import cv2
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    people = [dict(p, side=s) for s in ("liberal", "conservative") for p in sel["people"][s]]
    bgs = [dict(b, side=s) for s in ("liberal", "conservative") for b in sel["backgrounds"][s]]
    cache = {it["uuid"]: load(source, it) for it in people + bgs}
    plates = {it["uuid"]: plate(*cache[it["uuid"]]) for it in people + bgs}
    rows = []

    def save(name: str, img: np.ndarray, **meta: Any) -> None:
        cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"item_id": name, "image_path": str((out_dir / "images" / f"{name}.jpg").relative_to(ROOT)), **meta})

    for i, p in enumerate(people):
        pi, pm = cache[p["uuid"]]
        for j, b in enumerate(bgs):
            bi, bm = cache[b["uuid"]]
            save(f"p{i:02d}_b{j:02d}", composite(pi, pm, plates[b["uuid"]], bm), kind="swap",
                 person=p["uuid"], person_side=p["side"], person_score=round(p["fg_mean"], 4),
                 background=b["uuid"], background_side=b["side"], background_score=round(b["bg_mean"], 4))
        # the check: the person put back on their own, equally edited, background
        save(f"p{i:02d}_self", composite(pi, pm, plates[p["uuid"]], pm), kind="self",
             person=p["uuid"], person_side=p["side"], person_score=round(p["fg_mean"], 4),
             background=p["uuid"], background_side="own", background_score=round(p["bg_mean"], 4))
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


FRAME = (600, 800)   # v2: width, height of every image


def cover(img: np.ndarray, mask: np.ndarray):
    """Scale to cover the v2 frame, then crop its centre, image and mask alike."""
    import cv2
    W, H = FRAME
    s = max(W / img.shape[1], H / img.shape[0])
    size = (max(W, round(img.shape[1] * s)), max(H, round(img.shape[0] * s)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    x0, y0 = (size[0] - W) // 2, (size[1] - H) // 2
    return img[y0:y0 + H, x0:x0 + W], mask[y0:y0 + H, x0:x0 + W]


def build_v2(sel: Dict[str, Any], source: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """Every person at one size and place in all their images; backgrounds filled and framed alike."""
    import cv2
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    people = [dict(p, side=s) for s in ("liberal", "conservative") for p in sel["people"][s]]
    bgs = [dict(b, side=s) for s in ("liberal", "conservative") for b in sel["backgrounds"][s]]
    framed = {}
    for it in people + bgs:
        img, mask = load(source, it)
        framed[it["uuid"]] = (img, mask, plate(img, mask))
    for u, (img, mask, pl) in list(framed.items()):
        fi, fm = cover(img, mask)
        fp, _ = cover(pl, mask)
        framed[u] = (fi, fm, fp)
    rows = []

    def paste(person_img, person_mask, bg_plate):
        a = cv2.GaussianBlur(person_mask.astype(np.float32), (2 * FEATHER + 1, 2 * FEATHER + 1), 0)[..., None]
        return np.clip(a * person_img + (1 - a) * bg_plate.astype(np.float32), 0, 255).astype(np.uint8)

    def save(name: str, img: np.ndarray, **meta: Any) -> None:
        cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"item_id": name, "image_path": str((out_dir / "images" / f"{name}.jpg").relative_to(ROOT)), **meta})

    for i, p in enumerate(people):
        pi, pm, own = framed[p["uuid"]]
        for j, b in enumerate(bgs):
            save(f"p{i:02d}_b{j:02d}", paste(pi, pm, framed[b["uuid"]][2]), kind="swap",
                 person=p["uuid"], person_side=p["side"], person_score=round(p["fg_mean"], 4),
                 background=b["uuid"], background_side=b["side"], background_score=round(b["bg_mean"], 4))
        save(f"p{i:02d}_self", paste(pi, pm, own), kind="self",
             person=p["uuid"], person_side=p["side"], person_score=round(p["fg_mean"], 4),
             background=p["uuid"], background_side="own", background_score=round(p["bg_mean"], 4))
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def build_frames(sel: Dict[str, Any], source: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """Each source photo, unedited, in the v2 frame, and its person mask (0 or 255)."""
    import cv2
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for role in ("people", "backgrounds"):
        for side, items in sel[role].items():
            for it in items:
                img, mask = cover(*load(source, it))
                cv2.imwrite(str(out_dir / "frames" / f"{it['uuid']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(out_dir / "masks" / f"{it['uuid']}.png"), mask.astype(np.uint8) * 255)
                rows.append({"uuid": it["uuid"], "role": role, "side": side})
    return rows


def portrait_prompt() -> str:
    """The prompt EasyPortrait was scored with, read from the script that wrote its JSONL."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "portrait_jsonl", ROOT / "scripts" / "probes" / "generate_portrait_score_data_jsonl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEFAULT_PROMPT


def write_jsonl(out_dir: Path, path: Path, source_name: str = "face_swap") -> int:
    prompt = portrait_prompt()
    rows = list(csv.DictReader(open(out_dir / "manifest.csv", encoding="utf-8")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            img = r["image_path"]
            rec = {"id": f"{source_name}_{r['item_id']}", "source": source_name, "name": Path(img).name,
                   "image_path": img, "prompt": prompt, "text": prompt,
                   "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                            {"type": "image", "image": img}]}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the face_swap composites")
    ap.add_argument("phase", choices=["select", "build", "jsonl", "frames"])
    ap.add_argument("--csv", default=str(ROOT / CSV))
    ap.add_argument("--index", default=str(ROOT / INDEX))
    ap.add_argument("--source", default=str(ROOT / SOURCE))
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--out", default=None, help="where the images and manifest go")
    a = ap.parse_args(argv)
    name = "face_swap" if a.version == "v1" else f"face_swap_{a.version}"
    out = Path(a.out) if a.out else ROOT / "data" / name
    if a.phase == "select":
        sel = select(Path(a.csv), Path(a.index))
        (HERE / "selection.json").write_text(json.dumps(sel, indent=1) + "\n", encoding="utf-8")
        for role in ("people", "backgrounds"):
            for side, items in sel[role].items():
                print(f"{role:<12}{side:<14}" + "  ".join(f"{it['score']:+.2f}" for it in items))
        print(f"pool {sel['pool']} portraits -> {HERE / 'selection.json'}")
        return 0
    if a.phase == "jsonl":
        path = ROOT / "data" / "probes" / f"{name}_score_data.jsonl"
        print(f"{write_jsonl(out, path, name)} records -> {path}")
        return 0
    sel = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    if a.phase == "frames":
        out = Path(a.out) if a.out else ROOT / "data" / "face_swap_v2"
        print(f"{len(build_frames(sel, Path(a.source), out))} photos -> {out / 'frames'}, masks in {out / 'masks'}")
        return 0
    rows = (build if a.version == "v1" else build_v2)(sel, Path(a.source), out)
    print(f"{len(rows)} images -> {out / 'images'}, manifest {out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
