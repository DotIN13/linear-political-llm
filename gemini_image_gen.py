import argparse
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw
from google import genai
from google.genai import types


def _extract_first_image(response) -> Image.Image:
    """Extract the first generated image from a Gemini response."""
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                return Image.open(io.BytesIO(inline_data.data)).convert("RGB")
    raise RuntimeError("No image found in Gemini response.")


def _to_image_part(image: Image.Image) -> types.Part:
    """Convert a PIL image to a Gemini image part for image editing calls."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def _generate_image_from_prompt(client: genai.Client, model: str, prompt: str) -> Image.Image:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    return _extract_first_image(response)


def _edit_image_with_prompt(
    client: genai.Client,
    model: str,
    source_image: Image.Image,
    prompt: str,
) -> Image.Image:
    response = client.models.generate_content(
        model=model,
        contents=[_to_image_part(source_image), prompt],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    return _extract_first_image(response)


def _compose_side_by_side(
    left: Image.Image,
    right: Image.Image,
    left_label: str = "Democrat (Blue Tie)",
    right_label: str = "Republican (Red Tie)",
) -> Image.Image:
    max_h = max(left.height, right.height)
    left_resized = left.resize((int(left.width * max_h / left.height), max_h), Image.Resampling.LANCZOS)
    right_resized = right.resize((int(right.width * max_h / right.height), max_h), Image.Resampling.LANCZOS)

    pad = 24
    label_h = 44
    out_w = left_resized.width + right_resized.width + pad * 3
    out_h = max_h + pad * 2 + label_h

    canvas = Image.new("RGB", (out_w, out_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    x1, y = pad, pad + label_h
    x2 = x1 + left_resized.width + pad

    canvas.paste(left_resized, (x1, y))
    canvas.paste(right_resized, (x2, y))

    draw.text((x1, pad), left_label, fill=(25, 25, 25))
    draw.text((x2, pad), right_label, fill=(25, 25, 25))

    return canvas


def generate_tie_pair(
    person_description: str,
    scene_description: str,
    output_dir: str,
    model: str = "gemini-3.1-flash-image-preview",
    basename: Optional[str] = None,
    base_prompt_template: Optional[str] = None,
    red_edit_prompt_template: Optional[str] = None,
) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    if base_prompt_template is None:
        base_constraints = (
            "Photorealistic editorial portrait. "
            f"Person: {person_description}. "
            f"Scene: {scene_description}. "
            "Keep expression, face identity, hair, body pose, camera angle, lighting, background, clothing, "
            "and all accessories identical across variants. "
            "The only allowed difference is tie color. "
            "No logos, no text overlays."
        )
        blue_prompt = (
            f"{base_constraints} "
            "Variant A (Democrat): subject wears a solid blue necktie."
        )
    else:
        blue_prompt = base_prompt_template.format(
            person_description=person_description,
            scene_description=scene_description,
        )

    if red_edit_prompt_template is None:
        red_edit_prompt = (
            "Create Variant B (Republican) from this exact image. "
            "Keep the same person and scene exactly unchanged. "
            "Change only the necktie color from blue to solid red. "
            "Do not alter identity, pose, facial expression, background, lighting, crop, or any other clothing."
        )
    else:
        red_edit_prompt = red_edit_prompt_template.format(
            person_description=person_description,
            scene_description=scene_description,
        )

    blue_image = _generate_image_from_prompt(client=client, model=model, prompt=blue_prompt)
    red_image = _edit_image_with_prompt(
        client=client,
        model=model,
        source_image=blue_image,
        prompt=red_edit_prompt,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = basename or f"tie_pair_{stamp}"

    blue_path = out_dir / f"{base}_democrat_blue.png"
    red_path = out_dir / f"{base}_republican_red.png"
    side_by_side_path = out_dir / f"{base}_side_by_side.png"
    meta_path = out_dir / f"{base}_meta.json"

    blue_image.save(blue_path)
    red_image.save(red_path)

    side_by_side = _compose_side_by_side(blue_image, red_image)
    side_by_side.save(side_by_side_path)

    metadata = {
        "model": model,
        "created_at": datetime.now().isoformat(),
        "person_description": person_description,
        "scene_description": scene_description,
        "blue_prompt": blue_prompt,
        "red_edit_prompt": red_edit_prompt,
        "files": {
            "democrat_blue": str(blue_path),
            "republican_red": str(red_path),
            "side_by_side": str(side_by_side_path),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired Gemini images with only tie color changed (blue vs red)."
    )
    parser.add_argument(
        "--person",
        default="a middle-aged male public official with short brown hair in a navy suit and white shirt",
        help="Person description used in the prompt.",
    )
    parser.add_argument(
        "--scene",
        default="standing at a podium in a press briefing room with blurred flags in the background",
        help="Scene description used in the prompt.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/gemini_tie_pairs",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-image-preview",
        help="Gemini image-capable model name.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Optional output base name without extension.",
    )
    parser.add_argument(
        "--base-prompt-template",
        default=None,
        help="Optional template for Variant A prompt. Supports {person_description} and {scene_description}.",
    )
    parser.add_argument(
        "--red-edit-prompt-template",
        default=None,
        help="Optional template for Variant B edit prompt. Supports {person_description} and {scene_description}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_tie_pair(
        person_description=args.person,
        scene_description=args.scene,
        output_dir=args.output_dir,
        model=args.model,
        basename=args.basename,
        base_prompt_template=args.base_prompt_template,
        red_edit_prompt_template=args.red_edit_prompt_template,
    )
    print("Saved files:")
    for k, v in result["files"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
