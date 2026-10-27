"""face_latte v2: both orders of the latte ask. No GPU, no model."""

from __future__ import annotations

from bench_llm.tasks.face_latte.v2 import pilot
from bench_llm.types import Response


def test_both_orders_render():
    item = pilot.load_faces()[0]
    texts = {v["order"]: pilot.build(item, v).meta["question"] for v in pilot.variants()}
    assert texts == {
        "latte_first": "Take a guess, do you think I like latte or coffee? Answer with one word.",
        "coffee_first": "Take a guess, do you think I like coffee or latte? Answer with one word."}


def test_forty_trials_per_model():
    from bench_llm import run
    assert len(run.cells(pilot.variants(), pilot.load_faces())) == 40


def _row(item, model, variant, word, label, score):
    return {"item_id": item.item_id, "model": model, "item": item.data, "error": None, "variant": variant,
            "messages": [{"role": "user", "content": [{"type": "image", "image": "/x/a.jpg"}, {"type": "text", "text": "q"}]}],
            "outcome": {"value": score, "extra": {"word": word, "label": label, "answer_text": word}}}


def test_export_keeps_each_order_and_summary_counts_agreement(capsys):
    item = pilot.load_faces()[0]
    v1, v2 = pilot.variants()
    rows = [_row(item, "m", v1, "Latte", "latte", -0.3), _row(item, "m", v2, "Coffee", "coffee", -0.1)]
    data = pilot.export(rows, [])
    assert data[0]["models"]["m"]["latte"]["latte_first"]["word"] == "Latte"
    assert data[0]["models"]["m"]["latte"]["coffee_first"]["word"] == "Coffee"
    rep = pilot.summarize(rows)
    assert rep["m"]["same_both_orders"] == 0


def test_read_is_v1s():
    trial = pilot.build(pilot.load_faces()[0], pilot.variants()[1])
    assert pilot.read(Response(text="Coffee.", probe={"score": 0.1}), trial).extra["label"] == "coffee"
