"""face_latte v1: the ask, the read and the join with face_lean. No GPU, no model."""

from __future__ import annotations

from bench_llm.tasks.face_latte.v1 import pilot
from bench_llm.tasks.face_lean.v1 import pilot as lean
from bench_llm.types import Response


def test_same_faces_as_face_lean():
    assert [it.item_id for it in pilot.load_faces()] == [it.item_id for it in lean.load_faces()]


def test_the_ask():
    trial = pilot.build(pilot.load_faces()[0], pilot.variants()[0])
    content = trial.conversation.messages[-1]["content"]
    assert [p["type"] for p in content] == ["image", "text"]
    assert content[1]["text"] == "Take a guess, do you think I like latte or coffee? Answer with one word."


def test_the_read():
    trial = pilot.build(pilot.load_faces()[0], pilot.variants()[0])
    probe = {"probe": "textual_ideology_headwise_linear", "top_k": 16, "score": -0.4, "last_prompt_token": "\n"}
    for text, label in [("Latte.", "latte"), ("coffee", "coffee"), ("Coffee!", "coffee"),
                        ("I cannot guess that.", "other")]:
        out = pilot.read(Response(text=text, probe=probe), trial)
        assert out.extra["label"] == label and out.value == -0.4


def _row(item, model, word, label, score, task):
    return {"item_id": item.item_id, "model": model, "item": item.data, "error": None,
            "messages": [{"role": "user", "content": [{"type": "image", "image": "/x/a.jpg"}, {"type": "text", "text": "q"}]}],
            "outcome": {"value": score, "extra": {"word": word, "label": label, "answer_text": word}}, "task": task}


def test_export_puts_the_latte_answer_beside_the_lean_answer():
    item = pilot.load_faces()[0]
    data = pilot.export([_row(item, "m", "Latte", "latte", -0.3, "face_latte/v1")],
                        [_row(item, "m", "liberal", "liberal", -0.5, "face_lean/v1")])
    m = data[0]["models"]["m"]
    assert m["word"] == "liberal" and m["latte"]["word"] == "Latte" and m["latte"]["score"] == -0.3
