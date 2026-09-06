"""The judge has to find the answer text in either record shape.

`bench run` nests the reply under `response`; a hand-built pilot record writes
it flat as `text`. The judge knew only the nested shape, so it reported
"no generated answers to judge" on 336 real round-9 records -- which reads
exactly like "there is nothing to do".
"""

from bench.cli import _answer_text


def test_nested_shape():
    assert _answer_text({"response": {"text": "hello"}}) == "hello"


def test_flat_shape():
    assert _answer_text({"text": "hello"}) == "hello"


def test_nested_wins_when_both_are_present():
    assert _answer_text({"response": {"text": "nested"}, "text": "flat"}) == "nested"


def test_nothing_to_judge():
    for row in ({}, {"response": None}, {"response": {}}, {"text": ""},
                {"response": {"text": ""}}, {"text": None}, {"text": 42}):
        assert _answer_text(row) is None, row
