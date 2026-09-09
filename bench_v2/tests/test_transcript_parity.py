"""The transcript wording moved into a template; the messages must not change.

The build-parity test only exercises the default clause (``bare``). This one
builds every scheme x clause x photo-count through both trees, so the memory
clause and the filenames are covered too.
"""

from __future__ import annotations

import pytest

from bench.surfaces.shared import transcript as old
from bench_v2.helpers import transcript as new


@pytest.mark.parametrize("scheme", ["chat", "agentic"])
@pytest.mark.parametrize("clause", ["bare", "memory"])
@pytest.mark.parametrize("n_files", [1, 3, 5, 10])
def test_build_scheme_messages_matches_bench(scheme, clause, n_files):
    images = [f"/tmp/img/{i}.jpg" for i in range(n_files)]
    question = "What do you think I should do?"
    assert (new.build_scheme_messages(scheme, images, question, n_files, clause)
            == old.build_scheme_messages(scheme, images, question, n_files, clause))


@pytest.mark.parametrize("clause", ["bare", "memory"])
def test_wording_matches_bench(clause):
    assert new.share_line(clause) == old.share_line(clause)
    assert new.system_agentic(clause) == old.system_agentic(clause)


def test_no_image_agentic_keeps_the_filenames():
    assert (new.build_scheme_messages("agentic", [], "q", 3, "bare")
            == old.build_scheme_messages("agentic", [], "q", 3, "bare"))


def test_unknown_clause_refused():
    with pytest.raises(ValueError):
        new.share_line("loud")
