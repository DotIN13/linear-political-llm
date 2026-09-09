"""bench_v2: a pilot-first rewrite of the bench harness.

Each task is a directory of versioned, self-contained experiment scripts. A
pilot owns its conditions, ordering, transcript choice and dependent-variable
reader; the shared ``helpers/`` package owns only what every pilot would
otherwise duplicate (prompt rendering, the conversation transcript, the readers
that are genuinely generic, the judge, and the run loop).

There is no surface registry, no ``GenerationSurface`` and no generic
``bench run``: a pilot *is* the entry point.
"""
