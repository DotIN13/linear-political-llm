"""One task, one directory: an entry file, and the wording it asks in ``prompts/``.

Adding a task is adding a directory here and one line in each of the three tables in
``bench/surfaces/registry.py``. Nothing in this package holds shared machinery -- that
is ``bench/surfaces/shared/`` -- and nothing outside a task's directory holds its
wording.
"""
