"""Shared helpers for pilots.

Only things every pilot would otherwise copy live here: prompt rendering, the
conversation transcript builders, the generic readers, item loading and the run
loop. The judge machinery is its own root-level package (``bench_v2.judge``), and
each task's judge *spec* lives with the task. Task-specific logic -- conditions,
ordering, outlet matching, the dependent variable -- belongs to the pilot.
"""
