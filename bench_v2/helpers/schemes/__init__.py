"""One package per scheme, with a j2 component layer and a shape escape hatch.

Three levels, in increasing order of how much they change:

* a **component** -- ``<task_dir>/<scheme>/<name>.j2`` overrides one piece of text.
  This is what a task normally does, and it needs no Python at all.
* a **shape** -- ``<task_dir>/<scheme>.py`` replaces the arm's assembly. Only when
  the global shape genuinely does not fit, because it also changes how many turns
  there are, which moves the shared prefix.
* the defaults here, which are what every task gets if it does nothing.

``build_scheme_messages`` is the default assembly and keeps its old signature, so
nothing that already imported it had to change.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench_v2.helpers import prompt_parts as parts
from bench_v2.helpers.schemes import agentic, agentic_live, chat, components

Messages = List[Dict[str, Any]]
Tools = Optional[List[Dict[str, Any]]]
Builder = Any

# Keys are scheme names, so a task's file and directory names say which scheme they
# replace.
BUILDERS: Dict[str, Builder] = {
    "chat": chat.build,
    "agentic": agentic.build,
    "agentic_live": agentic_live.build,
}


def _check(scheme: str) -> None:
    if scheme not in BUILDERS:
        raise ValueError(f"unknown scheme {scheme!r}; expected {sorted(BUILDERS)}")


def _n_files(n_files: Optional[int], image_paths: Sequence[str]) -> int:
    return n_files if n_files is not None else (len(image_paths) or 3)


def build_scheme_messages(scheme: str, image_paths: Sequence[str], question: str,
                          n_files: Optional[int] = None, variant: str = "bare",
                          portrait: Optional[str] = None,
                          portrait_name: str = parts.ME_FILE,
                          style: Optional[Dict[str, Any]] = None,
                          prompts: Optional[components.Prompts] = None
                          ) -> Tuple[Messages, Tools]:
    """One scheme's message list and its tools, or raise on an unknown scheme."""
    _check(scheme)
    p = components.resolve(prompts, style)
    return BUILDERS[scheme](image_paths, question, _n_files(n_files, image_paths),
                            variant, portrait, portrait_name, prompts=p)


@lru_cache(maxsize=64)
def _load_module(path_str: str, _mtime: float):
    """Import a task's scheme file by path, cached on (path, mtime).

    Cached because ``build_base`` asks once per trial: without this a 1,200-trial
    run would execute the same module 1,200 times. The mtime is in the key so an
    edit during a session is picked up rather than served stale.
    """
    path = Path(path_str)
    spec = importlib.util.spec_from_file_location(f"_task_scheme_{path.stem}_{abs(hash(path_str))}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_task_builder(task_dir: Path, scheme: str) -> Optional[Builder]:
    """``<task_dir>/<scheme>.py``'s ``build``, if the task shipped one."""
    path = Path(task_dir) / f"{scheme}.py"
    if not path.is_file():
        return None
    module = _load_module(str(path), path.stat().st_mtime)
    return getattr(module, "build", None) if module is not None else None


def builder_for(task_dir: str | Path) -> Builder:
    """A builder that prefers the task's own shape and components over the defaults.

    Pass it to ``build_base`` as ``messages_fn``, or let ``build_base(task_dir=...)``
    do it. The signature is ``build_scheme_messages``'s, so it is a drop-in.
    """

    def build(scheme: str, image_paths: Sequence[str], question: str,
              n_files: Optional[int] = None, variant: str = "bare",
              portrait: Optional[str] = None, portrait_name: str = parts.ME_FILE,
              style: Optional[Dict[str, Any]] = None,
              prompts: Optional[components.Prompts] = None) -> Tuple[Messages, Tools]:
        _check(scheme)
        p = prompts if prompts is not None else components.Prompts(task_dir=task_dir, style=style)
        builder = _load_task_builder(Path(task_dir), scheme) or BUILDERS[scheme]
        return builder(image_paths, question, _n_files(n_files, image_paths),
                       variant, portrait, portrait_name, prompts=p)

    return build
