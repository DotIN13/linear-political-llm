"""Append-only JSONL store with a content-addressed dedup key.

48k prefills will crash at least once, so ``--resume`` is the default behaviour
rather than a flag: an already-present ``trial_key`` is skipped (docs/bench/02).
Conversations are stored once by sha under ``conversations/`` and the trial row
carries only the hash, so trials.jsonl stays greppable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from bench.types import Conversation


def trial_key(
    surface: str,
    item_id: str,
    condition: str,
    adaptor: str,
    model: str,
    seed: int,
    code_rev: str,
) -> str:
    fields = [surface, item_id, condition, adaptor, model, seed, code_rev]
    names = ["surface", "item_id", "condition", "adaptor", "model", "seed", "code_rev"]
    for name, value in zip(names, fields):
        if value is None or value == "":
            raise ValueError(f"trial_key field {name!r} is empty; the key would not identify anything")
    payload = "|".join(str(v) for v in fields)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_rev(root: Optional[str] = None) -> str:
    """Short git HEAD, with a -dirty suffix when the tree has changes."""
    root = root or os.getcwd()
    try:
        rev = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:
        dirty = ""
    return f"{rev}-dirty" if dirty else rev


class ConversationStore:
    """Content-addressed store of full conversations."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def path_for(self, sha: str) -> str:
        digest = sha.split(":", 1)[-1]
        return os.path.join(self.root, digest[:2], f"{digest}.json")

    def put(self, conversation: Conversation) -> str:
        sha = conversation.sha
        path = self.path_for(sha)
        if os.path.exists(path):
            return sha
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(conversation.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return sha

    def get(self, sha: str) -> Dict[str, Any]:
        with open(self.path_for(sha), encoding="utf-8") as handle:
            return json.load(handle)

    def has(self, sha: str) -> bool:
        return os.path.exists(self.path_for(sha))


@dataclass
class RunStore:
    """One run directory: trials.jsonl + manifest.json."""

    run_dir: str
    conversations_dir: str = "conversations"

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        self.trials_path = os.path.join(self.run_dir, "trials.jsonl")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.conversations = ConversationStore(self.conversations_dir)
        self._seen = self._load_keys()
        self._handle = None

    # -- dedup ---------------------------------------------------------------
    def _load_keys(self) -> set:
        keys: set = set()
        if not os.path.exists(self.trials_path):
            return keys
        with open(self.trials_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    key = json.loads(line).get("trial_key")
                except json.JSONDecodeError:
                    continue  # torn last line from a killed job; ignore it
                if key:
                    keys.add(key)
        return keys

    def has(self, key: str) -> bool:
        return key in self._seen

    @property
    def n_done(self) -> int:
        return len(self._seen)

    # -- writing -------------------------------------------------------------
    def append(self, record: Dict[str, Any]) -> bool:
        """Write one trial. Returns False if the key was already present."""
        key = record.get("trial_key")
        if not key:
            raise ValueError("record is missing trial_key")
        if key in self._seen:
            return False
        if self._handle is None:
            self._handle = open(self.trials_path, "a", encoding="utf-8")
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._seen.add(key)
        return True

    def put_conversation(self, conversation: Conversation) -> str:
        return self.conversations.put(conversation)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading -------------------------------------------------------------
    def read(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.trials_path):
            return
        with open(self.trials_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # -- manifest ------------------------------------------------------------
    def write_manifest(
        self,
        argv: Iterable[str],
        code_rev: str,
        adaptor_config: Dict[str, Any],
        started_at: float,
        finished_at: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        manifest = {
            "run_dir": self.run_dir,
            "command": list(argv),
            "code_rev": code_rev,
            "started_at": _iso(started_at),
            "finished_at": _iso(finished_at) if finished_at else None,
            "adaptor": adaptor_config,
            "conversations_dir": self.conversations_dir,
            "n_trials": self.n_done,
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        if extra:
            manifest.update(extra)
        tmp = f"{self.manifest_path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)
        return self.manifest_path


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))


def read_items(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_items(path: str, records: Iterable[Dict[str, Any]]) -> int:
    """Items are write-once. Refuse to clobber an existing stimulus file."""
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} already exists. Items are frozen once written -- write a new name instead."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n
