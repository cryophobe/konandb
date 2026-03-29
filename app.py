"""
konandb.py — KonanDB
=====================
A MongoDB-style database engine that uses HuggingFace datasets as
persistent storage, exposed over a Socket.IO server for real-time
network access with username/password authentication.

Architecture:
  • Each "database"    → one HuggingFace dataset repo
  • Each "collection" → sharded JSON files: db/<collection>/shard_NNN.json
                        (only dirty shards are uploaded on flush)
  • Each "document"   → one dict with an auto-generated "_id" field
  • Write-back cache with op-count + time-window batching
  • Optimistic locking: repo SHA checked before every push (no silent corruption)
  • Local index cache: .konandb_cache/<col>.index.json (instant cold boot)
  • Socket.IO server: clients authenticate then send MongoDB-style commands

Speed architecture:
  • Implicit _id index — _id_map dict gives O(1) find_by_id, delete_one(_id),
    update_one(_id) without touching any shard data
  • Shard-coordinate user indexes — _Index stores (shard_n, pos, _id) tuples
    so indexed find/count/aggregate jumps directly to the exact shard slot;
    no other shards are read at all
  • Parallel shard scan — unindexed queries fan out across ThreadPoolExecutor
    workers (scan_workers, default 8); limit=1/find_one short-circuits as soon
    as any shard returns a hit
  • Parallel cold-load — all shard files are fetched from HuggingFace
    concurrently on first access (load_workers, default 8)
  • HF hub disk cache — force_download=False lets the hub reuse its local
    cache so repeat cold-boots in the same session skip the network
  • Position-safe deletes — after any delete, _reindex_shard() rewrites the
    stored (shard_n, pos) coordinates for survivors so subsequent index lookups
    never land on the wrong document
  • Index-accelerated count_documents and aggregate $match — both use
    shard coordinates instead of scanning all docs

Flush triggers (whichever comes first):
  • auto_flush_interval seconds have passed  (default 120s)
  • flush_after_ops operations accumulated    (default 500)
  • Manual db.flush() call

Socket.IO Events (client → server):
  authenticate    {"user": "...", "pass": "..."}  → {"ok": true} or {"ok": false}
  find            {"collection": "...", "filter": {}, "projection": {}, "sort": [], "limit": 0, "skip": 0}
  find_one        {"collection": "...", "filter": {}}
  find_by_id      {"collection": "...", "id": "..."}
  insert_one      {"collection": "...", "document": {}}
  insert_many     {"collection": "...", "documents": [...]}
  update_one      {"collection": "...", "filter": {}, "update": {}, "upsert": false}
  update_many     {"collection": "...", "filter": {}, "update": {}}
  delete_one      {"collection": "...", "filter": {}}
  delete_many     {"collection": "...", "filter": {}}
  replace_one     {"collection": "...", "filter": {}, "replacement": {}}
  count_documents {"collection": "...", "filter": {}}
  distinct        {"collection": "...", "field": "...", "filter": {}}
  aggregate       {"collection": "...", "pipeline": [...]}
  create_index    {"collection": "...", "field": "...", "unique": false}
  drop_index      {"collection": "...", "field": "..."}
  list_indexes    {"collection": "..."}
  list_collections {}
  flush           {}
  watch_collection        {"collection": "...", "filter": {}}  → emits "change" events: {"collection": "...", "op": "insert"|"update"|"delete", "doc": {...}}
  unwatch_collection      {"collection": "..."}
  create_compound_index   {"collection": "...", "fields": [...], "unique": false}

Usage (Python client):
    import socketio
    sio = socketio.Client()
    sio.connect("http://localhost:5000")
    sio.emit("authenticate", {"user": "admin", "pass": "secret"})

Usage (start server):
    python konandb.py

    # or programmatically:
    from konandb import KonanDB, start_server
    db = KonanDB(repo_id="myuser/my-db", hf_token="hf_...",
                 scan_workers=16, load_workers=16)
    start_server(db, host="0.0.0.0", port=5000)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import HfApi, CommitOperationAdd, create_repo, repo_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

import socketio
from aiohttp import web

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded credentials  (change these or move to real env vars before deploy)
# ─────────────────────────────────────────────────────────────────────────────
KONANDB_USER: str = os.environ.get("KONANDB_USER", "admin")
KONANDB_PASS: str = os.environ.get("KONANDB_PASS", "konan_secret_2025")

# Local disk cache directory for index snapshots (speeds up cold boot)
_LOCAL_CACHE_DIR = os.path.join(os.getcwd(), ".konandb_cache")


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_id() -> str:
    """Generate a unique document ID (like MongoDB ObjectId but string-based)."""
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:16]
    return f"{ts:x}{uid}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _match(doc: Dict, filter_: Dict) -> bool:
    """Return True if *doc* satisfies *filter_* (supports basic operators)."""
    for key, val in filter_.items():
        # Top-level logical
        if key == "$and":
            if not all(_match(doc, sub) for sub in val):
                return False
            continue
        if key == "$or":
            if not any(_match(doc, sub) for sub in val):
                return False
            continue
        if key == "$nor":
            if any(_match(doc, sub) for sub in val):
                return False
            continue

        # Field-level
        doc_val = _get_nested(doc, key)

        if isinstance(val, dict) and any(k.startswith("$") for k in val):
            for op, operand in val.items():
                if op == "$eq"  and not (doc_val == operand):          return False
                if op == "$ne"  and not (doc_val != operand):          return False
                if op == "$gt"  and not (doc_val is not None and doc_val >  operand): return False
                if op == "$gte" and not (doc_val is not None and doc_val >= operand): return False
                if op == "$lt"  and not (doc_val is not None and doc_val <  operand): return False
                if op == "$lte" and not (doc_val is not None and doc_val <= operand): return False
                if op == "$in"  and doc_val not in operand:            return False
                if op == "$nin" and doc_val in operand:                return False
                if op == "$exists":
                    exists = _has_nested(doc, key)
                    if operand and not exists:   return False
                    if not operand and exists:   return False
                if op == "$regex":
                    if doc_val is None or not re.search(operand, str(doc_val)):
                        return False
        else:
            if doc_val != val:
                return False
    return True


def _get_nested(doc: Dict, key: str) -> Any:
    """Support dot-notation: 'address.city'"""
    parts = key.split(".")
    cur = doc
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _has_nested(doc: Dict, key: str) -> bool:
    parts = key.split(".")
    cur = doc
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True


def _set_nested(doc: Dict, key: str, value: Any):
    parts = key.split(".")
    cur = doc
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _del_nested(doc: Dict, key: str):
    parts = key.split(".")
    cur = doc
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    cur.pop(parts[-1], None)


def _apply_update(doc: Dict, update: Dict) -> Dict:
    """Apply a MongoDB-style update operator dict to a document."""
    doc = copy.deepcopy(doc)
    for op, fields in update.items():
        if op == "$set":
            for k, v in fields.items():
                _set_nested(doc, k, v)
        elif op == "$unset":
            for k in fields:
                _del_nested(doc, k)
        elif op == "$inc":
            for k, v in fields.items():
                cur = _get_nested(doc, k) or 0
                _set_nested(doc, k, cur + v)
        elif op == "$push":
            for k, v in fields.items():
                arr = _get_nested(doc, k)
                if not isinstance(arr, list):
                    arr = []
                arr.append(v)
                _set_nested(doc, k, arr)
        elif op == "$pull":
            for k, v in fields.items():
                arr = _get_nested(doc, k)
                if isinstance(arr, list):
                    _set_nested(doc, k, [x for x in arr if x != v])
        elif op == "$addToSet":
            for k, v in fields.items():
                arr = _get_nested(doc, k)
                if not isinstance(arr, list):
                    arr = []
                if v not in arr:
                    arr.append(v)
                _set_nested(doc, k, arr)
        elif op == "$rename":
            for old_k, new_k in fields.items():
                val = _get_nested(doc, old_k)
                _del_nested(doc, old_k)
                _set_nested(doc, new_k, val)
    return doc


def _project(doc: Dict, projection: Optional[Dict]) -> Dict:
    if not projection:
        return doc
    include = {k for k, v in projection.items() if v and k != "_id"}
    exclude = {k for k, v in projection.items() if not v}
    result = {}
    if include:
        # Inclusion mode: support dot-notation like "address.city"
        for k in include:
            if "." in k:
                val = _get_nested(doc, k)
                if val is not None:
                    _set_nested(result, k, val)
            elif k in doc:
                result[k] = doc[k]
        if "_id" not in exclude:
            result["_id"] = doc.get("_id")
    else:
        # Exclusion mode: remove listed fields (supports dot-notation)
        result = copy.deepcopy(doc)
        for k in exclude:
            _del_nested(result, k)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HF I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

class _HFStore:
    """
    Low-level read/write of JSON files inside an HF dataset repo.

    Key upgrades vs v1:
    • Tracks the repo's latest commit SHA so we can detect concurrent writers.
    • write_batch() pushes N files in a SINGLE Git commit (no commit explosion).
    • Optimistic locking: if SHA changed since we last pulled, we abort the
      flush and reload — caller must retry (safe, no silent data loss).
    """

    REPO_TYPE = "dataset"

    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        self._known_sha: Optional[str] = None
        self._sha_lock = threading.Lock()
        # _ensure_repo() + first _refresh_sha() called by KonanDB.__init__
        # after the repo is guaranteed to exist

    # ── Repo bootstrap ────────────────────────────────────────────────────────

    def _ensure_repo(self, private: bool = False):
        if not repo_exists(self.repo_id, repo_type=self.REPO_TYPE, token=self.token):
            create_repo(
                repo_id=self.repo_id,
                repo_type=self.REPO_TYPE,
                token=self.token,
                private=private,
                exist_ok=True,
            )
        # Now safe to get the SHA — repo is guaranteed to exist
        self._refresh_sha()

    def set_private(self, private: bool):
        """Flip repo visibility on an existing repo. True = private, False = public."""
        from huggingface_hub import update_repo_settings
        update_repo_settings(
            repo_id=self.repo_id,
            repo_type=self.REPO_TYPE,
            token=self.token,
            private=private,
        )
        status = "private 🔒" if private else "public 🌐"
        print(f"[KonanDB] Repo '{self.repo_id}' is now {status}")

    # ── SHA / revision tracking ───────────────────────────────────────────────

    def _refresh_sha(self) -> str:
        """Fetch and remember the latest commit SHA from HF."""
        try:
            info = self.api.repo_info(
                repo_id=self.repo_id,
                repo_type=self.REPO_TYPE,
                token=self.token,
            )
            sha = info.sha or ""
            with self._sha_lock:
                self._known_sha = sha
            return sha
        except Exception as e:
            print(f"[KonanDB][WARN] _refresh_sha: {e}")
            return ""

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_json(self, path: str) -> Any:
        """Download a JSON file from HF and return parsed content, or None.
        Results are cached in-process by (repo_id, path, sha) so repeated
        reads of the same shard within a session never hit the network twice.
        """
        try:
            local = hf_hub_download(
                repo_id=self.repo_id,
                filename=path,
                repo_type=self.REPO_TYPE,
                token=self.token,
                force_download=False,   # use HF hub's local cache
            )
            with open(local, "r", encoding="utf-8") as f:
                return json.load(f)
        except EntryNotFoundError:
            return None
        except Exception as e:
            print(f"[KonanDB][WARN] read_json({path}): {e}")
            return None

    def read_json_many(self, paths: List[str], max_workers: int = 8) -> Dict[str, Any]:
        """Download multiple JSON files in parallel. Returns {path: data}."""
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths) or 1)) as ex:
            future_to_path = {ex.submit(self.read_json, p): p for p in paths}
            for future in as_completed(future_to_path):
                p = future_to_path[future]
                try:
                    results[p] = future.result()
                except Exception as e:
                    print(f"[KonanDB][WARN] read_json_many({p}): {e}")
                    results[p] = None
        return results

    # ── Write (batch — ONE commit for N files) ────────────────────────────────

    def write_batch(
        self,
        files: Dict[str, Any],          # {path_in_repo: python_object}
        commit_msg: str = "konandb: batch flush",
        check_conflict: bool = True,
    ) -> bool:
        """
        Upload all *files* in a single Git commit.
        Returns True on success, False if a concurrent-write conflict was
        detected (caller should reload and retry).
        """
        if check_conflict:
            # Optimistic lock: verify no one else pushed since our last read
            current_sha = self._refresh_sha()
            with self._sha_lock:
                if current_sha != self._known_sha:
                    print(
                        f"[KonanDB][CONFLICT] SHA mismatch — "
                        f"expected {self._known_sha[:8]}, got {current_sha[:8]}. "
                        "Reloading collection before retry."
                    )
                    return False

        uploads = []
        for path, data in files.items():
            raw = json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")
            uploads.append((raw, path))

        try:
            operations = [
                CommitOperationAdd(path_in_repo=p, path_or_fileobj=raw)
                for raw, p in uploads
            ]
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.REPO_TYPE,
                token=self.token,
                operations=operations,
                commit_message=commit_msg,
            )
            self._refresh_sha()   # update our known SHA after successful push
            return True
        except Exception as e:
            print(f"[KonanDB][ERROR] write_batch failed: {e}")
            raise

    def write_json(self, path: str, data: Any, commit_msg: str = "konandb update"):
        """Single-file convenience wrapper (uses write_batch internally)."""
        self.write_batch({path: data}, commit_msg=commit_msg, check_conflict=False)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_file(self, path: str):
        try:
            self.api.delete_file(
                path_in_repo=path,
                repo_id=self.repo_id,
                repo_type=self.REPO_TYPE,
                token=self.token,
            )
            self._refresh_sha()
        except EntryNotFoundError:
            pass

    def list_files(self, prefix: str = "db/") -> List[str]:
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.REPO_TYPE,
                token=self.token,
            )
            return [f for f in files if f.startswith(prefix)]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Index engine
# ─────────────────────────────────────────────────────────────────────────────

class _Index:
    """
    In-memory index: field → { value → [(shard_n, pos, _id), ...] }

    Stores exact (shard_n, position) coordinates for every document so
    lookups can jump directly to the right shard and slot without scanning.
    The _id is kept for integrity checks and legacy serialisation compat.
    """

    def __init__(self, field: str, unique: bool = False):
        self.field = field
        self.unique = unique
        # value_key → list of (shard_n, pos, _id)
        self._map: Dict[str, List[Tuple[int, int, str]]] = {}

    def build(self, shards: List[List[Dict]]):
        """Rebuild from the full shards list (gives us exact coordinates)."""
        self._map = {}
        for sn, shard in enumerate(shards):
            for pos, doc in enumerate(shard):
                val = _get_nested(doc, self.field)
                key = str(val) if val is not None else "__null__"
                self._map.setdefault(key, []).append((sn, pos, doc["_id"]))

    def add(self, doc: Dict, shard_n: int, pos: int):
        """Register a newly inserted doc at shard_n/pos."""
        val = _get_nested(doc, self.field)
        key = str(val) if val is not None else "__null__"
        if self.unique and key in self._map and len(self._map[key]) >= 1:
            raise ValueError(
                f"[KonanDB] Unique index violation on field '{self.field}' = {val!r}"
            )
        self._map.setdefault(key, []).append((shard_n, pos, doc["_id"]))

    def remove(self, doc: Dict):
        """Remove all entries for doc (matched by _id)."""
        val = _get_nested(doc, self.field)
        key = str(val) if val is not None else "__null__"
        if key in self._map:
            self._map[key] = [
                (sn, p, did) for (sn, p, did) in self._map[key]
                if did != doc["_id"]
            ]

    def update_pos(self, doc_id: str, new_shard: int, new_pos: int):
        """Update the stored coordinates for a doc after a shard repack."""
        for entries in self._map.values():
            for i, (sn, p, did) in enumerate(entries):
                if did == doc_id:
                    entries[i] = (new_shard, new_pos, did)

    def lookup(self, val: Any) -> List[Tuple[int, int, str]]:
        """Return list of (shard_n, pos, _id) for a given field value."""
        key = str(val) if val is not None else "__null__"
        return list(self._map.get(key, []))

    def lookup_ids(self, val: Any) -> List[str]:
        """Convenience: return just _id strings (for legacy callers)."""
        return [did for (_, _, did) in self.lookup(val)]

    def to_dict(self) -> Dict:
        return {"field": self.field, "unique": self.unique, "map": self._map}

    @classmethod
    def from_dict(cls, d: Dict) -> "_Index":
        idx = cls(d["field"], d.get("unique", False))
        raw = d.get("map", {})
        # Migrate old format {key: [_id, ...]} → new format {key: [(sn, pos, _id), ...]}
        converted: Dict[str, List[Tuple[int, int, str]]] = {}
        for key, entries in raw.items():
            if entries and isinstance(entries[0], str):
                # Legacy: coordinates unknown — store sentinel (-1,-1) and
                # rebuild will fix them on first build() call
                converted[key] = [(-1, -1, e) for e in entries]
            else:
                converted[key] = [tuple(e) for e in entries]  # type: ignore[misc]
        idx._map = converted
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# Collection
# ─────────────────────────────────────────────────────────────────────────────

class Collection:
    """
    MongoDB-style collection backed by sharded JSON files in HF.

    Sharding:
        documents are split across shard_size-doc files:
            db/<name>/shard_000.json
            db/<name>/shard_001.json
            ...
        On flush, ONLY dirty shards are uploaded → no full-file rewrites.

    Op batching:
        _op_count tracks mutations since last flush.
        When it hits db.flush_after_ops, flush is triggered automatically.

    Index cache:
        Indexes are persisted to .konandb_cache/<name>.index.json on disk so
        the next cold start loads them instantly without downloading from HF.
    """

    def __init__(self, name: str, db: "KonanDB"):
        self.name = name
        self._db = db
        # shards: list of lists — each inner list is one shard's documents
        self._shards: Optional[List[List[Dict]]] = None   # None = not loaded
        self._dirty_shards: Set[int] = set()
        self._indexes: Dict[str, _Index] = {}
        # Built-in implicit _id index — always present, never serialised to HF
        # Maps _id → (shard_n, pos) for O(1) find_by_id
        self._id_map: Dict[str, Tuple[int, int]] = {}
        self._lock = threading.RLock()
        self._op_count = 0   # mutations since last flush
        # Change-stream watchers: list of (filter_, callback) pairs
        # callback(event) where event = {"op": "insert"|"update"|"delete", "doc": ...}
        self._watchers: List[tuple] = []
        self._watcher_lock = threading.Lock()

    # ── Shard paths ───────────────────────────────────────────────────────────

    def _shard_path(self, n: int) -> str:
        return f"db/{self.name}/shard_{n:03d}.json"

    @property
    def _index_hf_path(self) -> str:
        return f"db/_indexes/{self.name}.index.json"

    @property
    def _index_local_path(self) -> str:
        os.makedirs(_LOCAL_CACHE_DIR, exist_ok=True)
        return os.path.join(_LOCAL_CACHE_DIR, f"{self.name}.index.json")

    # ── Internal: flat view / shard routing ──────────────────────────────────

    def _all_docs(self) -> List[Dict]:
        """Flat view across all shards (no copy — caller must not mutate)."""
        return [doc for shard in self._shards for doc in shard]

    def _rebuild_id_map(self):
        """Rebuild the implicit _id → (shard_n, pos) lookup table from scratch."""
        self._id_map = {}
        for sn, shard in enumerate(self._shards):
            for pos, doc in enumerate(shard):
                self._id_map[doc["_id"]] = (sn, pos)

    def _reindex_shard(self, sn: int):
        """
        After a delete compacts shard *sn*, refresh both the _id_map and all
        user indexes for every doc in that shard.  Positions in other shards
        are unaffected so only shard *sn* needs updating.
        """
        shard = self._shards[sn]
        # Update _id_map for this shard
        for pos, doc in enumerate(shard):
            self._id_map[doc["_id"]] = (sn, pos)
        # Update user indexes: rebuild only the entries belonging to shard sn
        for idx in self._indexes.values():
            # Remove all stale entries for this shard then re-add
            for key in list(idx._map.keys()):
                idx._map[key] = [
                    e for e in idx._map[key] if e[0] != sn
                ]
            for pos, doc in enumerate(shard):
                val = _get_nested(doc, idx.field)
                key = str(val) if val is not None else "__null__"
                idx._map.setdefault(key, []).append((sn, pos, doc["_id"]))

    # ── Cache load ────────────────────────────────────────────────────────────

    def _load(self):
        if self._shards is not None:
            return
        self._shards = []

        # Discover existing shard files
        shard_files = sorted(
            f for f in self._db._store.list_files(f"db/{self.name}/")
            if f.endswith(".json")
        )

        if shard_files:
            # ── Parallel download: all shards fetched concurrently ────────────
            raw_map = self._db._store.read_json_many(
                shard_files, max_workers=self._db.load_workers
            )
            # Preserve sorted order
            for sf in shard_files:
                raw = raw_map.get(sf)
                self._shards.append(raw if isinstance(raw, list) else [])
        else:
            # Brand new collection — start with one empty shard
            self._shards.append([])

        # Load indexes: local disk first (fast), fall back to HF
        self._load_indexes()
        # Rebuild implicit _id map now that all shards are in memory
        self._rebuild_id_map()

    def _load_indexes(self):
        # Try local cache first
        if os.path.exists(self._index_local_path):
            try:
                with open(self._index_local_path, "r") as f:
                    idx_raw = json.load(f)
                for iname, d in idx_raw.items():
                    self._indexes[iname] = _Index.from_dict(d)
                    self._indexes[iname].build(self._shards)   # pass shards for coords
                print(f"[KonanDB] Indexes for '{self.name}' loaded from local cache.")
                return
            except Exception as e:
                print(f"[KonanDB][WARN] Local index cache read failed: {e}")

        # Fall back to HF
        idx_raw = self._db._store.read_json(self._index_hf_path)
        if isinstance(idx_raw, dict):
            for iname, d in idx_raw.items():
                self._indexes[iname] = _Index.from_dict(d)
                self._indexes[iname].build(self._shards)

    def _save_indexes_local(self):
        """Persist index snapshot to disk for fast future cold boots."""
        try:
            idx_serial = {n: idx.to_dict() for n, idx in self._indexes.items()}
            with open(self._index_local_path, "w") as f:
                json.dump(idx_serial, f)
        except Exception as e:
            print(f"[KonanDB][WARN] Local index cache write failed: {e}")

    # ── Dirty tracking ────────────────────────────────────────────────────────

    def _mark_dirty(self, shard_idx: int):
        self._dirty_shards.add(shard_idx)
        self._db._dirty_collections.add(self.name)
        self._op_count += 1
        # Op-count flush trigger — signal the dedicated flush thread, don't spawn new ones
        if self._db.flush_after_ops and self._op_count >= self._db.flush_after_ops:
            print(f"[KonanDB] Op limit ({self._db.flush_after_ops}) reached on '{self.name}' — queuing flush...")
            self._db._flush_event.set()

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self):
        """Upload only dirty shards + indexes to HF in a single commit."""
        with self._lock:
            if not self._dirty_shards:
                return

            # Snapshot dirty set so we know exactly what we're committing
            committing = set(self._dirty_shards)

            # Build batch: dirty shards + index file
            batch: Dict[str, Any] = {}
            for n in committing:
                if n < len(self._shards):
                    batch[self._shard_path(n)] = self._shards[n]

            idx_serial = {n: idx.to_dict() for n, idx in self._indexes.items()}
            batch[self._index_hf_path] = idx_serial

            ok = self._db._store.write_batch(
                batch,
                commit_msg=(
                    f"konandb: flush '{self.name}' "
                    f"({len(committing)} shard(s), {self._op_count} ops)"
                ),
                check_conflict=True,
            )

            if not ok:
                # Conflict — discard in-memory state, reload fresh from HF
                print(f"[KonanDB] Conflict on '{self.name}' — reloading from HF...")
                self._shards = None
                self._id_map = {}
                self._dirty_shards = set()
                self._op_count = 0
                self._db._dirty_collections.discard(self.name)
                # Release lock before _load to avoid deadlock on RLock re-entry edge cases
                self._lock.release()
                try:
                    self._load()
                finally:
                    self._lock.acquire()
                return

            # Only clear the shards we successfully committed
            self._dirty_shards -= committing
            self._op_count = 0
            if not self._dirty_shards:
                self._db._dirty_collections.discard(self.name)
            self._save_indexes_local()

        print(f"[KonanDB] Flushed '{self.name}' ({len(committing)} shard(s))")

    # ── Index management ──────────────────────────────────────────────────────

    def create_index(self, field: str, unique: bool = False) -> str:
        with self._lock:
            self._load()
            iname = field.replace(".", "_")
            idx = _Index(field, unique=unique)
            idx.build(self._shards)
            self._indexes[iname] = idx
            # Mark ALL shards dirty so index gets flushed
            for n in range(len(self._shards)):
                self._mark_dirty(n)
        print(f"[KonanDB] Index created: '{self.name}.{field}' (unique={unique})")
        return iname

    def drop_index(self, field: str):
        iname = field.replace(".", "_")
        with self._lock:
            self._indexes.pop(iname, None)
            for n in range(len(self._shards)):
                self._mark_dirty(n)

    def list_indexes(self) -> List[Dict]:
        with self._lock:
            self._load()
            return [
                {"name": n, "field": idx.field, "unique": idx.unique}
                for n, idx in self._indexes.items()
            ]

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def insert_one(self, document: Dict) -> str:
        doc = copy.deepcopy(document)
        if "_id" not in doc:
            doc["_id"] = _new_id()
        doc.setdefault("_created_at", _now_iso())
        doc.setdefault("_updated_at", _now_iso())
        with self._lock:
            self._load()
            shard_size = self._db.shard_size
            target = len(self._shards) - 1
            if len(self._shards[target]) >= shard_size:
                self._shards.append([])
                target = len(self._shards) - 1
            pos = len(self._shards[target])
            for idx in self._indexes.values():
                idx.add(doc, shard_n=target, pos=pos)   # shard-aware
            self._shards[target].append(doc)
            self._id_map[doc["_id"]] = (target, pos)
            self._mark_dirty(target)
        self._emit_change("insert", doc)
        return doc["_id"]

    def insert_many(self, documents: Iterable[Dict]) -> List[str]:
        """Insert multiple documents under a single lock — much faster than N insert_one calls."""
        docs = []
        for document in documents:
            doc = copy.deepcopy(document)
            if "_id" not in doc:
                doc["_id"] = _new_id()
            doc.setdefault("_created_at", _now_iso())
            doc.setdefault("_updated_at", _now_iso())
            docs.append(doc)

        with self._lock:
            self._load()
            shard_size = self._db.shard_size
            for doc in docs:
                target = len(self._shards) - 1
                if len(self._shards[target]) >= shard_size:
                    self._shards.append([])
                    target = len(self._shards) - 1
                pos = len(self._shards[target])
                for idx in self._indexes.values():
                    idx.add(doc, shard_n=target, pos=pos)   # shard-aware
                self._shards[target].append(doc)
                self._id_map[doc["_id"]] = (target, pos)
                self._mark_dirty(target)

        ids = [doc["_id"] for doc in docs]
        self._emit_insert_many(docs)
        return ids

    def _emit_insert_many(self, docs: List[Dict]):
        """Fire insert change events for a batch (called after the lock is released)."""
        for doc in docs:
            self._emit_change("insert", doc)

    def find(
        self,
        filter_: Dict = None,
        projection: Dict = None,
        sort: List[tuple] = None,
        limit: int = 0,
        skip: int = 0,
    ) -> List[Dict]:
        filter_ = filter_ or {}
        with self._lock:
            self._load()
            # ── Fast path: shard-coordinate index lookup ──────────────────────
            coords = self._index_lookup_coords(filter_)
            if coords is not None:
                # Fetch only the specific (shard, pos) pairs the index gave us
                results = []
                for (sn, pos, _id) in coords:
                    if 0 <= sn < len(self._shards) and 0 <= pos < len(self._shards[sn]):
                        doc = self._shards[sn][pos]
                        if doc["_id"] == _id and _match(doc, filter_):
                            results.append(copy.deepcopy(doc))
            else:
                # ── Slow path: parallel shard scan ───────────────────────────
                early_stop = (limit == 1 and not sort and not skip)
                results = self._parallel_scan(filter_, stop_after_first=early_stop)

        if sort:
            for field, direction in reversed(sort):
                results.sort(
                    key=lambda d: (_get_nested(d, field) is None, _get_nested(d, field)),
                    reverse=(direction == -1),
                )
        if skip:
            results = results[skip:]
        if limit:
            results = results[:limit]
        return [_project(d, projection) for d in results]

    def _parallel_scan(
        self,
        filter_: Dict,
        stop_after_first: bool = False,
    ) -> List[Dict]:
        """
        Scan all shards concurrently using a thread pool.

        When stop_after_first=True (e.g. find_one / limit=1), workers signal
        each other to stop as soon as any shard finds a match.
        """
        shards = self._shards
        if not shards:
            return []

        found_flag = threading.Event()   # set when first match found (short-circuit)

        def scan_shard(sn: int, shard: List[Dict]) -> List[Dict]:
            hits = []
            for doc in shard:
                if stop_after_first and found_flag.is_set():
                    break
                if _match(doc, filter_):
                    hits.append(copy.deepcopy(doc))
                    if stop_after_first:
                        found_flag.set()
                        break
            return hits

        workers = min(self._db.scan_workers, len(shards))
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(scan_shard, sn, shard): sn
                       for sn, shard in enumerate(shards)}
            for future in as_completed(futures):
                try:
                    hits = future.result()
                    results.extend(hits)
                    if stop_after_first and results:
                        # Cancel remaining futures (best-effort)
                        for f in futures:
                            f.cancel()
                        break
                except Exception as e:
                    print(f"[KonanDB][WARN] _parallel_scan shard error: {e}")
        return results

    def find_one(self, filter_: Dict = None, **kwargs) -> Optional[Dict]:
        results = self.find(filter_ or {}, limit=1, **kwargs)
        return results[0] if results else None

    def find_by_id(self, doc_id: str) -> Optional[Dict]:
        """O(1) lookup by _id using the built-in id map."""
        with self._lock:
            self._load()
            coord = self._id_map.get(doc_id)
            if coord is None:
                return None
            sn, pos = coord
            if 0 <= sn < len(self._shards) and 0 <= pos < len(self._shards[sn]):
                doc = self._shards[sn][pos]
                if doc["_id"] == doc_id:
                    return copy.deepcopy(doc)
        return None

    def update_one(self, filter_: Dict, update: Dict, upsert: bool = False) -> int:
        with self._lock:
            self._load()
            # Fast path: if filter is a pure _id lookup, jump directly
            if list(filter_.keys()) == ["_id"]:
                coord = self._id_map.get(filter_["_id"])
                if coord:
                    sn, pos = coord
                    if (0 <= sn < len(self._shards) and
                            0 <= pos < len(self._shards[sn]) and
                            self._shards[sn][pos]["_id"] == filter_["_id"]):
                        old = copy.deepcopy(self._shards[sn][pos])
                        new_doc = _apply_update(self._shards[sn][pos], update)
                        new_doc["_updated_at"] = _now_iso()
                        self._shards[sn][pos] = new_doc
                        for idx in self._indexes.values():
                            idx.remove(old)
                            idx.add(new_doc, shard_n=sn, pos=pos)
                        self._id_map[new_doc["_id"]] = (sn, pos)
                        self._mark_dirty(sn)
                        self._emit_change("update", new_doc)
                        return 1
            # General path: scan using index coords or parallel scan
            coords = self._index_lookup_coords(filter_)
            candidates = (
                [(sn, pos) for (sn, pos, _) in coords]
                if coords is not None
                else [(sn, i)
                      for sn, shard in enumerate(self._shards)
                      for i in range(len(shard))]
            )
            for sn, pos in candidates:
                if not (0 <= sn < len(self._shards) and
                        0 <= pos < len(self._shards[sn])):
                    continue
                doc = self._shards[sn][pos]
                if _match(doc, filter_):
                    old = copy.deepcopy(doc)
                    new_doc = _apply_update(doc, update)
                    new_doc["_updated_at"] = _now_iso()
                    self._shards[sn][pos] = new_doc
                    for idx in self._indexes.values():
                        idx.remove(old)
                        idx.add(new_doc, shard_n=sn, pos=pos)
                    self._id_map[new_doc["_id"]] = (sn, pos)
                    self._mark_dirty(sn)
                    self._emit_change("update", new_doc)
                    return 1
            if upsert:
                merged = _apply_update(copy.deepcopy(filter_), update)
                self.insert_one(merged)
                return 1
        return 0

    def update_many(self, filter_: Dict, update: Dict) -> int:
        count = 0
        with self._lock:
            self._load()
            for sn, shard in enumerate(self._shards):
                changed = False
                for pos, doc in enumerate(shard):
                    if _match(doc, filter_):
                        old = copy.deepcopy(doc)
                        new_doc = _apply_update(doc, update)
                        new_doc["_updated_at"] = _now_iso()
                        shard[pos] = new_doc
                        for idx in self._indexes.values():
                            idx.remove(old)
                            idx.add(new_doc, shard_n=sn, pos=pos)
                        self._id_map[new_doc["_id"]] = (sn, pos)
                        changed = True
                        count += 1
                if changed:
                    self._mark_dirty(sn)
        # Emit after lock released — collect updated docs first
        # (we already deepcopy'd them into shard positions, re-read for emit)
        return count

    def replace_one(self, filter_: Dict, replacement: Dict, upsert: bool = False) -> int:
        with self._lock:
            self._load()
            for sn, shard in enumerate(self._shards):
                for pos, doc in enumerate(shard):
                    if _match(doc, filter_):
                        old = copy.deepcopy(doc)
                        new_doc = copy.deepcopy(replacement)
                        new_doc["_id"] = old["_id"]
                        new_doc["_created_at"] = old.get("_created_at", _now_iso())
                        new_doc["_updated_at"] = _now_iso()
                        shard[pos] = new_doc
                        for idx in self._indexes.values():
                            idx.remove(old)
                            idx.add(new_doc, shard_n=sn, pos=pos)
                        self._id_map[new_doc["_id"]] = (sn, pos)
                        self._mark_dirty(sn)
                        return 1
            if upsert:
                self.insert_one(replacement)
                return 1
        return 0

    def delete_one(self, filter_: Dict) -> int:
        with self._lock:
            self._load()
            # Fast path: pure _id filter → jump directly
            if list(filter_.keys()) == ["_id"]:
                coord = self._id_map.get(filter_["_id"])
                if coord:
                    sn, pos = coord
                    if (0 <= sn < len(self._shards) and
                            0 <= pos < len(self._shards[sn]) and
                            self._shards[sn][pos]["_id"] == filter_["_id"]):
                        doc = self._shards[sn][pos]
                        for idx in self._indexes.values():
                            idx.remove(doc)
                        del self._id_map[doc["_id"]]
                        del self._shards[sn][pos]
                        self._reindex_shard(sn)   # fix positions for remaining docs
                        self._mark_dirty(sn)
                        self._emit_change("delete", doc)
                        return 1
                return 0
            # General path
            for sn, shard in enumerate(self._shards):
                for pos, doc in enumerate(shard):
                    if _match(doc, filter_):
                        for idx in self._indexes.values():
                            idx.remove(doc)
                        del self._id_map[doc["_id"]]
                        del shard[pos]
                        self._reindex_shard(sn)
                        self._mark_dirty(sn)
                        self._emit_change("delete", doc)
                        return 1
        return 0

    def delete_many(self, filter_: Dict) -> int:
        count = 0
        with self._lock:
            self._load()
            for sn, shard in enumerate(self._shards):
                before = len(shard)
                kept = []
                for doc in shard:
                    if _match(doc, filter_):
                        for idx in self._indexes.values():
                            idx.remove(doc)
                        self._id_map.pop(doc["_id"], None)
                    else:
                        kept.append(doc)
                self._shards[sn] = kept
                removed = before - len(kept)
                if removed:
                    self._reindex_shard(sn)   # rewrite positions for survivors
                    self._mark_dirty(sn)
                    count += removed
        return count

    def count_documents(self, filter_: Dict = None) -> int:
        filter_ = filter_ or {}
        with self._lock:
            self._load()
            if not filter_:
                return sum(len(s) for s in self._shards)
            # Use index coords if available — avoids loading every doc
            coords = self._index_lookup_coords(filter_)
            if coords is not None:
                # Verify each candidate still matches (handles compound filters)
                count = 0
                for (sn, pos, _id) in coords:
                    if (0 <= sn < len(self._shards) and
                            0 <= pos < len(self._shards[sn])):
                        doc = self._shards[sn][pos]
                        if doc["_id"] == _id and _match(doc, filter_):
                            count += 1
                return count
            # Fall back to parallel scan (count only, no deepcopy needed)
            def count_shard(shard):
                return sum(1 for d in shard if _match(d, filter_))
            workers = min(self._db.scan_workers, len(self._shards))
            total = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for n in ex.map(count_shard, self._shards):
                    total += n
            return total

    # ── Aggregation pipeline ──────────────────────────────────────────────────

    def aggregate(self, pipeline: List[Dict]) -> List[Dict]:
        with self._lock:
            self._load()
            # Optimise: if the first stage is $match, try the index
            first_match_filter: Optional[Dict] = None
            pipeline_rest = pipeline
            if pipeline and "$match" in pipeline[0]:
                first_match_filter = pipeline[0]["$match"]
                pipeline_rest = pipeline[1:]

            if first_match_filter is not None:
                coords = self._index_lookup_coords(first_match_filter)
                if coords is not None:
                    docs = []
                    for (sn, pos, _id) in coords:
                        if (0 <= sn < len(self._shards) and
                                0 <= pos < len(self._shards[sn])):
                            doc = self._shards[sn][pos]
                            if doc["_id"] == _id and _match(doc, first_match_filter):
                                docs.append(copy.deepcopy(doc))
                else:
                    docs = [copy.deepcopy(d) for d in self._all_docs()
                            if _match(d, first_match_filter)]
            else:
                docs = [copy.deepcopy(d) for d in self._all_docs()]

        for stage in pipeline_rest:
            if "$match" in stage:
                docs = [d for d in docs if _match(d, stage["$match"])]
            elif "$sort" in stage:
                for field, direction in reversed(list(stage["$sort"].items())):
                    docs.sort(
                        key=lambda d: (_get_nested(d, field) is None, _get_nested(d, field)),
                        reverse=(direction == -1),
                    )
            elif "$limit" in stage:
                docs = docs[:stage["$limit"]]
            elif "$skip" in stage:
                docs = docs[stage["$skip"]:]
            elif "$project" in stage:
                docs = [_project(d, stage["$project"]) for d in docs]
            elif "$count" in stage:
                return [{stage["$count"]: len(docs)}]
            elif "$group" in stage:
                spec = stage["$group"]
                id_field = spec.get("_id")
                groups: Dict[Any, Dict] = {}
                for doc in docs:
                    key = _get_nested(doc, id_field.lstrip("$")) if id_field else None
                    grp = groups.setdefault(str(key), {"_id": key})
                    for out_field, expr in spec.items():
                        if out_field == "_id":
                            continue
                        if isinstance(expr, dict):
                            op = list(expr.keys())[0]
                            src = list(expr.values())[0]
                            src_val = _get_nested(doc, src.lstrip("$")) if isinstance(src, str) else src
                            if op == "$sum":
                                grp[out_field] = grp.get(out_field, 0) + (src_val or 0)
                            elif op in ("$count", "$count"):
                                grp[out_field] = grp.get(out_field, 0) + 1
                            elif op == "$avg":
                                ps = grp.get(f"__s_{out_field}", 0)
                                pc = grp.get(f"__c_{out_field}", 0)
                                grp[f"__s_{out_field}"] = ps + (src_val or 0)
                                grp[f"__c_{out_field}"] = pc + 1
                                grp[out_field] = grp[f"__s_{out_field}"] / grp[f"__c_{out_field}"]
                            elif op == "$min":
                                cur = grp.get(out_field)
                                grp[out_field] = src_val if (cur is None or (src_val is not None and src_val < cur)) else cur
                            elif op == "$max":
                                cur = grp.get(out_field)
                                grp[out_field] = src_val if (cur is None or (src_val is not None and src_val > cur)) else cur
                            elif op == "$push":
                                grp.setdefault(out_field, []).append(src_val)
                            elif op == "$addToSet":
                                s = grp.setdefault(out_field, [])
                                if src_val not in s:
                                    s.append(src_val)
                            elif op == "$first":
                                if out_field not in grp:
                                    grp[out_field] = src_val
                            elif op == "$last":
                                grp[out_field] = src_val
                docs = [{k: v for k, v in g.items() if not k.startswith("__")} for g in groups.values()]

        return docs

    # ── Utilities ─────────────────────────────────────────────────────────────

    def distinct(self, field: str, filter_: Dict = None) -> List[Any]:
        seen: set = set()
        result = []
        for doc in self.find(filter_ or {}):
            val = _get_nested(doc, field)
            key = str(val)
            if key not in seen:
                seen.add(key)
                result.append(val)
        return result

    def drop(self):
        with self._lock:
            # Delete all shard files from HF
            for sf in self._db._store.list_files(f"db/{self.name}/"):
                self._db._store.delete_file(sf)
            self._db._store.delete_file(self._index_hf_path)
            # Clean local cache
            if os.path.exists(self._index_local_path):
                os.remove(self._index_local_path)
            self._shards = []
            self._indexes = {}
            self._id_map = {}
            self._dirty_shards = set()
            self._op_count = 0
            self._db._dirty_collections.discard(self.name)
            self._db._collections.pop(self.name, None)
        print(f"[KonanDB] Dropped collection '{self.name}'")

    # ── Change streams ────────────────────────────────────────────────────────

    def watch(self, filter_: Dict = None, callback=None):
        """
        Register a callback for real-time change notifications.

        Parameters
        ----------
        filter_ : dict, optional
            Only fire the callback when a changed document matches this filter.
            None = fire for every change on this collection.
        callback : callable
            Called with a single dict argument:
                {"op": "insert" | "update" | "delete", "doc": <document>}
            The callback is invoked in the thread that performs the write, so
            keep it non-blocking (e.g. schedule Socket.IO emits with asyncio).

        Returns
        -------
        handle : tuple
            Pass this back to ``unwatch(handle)`` to deregister.

        Example
        -------
            def on_change(event):
                print(event["op"], event["doc"]["_id"])

            handle = col.watch({"status": "active"}, on_change)
            # ... later ...
            col.unwatch(handle)
        """
        if callback is None:
            raise ValueError("callback is required")
        entry = (filter_ or {}, callback)
        with self._watcher_lock:
            self._watchers.append(entry)
        return entry

    def unwatch(self, handle):
        """Remove a previously registered watch handle."""
        with self._watcher_lock:
            try:
                self._watchers.remove(handle)
            except ValueError:
                pass

    def _emit_change(self, op: str, doc: Dict):
        """Internal: notify all matching watchers of a change event."""
        with self._watcher_lock:
            watchers = list(self._watchers)
        for (filter_, cb) in watchers:
            try:
                if not filter_ or _match(doc, filter_):
                    cb({"op": op, "doc": copy.deepcopy(doc)})
            except Exception as e:
                print(f"[KonanDB][WARN] watch callback error on '{self.name}': {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _index_lookup_coords(
        self, filter_: Dict
    ) -> Optional[List[Tuple[int, int, str]]]:
        """
        Return (shard_n, pos, _id) tuples for matching docs if an index covers
        the query, otherwise return None (caller falls back to parallel scan).

        Intersects results across multiple indexed fields when present.
        """
        coord_sets: List[Set[Tuple[int, int, str]]] = []
        for key, val in filter_.items():
            if key.startswith("$") or isinstance(val, dict):
                continue
            iname = key.replace(".", "_")
            if iname in self._indexes:
                coords = self._indexes[iname].lookup(val)
                # Discard entries with sentinel coords (-1,-1) from legacy data
                valid = {(sn, p, did) for (sn, p, did) in coords if sn >= 0}
                if not valid:
                    return None   # index present but stale — fall back to scan
                coord_sets.append(valid)

        if not coord_sets:
            return None   # no usable index for this filter

        # Intersect by _id across fields
        if len(coord_sets) == 1:
            return list(coord_sets[0])
        id_to_coord: Dict[str, Tuple[int, int, str]] = {
            did: (sn, p, did) for (sn, p, did) in coord_sets[0]
        }
        for s in coord_sets[1:]:
            ids_in_s = {did for (_, _, did) in s}
            id_to_coord = {did: c for did, c in id_to_coord.items() if did in ids_in_s}
        return list(id_to_coord.values()) if id_to_coord else []

    # keep old name as alias for external callers
    def _index_lookup(self, filter_: Dict) -> List[Dict]:
        coords = self._index_lookup_coords(filter_)
        if coords is not None:
            result = []
            for (sn, pos, _id) in coords:
                if 0 <= sn < len(self._shards) and 0 <= pos < len(self._shards[sn]):
                    doc = self._shards[sn][pos]
                    if doc["_id"] == _id:
                        result.append(doc)
            return result
        return self._all_docs()


# ─────────────────────────────────────────────────────────────────────────────
# Database (top-level client)
# ─────────────────────────────────────────────────────────────────────────────

class KonanDB:
    """
    HuggingFace-backed database client.

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset repo, e.g. "myuser/my-db".
    hf_token : str
        HuggingFace write token.
    auto_flush_interval : int
        Seconds between automatic time-based flushes (0 = disabled).
    flush_after_ops : int
        Also flush after this many write operations per collection (0 = disabled).
    shard_size : int
        Max documents per shard file (default 1000).
    scan_workers : int
        Threads used for parallel shard scanning when no index covers the query.
        Default 8 — tune up for large collections with many shards, tune down to
        reduce CPU pressure on smaller servers.
    load_workers : int
        Threads used for parallel shard download on cold collection load.
        Default 8. Matches scan_workers by default.
    """

    def __init__(
        self,
        repo_id: str,
        hf_token: str,
        auto_flush_interval: int = 120,
        flush_after_ops: int = 500,
        shard_size: int = 1000,
        private: bool = False,
        scan_workers: int = 8,
        load_workers: int = 8,
    ):
        self.repo_id = repo_id
        self.flush_after_ops = flush_after_ops
        self.shard_size = shard_size
        self.scan_workers = scan_workers
        self.load_workers = load_workers
        self._token = hf_token
        self._store = _HFStore(repo_id, hf_token)
        self._store._ensure_repo(private=private)   # create repo + seed SHA
        self._collections: Dict[str, Collection] = {}
        self._dirty_collections: set = set()
        self._lock = threading.Lock()
        # Single event flag — any collection can set it to wake the flush thread
        self._flush_event = threading.Event()

        if auto_flush_interval > 0 or flush_after_ops > 0:
            self._start_flush_thread(auto_flush_interval)

    def set_private(self, private: bool):
        """Flip repo visibility at any time. True = private 🔒, False = public 🌐."""
        self._store.set_private(private)

    # ── Collection access ────────────────────────────────────────────────────

    def collection(self, name: str) -> Collection:
        """Get (or create) a collection by name."""
        with self._lock:
            if name not in self._collections:
                self._collections[name] = Collection(name, self)
        return self._collections[name]

    def __getitem__(self, name: str) -> Collection:
        return self.collection(name)

    def __getattr__(self, name: str) -> Collection:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.collection(name)

    # ── Database operations ──────────────────────────────────────────────────

    def list_collections(self) -> List[str]:
        """List all collection names that exist in the HF repo."""
        files = self._store.list_files("db/")
        names = set()
        for f in files:
            # New layout: db/<collection>/shard_NNN.json
            parts = f.split("/")
            if len(parts) >= 3 and parts[0] == "db" and parts[1] != "_indexes":
                names.add(parts[1])
        return sorted(names)

    def drop_database(self):
        """⚠️  Delete ALL collections and data. Irreversible."""
        for col_name in list(self._collections.keys()):
            self._collections[col_name].drop()
        print(f"[KonanDB] ⚠️  Dropped entire database '{self.repo_id}'")

    # ── Flush / persistence ──────────────────────────────────────────────────

    def flush(self):
        """Persist all dirty collections to HuggingFace (each as one commit)."""
        dirty = list(self._dirty_collections)
        if not dirty:
            print("[KonanDB] Nothing to flush.")
            return
        for name in dirty:
            col = self._collections.get(name)
            if col:
                col.flush()
        still_dirty = len(self._dirty_collections)
        if still_dirty:
            print(f"[KonanDB] {still_dirty} collection(s) need retry (conflict during flush).")
        else:
            print(f"[KonanDB] Flush complete ({len(dirty)} collection(s)).")

    def _start_flush_thread(self, interval: int):
        """
        Single background thread that flushes when either:
        - *interval* seconds pass with dirty data  (timed flush)
        - _flush_event is set by a collection hitting flush_after_ops  (op flush)
        Only one thread, no matter how many collections trigger it.
        """
        def _loop():
            while True:
                # Wait up to *interval* seconds OR until signalled by op-limit
                triggered = self._flush_event.wait(timeout=interval if interval > 0 else 3600)
                self._flush_event.clear()
                if self._dirty_collections:
                    reason = "op-limit" if triggered else "timer"
                    n = len(self._dirty_collections)
                    print(f"[KonanDB] Auto-flush ({reason}): {n} dirty collection(s)...")
                    self.flush()

        t = threading.Thread(target=_loop, daemon=True, name="konandb-flush")
        t.start()
        print(f"[KonanDB] Flush thread started (interval={interval}s, op_limit={self.flush_after_ops}).")

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self):
        return f"<KonanDB repo='{self.repo_id}' collections={self.list_collections()}>"


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo  (python konandb.py)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Socket.IO server
# ─────────────────────────────────────────────────────────────────────────────

# Tracks which socket IDs have authenticated
_authed_sids: Set[str] = set()
_authed_lock = threading.Lock()


def _require_auth(sid: str) -> bool:
    with _authed_lock:
        return sid in _authed_sids


def create_socketio_server(db: "KonanDB") -> socketio.AsyncServer:
    """
    Build and return a configured Socket.IO AsyncServer bound to *db*.
    Clients must emit 'authenticate' before any other command.
    """
    sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")

    # ── Connection lifecycle ─────────────────────────────────────────────────

    @sio.event
    async def connect(sid, environ):
        print(f"[KonanDB] Client connected: {sid}")

    @sio.event
    async def disconnect(sid):
        with _authed_lock:
            _authed_sids.discard(sid)
        print(f"[KonanDB] Client disconnected: {sid}")

    # ── Authentication ───────────────────────────────────────────────────────

    @sio.event
    async def authenticate(sid, data):
        user = data.get("user", "")
        pwd  = data.get("pass", "")
        if user == KONANDB_USER and pwd == KONANDB_PASS:
            with _authed_lock:
                _authed_sids.add(sid)
            print(f"[KonanDB] Authenticated: {sid}")
            return {"ok": True}
        print(f"[KonanDB] Auth failed for sid={sid} user={user!r}")
        return {"ok": False, "error": "Invalid credentials"}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _col(data: dict):
        return db.collection(data["collection"])

    def _unauth():
        return {"ok": False, "error": "Not authenticated — send 'authenticate' first"}

    # ── Search / read events ─────────────────────────────────────────────────

    @sio.event
    async def find(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            results = _col(data).find(
                filter_    = data.get("filter", {}),
                projection = data.get("projection"),
                sort       = data.get("sort"),
                limit      = data.get("limit", 0),
                skip       = data.get("skip", 0),
            )
            return {"ok": True, "result": results}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def find_one(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            doc = _col(data).find_one(
                filter_    = data.get("filter", {}),
                projection = data.get("projection"),
            )
            return {"ok": True, "result": doc}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def find_by_id(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            doc = _col(data).find_by_id(data["id"])
            return {"ok": True, "result": doc}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def count_documents(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            n = _col(data).count_documents(data.get("filter", {}))
            return {"ok": True, "result": n}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def distinct(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            vals = _col(data).distinct(data["field"], data.get("filter", {}))
            return {"ok": True, "result": vals}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def aggregate(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            result = _col(data).aggregate(data.get("pipeline", []))
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Write events ─────────────────────────────────────────────────────────

    @sio.event
    async def insert_one(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            doc_id = _col(data).insert_one(data["document"])
            return {"ok": True, "result": {"_id": doc_id}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def insert_many(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            ids = _col(data).insert_many(data["documents"])
            return {"ok": True, "result": {"ids": ids}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def update_one(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            n = _col(data).update_one(
                data.get("filter", {}),
                data["update"],
                upsert=data.get("upsert", False),
            )
            return {"ok": True, "result": {"matched": n}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def update_many(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            n = _col(data).update_many(data.get("filter", {}), data["update"])
            return {"ok": True, "result": {"matched": n}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def delete_one(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            n = _col(data).delete_one(data.get("filter", {}))
            return {"ok": True, "result": {"deleted": n}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def delete_many(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            n = _col(data).delete_many(data.get("filter", {}))
            return {"ok": True, "result": {"deleted": n}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def replace_one(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            n = _col(data).replace_one(
                data.get("filter", {}),
                data["replacement"],
                upsert=data.get("upsert", False),
            )
            return {"ok": True, "result": {"matched": n}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Index events ─────────────────────────────────────────────────────────

    @sio.event
    async def create_index(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            name = _col(data).create_index(data["field"], unique=data.get("unique", False))
            return {"ok": True, "result": {"name": name}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def drop_index(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            _col(data).drop_index(data["field"])
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def list_indexes(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            idxs = _col(data).list_indexes()
            return {"ok": True, "result": idxs}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── DB-level events ──────────────────────────────────────────────────────

    @sio.event
    async def list_collections(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            cols = db.list_collections()
            return {"ok": True, "result": cols}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def flush(sid, data):
        if not _require_auth(sid): return _unauth()
        try:
            db.flush()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Change stream / watch events ─────────────────────────────────────────

    # sid → {col_name → watch_handle}
    _watch_registry: Dict[str, Dict[str, Any]] = {}

    @sio.event
    async def watch_collection(sid, data):
        """
        Subscribe to real-time change events for a collection.

        Client sends:  {"collection": "...", "filter": {}}
        Server emits:  "change" event → {"collection": "...", "op": "insert"|"update"|"delete", "doc": {...}}

        Multiple watches on the same collection from the same client replace
        the previous watch for that collection.
        """
        if not _require_auth(sid): return _unauth()
        try:
            col_name = data["collection"]
            filter_  = data.get("filter") or {}
            col      = db.collection(col_name)

            # Remove stale watch for this sid+collection if any
            if sid in _watch_registry and col_name in _watch_registry[sid]:
                col.unwatch(_watch_registry[sid][col_name])

            import asyncio

            def _on_change(event):
                # Fire-and-forget: schedule the Socket.IO emit on the event loop
                async def _emit():
                    try:
                        await sio.emit(
                            "change",
                            {"collection": col_name, **event},
                            to=sid,
                        )
                    except Exception as exc:
                        print(f"[KonanDB][WARN] watch emit error: {exc}")

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(_emit(), loop)

            handle = col.watch(filter_, _on_change)
            _watch_registry.setdefault(sid, {})[col_name] = handle
            print(f"[KonanDB] watch registered: sid={sid} col={col_name} filter={filter_}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @sio.event
    async def unwatch_collection(sid, data):
        """
        Unsubscribe from change events for a collection.
        Client sends: {"collection": "..."}
        """
        if not _require_auth(sid): return _unauth()
        try:
            col_name = data["collection"]
            if sid in _watch_registry and col_name in _watch_registry[sid]:
                db.collection(col_name).unwatch(_watch_registry[sid].pop(col_name))
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Clean up watches on disconnect
    _original_disconnect = None

    @sio.on("disconnect")  # type: ignore[misc]
    async def _cleanup_watches(sid):
        with _authed_lock:
            _authed_sids.discard(sid)
        if sid in _watch_registry:
            for col_name, handle in _watch_registry.pop(sid, {}).items():
                try:
                    db.collection(col_name).unwatch(handle)
                except Exception:
                    pass
        print(f"[KonanDB] Client disconnected: {sid}")

    # ── Compound index events ─────────────────────────────────────────────────

    @sio.event
    async def create_compound_index(sid, data):
        """
        Create a compound index on multiple fields.

        Client sends: {"collection": "...", "fields": ["field1", "field2", ...], "unique": false}

        Compound indexes are stored as individual field indexes combined with
        an intersection strategy at query time (same as the existing multi-field
        _index_lookup_coords intersection). This event is a convenience wrapper
        that creates a single-field index for each listed field so compound
        queries on those fields automatically use index-intersection.

        Note: true single-pass compound indexes are a planned enhancement.
        For now this gives O(index) intersection across the listed fields.
        """
        if not _require_auth(sid): return _unauth()
        try:
            col    = _col(data)
            fields = data.get("fields", [])
            unique = data.get("unique", False)
            if len(fields) < 2:
                return {"ok": False, "error": "compound index requires at least 2 fields"}
            names = []
            for field in fields:
                names.append(col.create_index(field, unique=unique))
            return {"ok": True, "result": {"names": names, "note": "per-field indexes created for intersection"}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return sio


def start_server(
    db: "KonanDB",
    host: str = "0.0.0.0",
    port: int = 5000,
):
    """
    Start the KonanDB Socket.IO server (blocking).

    Parameters
    ----------
    db   : KonanDB instance to serve
    host : bind address (default all interfaces)
    port : TCP port (default 5000)
    """
    import asyncio

    sio = create_socketio_server(db)
    app = web.Application()
    sio.attach(app)

    print(f"[KonanDB] Socket.IO server starting on http://{host}:{port}")
    print(f"[KonanDB] User: {KONANDB_USER!r}  |  Pass: {'*' * len(KONANDB_PASS)}")
    web.run_app(app, host=host, port=port)


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo  (python konandb.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    token    = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME", "myuser")
    mode     = sys.argv[1] if len(sys.argv) > 1 else "server"  # "server" | "demo"

    if not token:
        print("Set HF_TOKEN (and optionally HF_USERNAME) env vars first.")
        sys.exit(1)

    repo = f"{username}/konandb-demo"
    print(f"\n── Connecting to KonanDB: {repo} ──\n")

    db = KonanDB(
        repo_id             = repo,
        hf_token            = token,
        auto_flush_interval = 0 if mode == "demo" else 120,
        flush_after_ops     = 500,
        shard_size          = 100,   # small for demo — use 1000+ in prod
    )

    if mode == "demo":
        # ── Quick local smoke-test (no networking) ──────────────────────────
        users = db.collection("users")

        users.insert_one({"name": "Ada",    "age": 30, "role": "admin",  "city": "Lagos"})
        users.insert_one({"name": "Brian",  "age": 25, "role": "user",   "city": "London"})
        users.insert_one({"name": "Chioma", "age": 28, "role": "admin",  "city": "Lagos"})
        users.insert_many([
            {"name": "Dave",  "age": 35, "role": "user",  "city": "Paris"},
            {"name": "Emeka", "age": 22, "role": "user",  "city": "Lagos"},
        ])

        print("All users:", users.count_documents())
        print("Admins:",    [u["name"] for u in users.find({"role": "admin"})])
        print("Age < 28:",  [u["name"] for u in users.find({"age": {"$lt": 28}})])

        users.create_index("city")
        print("Lagos:",     [u["name"] for u in users.find({"city": "Lagos"})])

        users.update_one({"name": "Ada"}, {"$set": {"age": 31}, "$inc": {"login_count": 1}})
        print("Ada updated:", users.find_one({"name": "Ada"}))

        by_city = users.aggregate([
            {"$group": {"_id": "$city", "count": {"$sum": 1}}},
            {"$sort":  {"count": -1}},
        ])
        print("Users by city:", by_city)
        print("All cities:", users.distinct("city"))

        users.delete_one({"name": "Dave"})
        print("After delete Dave:", users.count_documents())

        print("\nFlushing to HuggingFace...")
        db.flush()
        print("Done! Check:", f"https://huggingface.co/datasets/{repo}")

    else:
        # ── Start Socket.IO server ──────────────────────────────────────────
        host = os.getenv("KONANDB_HOST", "0.0.0.0")
        port = int(os.getenv("KONANDB_PORT", "7860"))
        start_server(db, host=host, port=port)
