"""Visual memory: everything JARVIS has seen, and everything it learned from it.

Five stores share the assistant's SQLite file:

    observations  each time it looked at the screen, with text and a vector
    ui_map        where labelled controls live, per app, refined by use
    procedures    goal -> action sequence, replayable once it has worked
    actions       every click and keystroke with the diff it caused
    habits        which apps are open when, so it can anticipate

The reinforcement loop is the point of the `ui_map`: coordinates are not
trusted blindly, they carry hit/miss counts, and a failed click demotes them
so the next attempt re-locates the target from scratch.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .embeddings import Embedder
from .perception import Element, Observation

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    app          TEXT NOT NULL DEFAULT '',
    window_title TEXT NOT NULL DEFAULT '',
    width        INTEGER NOT NULL DEFAULT 0,
    height       INTEGER NOT NULL DEFAULT 0,
    text         TEXT NOT NULL DEFAULT '',
    thumb_path   TEXT NOT NULL DEFAULT '',
    change       REAL NOT NULL DEFAULT 0,
    reason       TEXT NOT NULL DEFAULT 'look',
    vec          BLOB
);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts);
CREATE INDEX IF NOT EXISTS idx_obs_app ON observations(app);

CREATE TABLE IF NOT EXISTS elements (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    obs_id     INTEGER NOT NULL,
    label      TEXT NOT NULL,
    x          INTEGER NOT NULL,
    y          INTEGER NOT NULL,
    w          INTEGER NOT NULL,
    h          INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    actionable INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_el_obs ON elements(obs_id);
CREATE INDEX IF NOT EXISTS idx_el_label ON elements(label);

CREATE TABLE IF NOT EXISTS ui_map (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    app        TEXT NOT NULL,
    label      TEXT NOT NULL,
    x          INTEGER NOT NULL,
    y          INTEGER NOT NULL,
    w          INTEGER NOT NULL DEFAULT 0,
    h          INTEGER NOT NULL DEFAULT 0,
    hits       INTEGER NOT NULL DEFAULT 0,
    misses     INTEGER NOT NULL DEFAULT 0,
    ts         REAL NOT NULL,
    UNIQUE(app, label)
);

CREATE TABLE IF NOT EXISTS procedures (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    goal      TEXT NOT NULL,
    app       TEXT NOT NULL DEFAULT '',
    steps     TEXT NOT NULL,
    successes INTEGER NOT NULL DEFAULT 0,
    failures  INTEGER NOT NULL DEFAULT 0,
    ts        REAL NOT NULL,
    UNIQUE(goal, app)
);

CREATE TABLE IF NOT EXISTS actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    kind       TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    x          INTEGER,
    y          INTEGER,
    app        TEXT NOT NULL DEFAULT '',
    obs_before INTEGER,
    obs_after  INTEGER,
    change     REAL NOT NULL DEFAULT 0,
    verdict    TEXT NOT NULL DEFAULT 'unknown',
    detail     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_act_ts ON actions(ts);

CREATE TABLE IF NOT EXISTS habits (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    key   TEXT NOT NULL,
    value REAL NOT NULL DEFAULT 0,
    ts    REAL NOT NULL,
    UNIQUE(scope, key)
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    body, content=''
);
CREATE TRIGGER IF NOT EXISTS obs_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, body) VALUES (new.id, new.text || ' ' || new.window_title);
END;
CREATE TRIGGER IF NOT EXISTS obs_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, body)
    VALUES ('delete', old.id, old.text || ' ' || old.window_title);
END;
"""


@dataclass
class UiEntry:
    app: str
    label: str
    x: int
    y: int
    w: int = 0
    h: int = 0
    hits: int = 0
    misses: int = 0
    ts: float = 0.0

    @property
    def confidence(self) -> float:
        """Posterior-ish score: a few hits outweigh many misses."""
        total = self.hits + self.misses
        if total == 0:
            return 0.35
        return (self.hits + 1.0) / (total + 2.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "hits": self.hits,
            "misses": self.misses,
            "confidence": round(self.confidence, 3),
            "last_used": time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))
            if self.ts
            else None,
        }


def _pack(vector: np.ndarray | None) -> bytes | None:
    return None if vector is None else np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes | None, dim: int) -> np.ndarray | None:
    if not blob or len(blob) < 4:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def _moved_far(
    old_x: int, old_y: int, old_w: int, old_h: int,
    new_x: int, new_y: int, new_w: int, new_h: int,
) -> bool:
    """Has a control moved enough that its old hit record no longer applies?

    A tolerance is needed because the map is fed from every OCR match, and a
    control that is re-detected at a slightly different pixel each time would
    otherwise reset its history on every observation and never learn at all.
    So the bar scales with the control: a small button has to shift a fraction
    of its own size, a full-width row has to shift a lot.

    When either size is unknown the comparison falls back to a fixed distance,
    since a missing size makes the scaled threshold meaningless.
    """
    if old_w <= 0 or old_h <= 0 or new_w <= 0 or new_h <= 0:
        limit = 24
    else:
        limit = max(12, max(old_w, old_h, new_w, new_h) // 4)
    dx = abs((new_x + new_w // 2) - (old_x + old_w // 2))
    dy = abs((new_y + new_h // 2) - (old_y + old_h // 2))
    return dx > limit or dy > limit


class VisionMemory:
    def __init__(self, path: str | Path, embedder: Embedder | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.embedder = embedder
        self.dim = getattr(embedder, "dim", 0) if embedder else 0
        self.fts = False
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            # An index built against a content table has to be thrown away and
            # remade, not left in place: FTS5 only creates it if absent, so
            # changing the declaration below would not reach an existing file.
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='observations_fts'"
            ).fetchone()
            if row and "content='observations'" in (row[0] or ""):
                self._conn.execute("DROP TABLE observations_fts")
            try:
                self._conn.executescript(FTS_SCHEMA)
                self._rebuild_fts()
                self.fts = True
            except sqlite3.OperationalError:
                self.fts = False
            self._conn.commit()

    def _rebuild_fts(self) -> None:
        """Refill the keyword index from the observations table.

        FTS5's own 'rebuild' reads the content table it was declared against,
        and this index is fed by triggers instead, so it is refilled by hand.
        Contentless tables also have no 'rebuild' at all. Callers check `self.fts`
        except in __init__, which is inside the try that sets it.
        """
        self._conn.execute(
            "INSERT INTO observations_fts(observations_fts) VALUES('delete-all')"
        )
        self._conn.execute(
            "INSERT INTO observations_fts(rowid, body) "
            "SELECT id, text || ' ' || window_title FROM observations"
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def set_embedder(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.dim = getattr(embedder, "dim", 0)

    def _embed(self, text: str) -> np.ndarray | None:
        if not self.embedder or not text.strip():
            return None
        try:
            return self.embedder.encode_one(text)
        except Exception:  # noqa: BLE001
            return None

    # -- observations ----------------------------------------------------
    def record(self, obs: Observation, reason: str = "look") -> int:
        vector = self._embed(obs.summary_text)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO observations "
                "(ts, app, window_title, width, height, text, thumb_path, change, reason, vec) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    obs.ts, obs.window_app, obs.window_title, obs.width, obs.height,
                    obs.text, obs.thumb_path, obs.change, reason, _pack(vector),
                ),
            )
            obs_id = int(cur.lastrowid)
            self._conn.executemany(
                "INSERT INTO elements (obs_id, label, x, y, w, h, confidence, actionable) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (obs_id, e.text, e.x, e.y, e.w, e.h, e.confidence, int(e.actionable))
                    for e in obs.elements
                ],
            )
            self._note_habit_locked(obs.window_app, obs.ts)
            self._conn.commit()
        return obs_id

    def get_observation(self, obs_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM observations WHERE id = ?", (obs_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "ts": row["ts"],
            "app": row["app"],
            "window": row["window_title"],
            "size": [row["width"], row["height"]],
            "text": row["text"],
            "thumbnail": row["thumb_path"],
            "reason": row["reason"],
        }

    def elements_of(self, obs_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT label,x,y,w,h,confidence,actionable FROM elements "
                "WHERE obs_id = ? ORDER BY y, x",
                (obs_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]

    def prune(self, retention_days: int, max_rows: int) -> int:
        cutoff = time.time() - max(1, retention_days) * 86400
        removed = 0
        with self._lock:
            cur = self._conn.execute("DELETE FROM observations WHERE ts < ?", (cutoff,))
            removed = cur.rowcount
            if self.fts:
                self._rebuild_fts()
            total = self._conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]
            if total > max_rows:
                excess = total - max_rows
                keep = [
                    r["id"]
                    for r in self._conn.execute(
                        "SELECT id FROM observations ORDER BY ts DESC LIMIT ?", (max_rows,)
                    )
                ]
                placeholders = ",".join("?" * len(keep))
                cur = self._conn.execute(
                    f"DELETE FROM observations WHERE id NOT IN ({placeholders})", keep
                )
                removed += cur.rowcount
                self._conn.execute(
                    f"DELETE FROM elements WHERE obs_id NOT IN ({placeholders})", keep
                )
                if self.fts:
                    self._rebuild_fts()
                self._conn.execute(
                    "DELETE FROM actions WHERE obs_before NOT IN (SELECT id FROM observations) "
                    "OR obs_after NOT IN (SELECT id FROM observations)"
                )
            self._conn.commit()
        return removed

    def wipe(self) -> dict[str, int]:
        with self._lock:
            counts = {
                table: self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in ("observations", "elements", "ui_map", "actions")
            }
            for table in ("observations", "elements", "ui_map", "actions"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.execute("DELETE FROM sqlite_sequence WHERE name IN "
                               "('observations','elements','ui_map','actions')")
            if self.fts:
                self._conn.execute(
                    "INSERT INTO observations_fts(observations_fts) VALUES('delete-all')"
                )
            self._conn.commit()
        return counts

    # -- recall ----------------------------------------------------------
    def recall(self, query: str, limit: int = 8, app: str = "") -> list[dict[str, Any]]:
        """Hybrid recall: vector similarity blended with keyword hits."""
        results: dict[int, dict[str, Any]] = {}

        with self._lock:
            rows: list[sqlite3.Row] = []
            if app:
                rows = self._conn.execute(
                    "SELECT id,ts,app,window_title,text,thumb_path,change,vec "
                    "FROM observations WHERE app = ? ORDER BY ts DESC LIMIT 400",
                    (app,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id,ts,app,window_title,text,thumb_path,change,vec "
                    "FROM observations ORDER BY ts DESC LIMIT 800"
                ).fetchall()

        for rank, row in enumerate(rows):
            results[row["id"]] = {
                "id": row["id"],
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(row["ts"])),
                "app": row["app"],
                "window": row["window_title"],
                "thumbnail": row["thumb_path"],
                "text": row["text"][:500],
                "score": 0.0,
                "matched": [],
            }

        # Keyword pass.
        if self.fts and query.strip():
            words = [w for w in query.lower().split() if len(w) > 2][:6]
            if words:
                match = " OR ".join(f'"{w}"' for w in words)
                try:
                    for hit in self._conn.execute(
                        "SELECT rowid FROM observations_fts WHERE observations_fts MATCH ? "
                        "ORDER BY rank LIMIT 40",
                        (match,),
                    ):
                        if hit["rowid"] in results:
                            results[hit["rowid"]]["score"] += 0.6
                            results[hit["rowid"]]["matched"].append("keyword")
                except sqlite3.OperationalError:
                    pass
        elif query.strip():
            like = f"%{query.strip()}%"
            for row in self._conn.execute(
                "SELECT id FROM observations WHERE text LIKE ? COLLATE NOCASE "
                "OR window_title LIKE ? COLLATE NOCASE LIMIT 40",
                (like, like),
            ):
                if row["id"] in results:
                    results[row["id"]]["score"] += 0.5
                    results[row["id"]]["matched"].append("keyword")

        # Vector pass.
        if self.embedder and rows:
            query_vec = self._embed(query)
            if query_vec is not None and query_vec.size == self.dim:
                for row in rows:
                    stored = _unpack(row["vec"], self.dim)
                    if stored is None or stored.size != query_vec.size:
                        continue
                    similarity = float(np.dot(query_vec, stored))
                    entry = results.get(row["id"])
                    if entry is not None:
                        entry["score"] += max(0.0, similarity) * 1.2
                        entry["matched"].append("semantic")

        # Recency nudge so "what was that screen" prefers recent frames.
        now = time.time()
        ts_by_id = {r["id"]: r["ts"] for r in rows}
        for obs_id, entry in results.items():
            ts = ts_by_id.get(obs_id)
            if ts is None:
                continue
            age_days = (now - ts) / 86400.0
            entry["score"] += max(0.0, 0.25 - age_days * 0.01)

        ranked = sorted(results.values(), key=lambda r: r["score"], reverse=True)
        return [r for r in ranked if r["score"] > 0.15][:limit]

    # -- ui map ----------------------------------------------------------
    def learn_element(
        self, app: str, label: str, element: Element, ts: float | None = None
    ) -> None:
        app = (app or "unknown").lower()
        label = label.strip()
        if not label:
            return
        ts = ts or time.time()
        x, y, w, h = int(element.x), int(element.y), int(element.w), int(element.h)
        with self._lock:
            row = self._conn.execute(
                "SELECT x, y, w, h FROM ui_map WHERE app = ? AND label = ? COLLATE NOCASE",
                (app, label),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO ui_map (app, label, x, y, w, h, hits, misses, ts) "
                    "VALUES (?,?,?,?,?,?,0,0,?)",
                    (app, label, x, y, w, h, ts),
                )
            elif _moved_far(row["x"], row["y"], row["w"], row["h"], x, y, w, h):
                # hits and misses describe a position, not a name. Carrying them
                # across a move leaves the control fully confident about a
                # coordinate where it no longer is, which is what gets clicked
                # when OCR cannot see it. A move therefore starts it unproven.
                self._conn.execute(
                    "UPDATE ui_map SET x=?, y=?, w=?, h=?, hits=0, misses=0, ts=? "
                    "WHERE app = ? AND label = ? COLLATE NOCASE",
                    (x, y, w, h, ts, app, label),
                )
            else:
                self._conn.execute(
                    "UPDATE ui_map SET x=?, y=?, w=?, h=?, ts=? "
                    "WHERE app = ? AND label = ? COLLATE NOCASE",
                    (x, y, w, h, ts, app, label),
                )
            self._conn.commit()

    def remember_position(self, app: str, label: str, x: int, y: int, w: int = 0, h: int = 0) -> None:
        self.learn_element(app, label, Element(text=label, x=int(x), y=int(y), w=int(w), h=int(h)))

    def reinforce(self, app: str, label: str, hit: bool) -> None:
        app = (app or "unknown").lower()
        column = "hits" if hit else "misses"
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE ui_map SET {column} = {column} + 1, ts = ? "
                "WHERE app = ? AND label = ? COLLATE NOCASE",
                (time.time(), app, label.strip()),
            )
            if cur.rowcount == 0 and not hit:
                return
            self._conn.commit()

    def known(self, app: str, label: str) -> UiEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ui_map WHERE app = ? AND label = ? COLLATE NOCASE",
                ((app or "unknown").lower(), label.strip()),
            ).fetchone()
        if row is None:
            return None
        return UiEntry(row["app"], row["label"], row["x"], row["y"], row["w"],
                       row["h"], row["hits"], row["misses"], row["ts"])

    def known_all(self, app: str = "", limit: int = 60) -> list[UiEntry]:
        with self._lock:
            if app:
                rows = self._conn.execute(
                    "SELECT * FROM ui_map WHERE app = ? ORDER BY hits DESC, ts DESC LIMIT ?",
                    ((app or "").lower(), limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM ui_map ORDER BY hits DESC, ts DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            UiEntry(r["app"], r["label"], r["x"], r["y"], r["w"], r["h"],
                    r["hits"], r["misses"], r["ts"])
            for r in rows
        ]

    def forget_element(self, app: str, label: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM ui_map WHERE app = ? AND label = ? COLLATE NOCASE",
                ((app or "unknown").lower(), label.strip()),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def forget_app(self, app: str) -> int:
        """Drop everything learned about one app. Returns rows removed."""
        token = (app or "").strip().lower()
        if not token:
            return 0
        with self._lock:
            total = 0
            for table in ("ui_map", "observations", "actions", "procedures"):
                cur = self._conn.execute(f"DELETE FROM {table} WHERE app = ?", (token,))
                total += max(0, cur.rowcount)
            self._conn.commit()
        return total

    def find_label(
        self, app: str, needle: str, min_confidence: float = 0.3
    ) -> list[UiEntry]:
        """Fuzzy lookup over learned labels, best match first."""
        needle = needle.strip().lower()
        if not needle:
            return []
        scored: list[tuple[float, UiEntry]] = []
        for entry in self.known_all(app, 400):
            label = entry.label.lower()
            if label == needle:
                score = 1.0
            elif needle in label or label in needle:
                score = 0.8
            else:
                a, b = set(needle.split()), set(label.split())
                score = len(a & b) / len(a | b) if (a | b) else 0.0
            confidence = score * entry.confidence
            if confidence >= min_confidence:
                scored.append((confidence, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _score, entry in scored]

    # -- actions and self-correction ------------------------------------
    def record_action(
        self,
        kind: str,
        label: str = "",
        x: int | None = None,
        y: int | None = None,
        app: str = "",
        obs_before: int | None = None,
        obs_after: int | None = None,
        change: float = 0.0,
        verdict: str = "unknown",
        detail: str = "",
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO actions (ts,kind,label,x,y,app,obs_before,obs_after,change,"
                "verdict,detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), kind, label, x, y, (app or "").lower(),
                 obs_before, obs_after, change, verdict, detail),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_actions(self, limit: int = 20, verdict: str = "") -> list[dict[str, Any]]:
        with self._lock:
            if verdict:
                rows = self._conn.execute(
                    "SELECT * FROM actions WHERE verdict = ? ORDER BY ts DESC LIMIT ?",
                    (verdict, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM actions ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            {
                "when": time.strftime("%H:%M:%S", time.localtime(r["ts"])),
                "kind": r["kind"],
                "label": r["label"],
                "at": [r["x"], r["y"]] if r["x"] is not None else None,
                "app": r["app"],
                "change": round(r["change"], 4),
                "verdict": r["verdict"],
                "detail": r["detail"],
            }
            for r in rows
        ]

    def failure_rate(self, label: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total, SUM(verdict = 'no_change') AS bad "
                "FROM actions WHERE label = ? COLLATE NOCASE",
                (label,),
            ).fetchone()
        if not row or not row["total"]:
            return 0.0
        return (row["bad"] or 0) / row["total"]

    # -- procedures ------------------------------------------------------
    def save_procedure(self, goal: str, steps: Sequence[dict[str, Any]], app: str = "") -> None:
        payload = json.dumps(list(steps))
        goal_key = goal.strip().lower()
        app_key = (app or "").lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT steps, successes, failures FROM procedures "
                "WHERE goal = ? AND app = ?",
                (goal_key, app_key),
            ).fetchone()
            if row is not None and row["steps"] == payload:
                self._conn.execute(
                    "UPDATE procedures SET ts = ? WHERE goal = ? AND app = ?",
                    (time.time(), goal_key, app_key),
                )
            elif row is not None:
                # The record scores the steps, not the goal. Re-teaching a goal
                # with different steps and keeping the old tally made JARVIS
                # report steps it had never run as proven, with the successes of
                # the steps they replaced.
                self._conn.execute(
                    "UPDATE procedures SET steps = ?, successes = 0, failures = 0, ts = ? "
                    "WHERE goal = ? AND app = ?",
                    (payload, time.time(), goal_key, app_key),
                )
            else:
                self._conn.execute(
                    "INSERT INTO procedures (goal, app, steps, successes, failures, ts) "
                    "VALUES (?,?,?,0,0,?)",
                    (goal_key, app_key, payload, time.time()),
                )
            self._conn.commit()

    def get_procedure(self, goal: str, app: str = "") -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM procedures WHERE goal = ? AND app = ?",
                (goal.strip().lower(), (app or "").lower()),
            ).fetchone()
        if row is None:
            return None
        return {
            "goal": row["goal"],
            "app": row["app"],
            "steps": json.loads(row["steps"]),
            "successes": row["successes"],
            "failures": row["failures"],
            "ts": row["ts"],
        }

    def find_procedures(self, goal: str, limit: int = 5) -> list[dict[str, Any]]:
        needle = set(goal.strip().lower().split())
        out: list[tuple[float, dict[str, Any]]] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM procedures ORDER BY successes DESC, ts DESC LIMIT 100"
            ).fetchall()
        for row in rows:
            words = set(row["goal"].split())
            score = len(needle & words) / len(needle | words) if (needle | words) else 0
            total = row["successes"] + row["failures"]
            rate = (row["successes"] + 1) / (total + 2)
            out.append((score * rate, {
                "goal": row["goal"],
                "app": row["app"],
                "steps": json.loads(row["steps"]),
                "successes": row["successes"],
                "failures": row["failures"],
                "ts": row["ts"],
            }))
        out.sort(key=lambda pair: pair[0], reverse=True)
        return [p for score, p in out if score > 0.1][:limit]

    def score_procedure(self, goal: str, app: str, success: bool) -> None:
        column = "successes" if success else "failures"
        with self._lock:
            self._conn.execute(
                f"UPDATE procedures SET {column} = {column} + 1, ts = ? "
                "WHERE goal = ? AND app = ?",
                (time.time(), goal.strip().lower(), (app or "").lower()),
            )
            self._conn.commit()

    def all_procedures(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT goal, app, steps, successes, failures, ts FROM procedures "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "goal": r["goal"],
                "app": r["app"],
                "step_count": len(json.loads(r["steps"])),
                "successes": r["successes"],
                "failures": r["failures"],
                "ts": r["ts"],
            }
            for r in rows
        ]

    # -- habits ----------------------------------------------------------
    def _note_habit_locked(self, app: str, ts: float) -> None:
        if not app:
            return
        app = app.lower()
        hour = time.strftime("%H", time.localtime(ts))
        weekday = time.strftime("%a", time.localtime(ts)).lower()
        for scope, key in (("app_total", app), ("app_hour", f"{app}@{hour}"),
                           ("app_dow", f"{app}@{weekday}")):
            self._conn.execute(
                "INSERT INTO habits (scope, key, value, ts) VALUES (?,?,1,?) "
                "ON CONFLICT(scope, key) DO UPDATE SET value = value + 1, ts = excluded.ts",
                (scope, key, ts),
            )

    def habits(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope, key, value, ts FROM habits ORDER BY value DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "scope": r["scope"],
                "key": r["key"],
                "count": int(r["value"]),
                "last": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"])),
            }
            for r in rows
        ]

    def likely_apps(self, limit: int = 5) -> list[dict[str, Any]]:
        """Which apps are plausibly open right now, best first.

        The habit counts are cumulative and never decay, so ranking on them
        alone reported an app last used a year ago as the most likely to be
        open, ahead of one opened seconds earlier. This is exposed to the model
        as `likely_open_now`, so it has to mean now. Volume is therefore scaled
        by recency, with a 30-day half-life, and the age is reported so a
        caller can see how much to trust the suggestion.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, ts FROM habits WHERE scope = 'app_total' "
                "ORDER BY value DESC LIMIT 200",
            ).fetchall()
        now = time.time()
        now_hour = int(time.strftime("%H"))
        out = []
        for r in rows:
            app = r["key"]
            hour_row = self._conn.execute(
                "SELECT value FROM habits WHERE scope = 'app_hour' AND key = ?",
                (f"{app}@{now_hour:02d}",),
            ).fetchone()
            age_days = max(0.0, (now - float(r["ts"] or now)) / 86400.0)
            recency = 0.5 ** (age_days / 30.0)
            seen = int(r["value"])
            out.append({
                "app": app,
                "seen": seen,
                "at_this_hour": int(hour_row["value"]) if hour_row else 0,
                "last": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"])),
                "days_since_last": round(age_days, 1),
                "likelihood": round(seen * recency, 2),
            })
        out.sort(key=lambda row: (row["likelihood"], -row["days_since_last"]),
                 reverse=True)
        return out[:limit]

    # -- stats -----------------------------------------------------------
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                table: self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()["n"]
                for table in ("observations", "elements", "ui_map", "actions",
                              "procedures", "habits")
            }
