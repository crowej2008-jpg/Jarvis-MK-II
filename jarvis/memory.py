"""Persistent memory: conversation history, long-term facts, and notes.

Everything lands in one SQLite file. Conversation turns are also mirrored into
an FTS5 index when the local SQLite build supports it, so recall is keyword
search rather than a full table scan; otherwise it degrades to LIKE matching.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    role      TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);

CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    ts         REAL NOT NULL,
    pinned     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS notes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    text    TEXT NOT NULL,
    tags    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_notes_ts ON notes(ts);

-- The oldest message id the replayed-history window currently starts at.
-- Persisted because it has to survive between processes: recomputing it each
-- turn is what made the window slide, see growing_window().
CREATE TABLE IF NOT EXISTS prompt_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Added after the first release. Existing databases need the column before any
# INSERT that mentions it will work, and CREATE TABLE IF NOT EXISTS will not add
# it to a table that is already there.
MIGRATIONS = (
    ("messages", "confirmed", "INTEGER NOT NULL DEFAULT 0"),
)

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
"""


@dataclass
class Message:
    role: str
    content: str
    ts: float = 0.0

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


_WINDOW_ANCHOR = "history_window_anchor_id"


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in text).split() if len(w) > 1]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words[:24])


class Memory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.fts = False
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            for table, column, decl in MIGRATIONS:
                have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            try:
                self._conn.executescript(FTS_SCHEMA)
                self.fts = True
            except sqlite3.OperationalError:
                self.fts = False
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- conversation history -------------------------------------------
    def add_message(
        self, role: str, content: str, confirmed: bool = False
    ) -> int:
        if not content:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (ts, role, content, confirmed) VALUES (?, ?, ?, ?)",
                (time.time(), role, content, int(bool(confirmed))),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def confirm(self, message_id: int) -> bool:
        """Mark one of the assistant's own replies as verified.

        Only ever called from outside the model, when something other than the
        model established the statement was right. A model that could set this
        flag itself would be able to certify its own guesses, which is the exact
        failure this flag exists to prevent.
        """
        if not message_id:
            return False
        with self._lock:
            cur = self._conn.execute(
                "UPDATE messages SET confirmed = 1 WHERE id = ? AND role != 'user'",
                (int(message_id),),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def unconfirm(self, message_id: int) -> bool:
        """Withdraw a confirmation, for when the user says the reply was wrong.

        A flag that can be set but never cleared turns one bad confirmation into
        a permanent wrong answer.
        """
        if not message_id:
            return False
        with self._lock:
            cur = self._conn.execute(
                "UPDATE messages SET confirmed = 0 WHERE id = ?", (int(message_id),)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def last_reply_id(self) -> int:
        """Id of the most recent non-user message, or 0 if there is none."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM messages WHERE role != 'user' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return int(row["id"]) if row else 0

    def unconfirmed_replies(self) -> list[dict[str, Any]]:
        """The assistant's own recent replies, oldest first, awaiting judgement.

        Used to attach a user's "yes, that's right" to the reply it refers to,
        which cannot be done by position alone if several turns have passed.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, role, content, ts FROM messages
                WHERE role != 'user' AND confirmed = 0
                ORDER BY id DESC LIMIT 12
                """
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def recent(self, limit: int = 40) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM messages ORDER BY id DESC LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
        return [Message(r["role"], r["content"], r["ts"]) for r in reversed(rows)]

    def history_for(self, limit: int = 40) -> list[dict[str, str]]:
        return [m.as_dict() for m in self.recent(limit)]

    def growing_window(
        self, max_chars: int, trim_slack: float = 0.25
    ) -> list[dict[str, str]]:
        """An append-only view of the transcript, anchored and trimmed in chunks.

        `recent()` is newest-n, so the message sitting at position 0 changes on
        every turn. Ollama can only reuse a cached prompt when the new prompt
        starts with the same tokens, so a sliding window makes the whole history
        block re-prefilled on every turn. Measured on real turns that was
        9.78-17.01s, against 0.15s when the prefix was stable.

        Two obvious fixes both fail, and it is worth being explicit about why,
        because both look correct on inspection:

        "Newest messages that fit the budget" slides just as much as recent().
        Once a conversation is longer than the budget it is over budget on
        *every* later turn, so it drops one message per turn. Measured: a
        23,662-char transcript against an 8,000-char budget moved position 0 on
        every single turn, for a 1.2x improvement rather than the ~100x the cache
        is worth.

        Trimming back by a slack margin per call, recomputed each time, also
        slides - and worse, it looked correct in a simulation. A chunked
        overshoot measured 1 move in 60 turns offline but 6 in 7 on the live
        transcript, because the break fires on the first message once the budget
        is already spent, so the window creeps forward one message per turn
        regardless of the chunk size. An offline harness with uniform synthetic
        messages cannot see that; the real transcript, with wildly uneven
        message lengths, can.

        So the anchor is *persisted* rather than recomputed. It moves only when
        the budget is genuinely exceeded, and then it jumps forward far enough
        to leave `trim_slack` of the budget free, so it stays put until that
        much new conversation has accumulated. Between trims the returned list
        only ever grows, which is the property the cache needs.

        Returns oldest-first, which is the order the model needs.
        """
        budget = max(0, int(max_chars))
        if budget <= 0:
            return []
        slack = min(0.9, max(0.0, float(trim_slack)))

        with self._lock:
            anchor = self._prompt_state_get(_WINDOW_ANCHOR)
            anchor_id = None
            if anchor is not None:
                try:
                    anchor_id = int(anchor)
                except ValueError:
                    anchor_id = None
            if anchor_id is not None:
                alive = self._conn.execute(
                    "SELECT 1 FROM messages WHERE id = ?", (anchor_id,)
                ).fetchone()
                if not alive:
                    # Messages were pruned out from under the anchor. Re-seed
                    # rather than return a window with a hole in it.
                    anchor_id = None

            if anchor_id is None:
                # First call, or the anchor was invalidated. Seed it to the
                # oldest message that still fits the budget.
                total = 0
                seed_id = None
                for row in self._conn.execute(
                    "SELECT id, content FROM messages ORDER BY id DESC"
                ):
                    content = row["content"] or ""
                    if total + len(content) > budget and seed_id is not None:
                        break
                    total += len(content)
                    seed_id = row["id"]
                if seed_id is None:
                    return []
                anchor_id = seed_id
                self._prompt_state_set(_WINDOW_ANCHOR, str(anchor_id))

            rows = self._conn.execute(
                "SELECT id, role, content FROM messages WHERE id >= ? ORDER BY id",
                (anchor_id,),
            ).fetchall()

            total = sum(len(r["content"] or "") for r in rows)
            if total > budget and len(rows) > 1:
                # Genuinely over budget. Jump the anchor forward until what
                # remains fits within (1 - slack) of the budget, so it will not
                # need to move again until that much has been said.
                target = int(budget * (1.0 - slack))
                drop = total - target
                dropped = 0
                new_anchor = anchor_id
                for row in rows:
                    if dropped >= drop and new_anchor != anchor_id:
                        break
                    n = len(row["content"] or "")
                    dropped += n
                    new_anchor = row["id"]
                # Never trim away the entire window.
                if new_anchor > anchor_id:
                    self._prompt_state_set(_WINDOW_ANCHOR, str(new_anchor))
                    rows = [r for r in rows if r["id"] >= new_anchor]
                    if not rows:
                        rows = self._conn.execute(
                            "SELECT id, role, content FROM messages "
                            "WHERE id >= ? ORDER BY id", (new_anchor,),
                        ).fetchall()

        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def _prompt_state_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM prompt_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def _prompt_state_set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO prompt_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def reset_history_window(self) -> None:
        """Forget the window anchor, so the next growing_window() re-seeds it."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM prompt_state WHERE key = ?", (_WINDOW_ANCHOR,)
            )
            self._conn.commit()

    def search_messages(self, query: str, limit: int = 15) -> list[dict[str, Any]]:
        with self._lock:
            if self.fts:
                match = _fts_query(query)
                if not match:
                    return []
                try:
                    rows = self._conn.execute(
                        """
                        SELECT m.role, m.content, m.ts, m.confirmed,
                               snippet(messages_fts, 0, '[', ']', ' ... ', 12) AS snip
                        FROM messages_fts
                        JOIN messages m ON m.id = messages_fts.rowid
                        WHERE messages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (match, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                if rows:
                    return [dict(r) for r in rows]
            like = f"%{query.strip()}%"
            rows = self._conn.execute(
                """
                SELECT role, content, ts, confirmed, '' AS snip FROM messages
                WHERE content LIKE ? COLLATE NOCASE
                ORDER BY ts DESC LIMIT ?
                """,
                (like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_messages(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM messages")
            if self.fts:
                self._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
            self._conn.commit()
            return cur.rowcount

    def message_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]

    # -- long-term facts -------------------------------------------------
    def remember(self, key: str, value: str, pinned: bool = False) -> None:
        key = key.strip().lower()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO facts (key, value, ts, pinned) VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts
                """,
                (key, value, time.time(), int(pinned)),
            )
            self._conn.commit()

    def recall(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key.strip().lower(),)
            ).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, ts FROM facts ORDER BY pinned DESC, key"
            ).fetchall()
        return [dict(r) for r in rows]

    def forget(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE key = ?", (key.strip().lower(),))
            self._conn.commit()
            return cur.rowcount > 0

    def fact_block(self) -> str:
        facts = self.all_facts()
        if not facts:
            return ""
        lines = "\n".join(f"- {f['key']}: {f['value']}" for f in facts)
        return "Known facts about the user:\n" + lines

    # -- notes -----------------------------------------------------------
    def add_note(self, text: str, tags: Sequence[str] | str = ()) -> int:
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        tag_str = ", ".join(sorted({t.strip() for t in tags if t.strip()}))
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO notes (ts, text, tags) VALUES (?, ?, ?)",
                (time.time(), text, tag_str),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_notes(self, limit: int = 50, tag: str = "") -> list[dict[str, Any]]:
        with self._lock:
            if tag:
                rows = self._conn.execute(
                    "SELECT id, ts, text, tags FROM notes WHERE ',' || tags || ',' LIKE ? "
                    "ORDER BY ts DESC LIMIT ?",
                    (f"%,{tag.strip().lower()},%", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, ts, text, tags FROM notes ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def search_notes(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        like = f"%{query.strip()}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, text, tags FROM notes "
                "WHERE text LIKE ? COLLATE NOCASE OR tags LIKE ? COLLATE NOCASE "
                "ORDER BY ts DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # -- bulk load used once at startup ---------------------------------
    def recent_facts(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM facts ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        with self._lock:
            out = {}
            for table in ("messages", "facts", "notes"):
                out[table] = self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()["n"]
            return out
