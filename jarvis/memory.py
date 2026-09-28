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
