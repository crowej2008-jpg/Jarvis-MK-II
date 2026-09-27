"""Memory tools: the model's way to read and write long-term state.

The registry holds handler functions, but handlers cannot close over runtime
state at import time, so the active Memory instance is attached here.
"""

from __future__ import annotations

import time
from typing import Any

from . import ToolError, registry

_memory = None


def bind(memory) -> None:
    global _memory
    _memory = memory


def _mem():
    if _memory is None:
        raise ToolError("memory is not available in this session")
    return _memory


@registry.add(
    "remember",
    "Save something so you know it next time: a fact about the user or their "
    "setup, a preference, or a reminder to do something later. Use short "
    "snake_case keys, e.g. key='user_name' or key='remind_dentist'. Use this for "
    "anything the user asks you to keep in mind, not only facts.",
    {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short identifier for the fact."},
            "value": {"type": "string", "description": "The fact itself."},
        },
        "required": ["key", "value"],
    },
    tags=("memory",),
)
def remember(key: str, value: str) -> dict[str, Any]:
    if not key.strip():
        raise ToolError("key must not be empty")
    _mem().remember(key, value)
    return {"remembered": key.strip().lower(), "value": value}


@registry.add(
    "recall",
    "Look up a single fact previously saved with remember.",
    {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    tags=("memory",),
)
def recall(key: str) -> dict[str, Any]:
    value = _mem().recall(key)
    if value is None:
        return {"key": key.strip().lower(), "found": False}
    return {"key": key.strip().lower(), "found": True, "value": value}


@registry.add(
    "forget",
    "Delete a fact saved with remember.",
    {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    tags=("memory",),
)
def forget(key: str) -> dict[str, Any]:
    removed = _mem().forget(key)
    return {"key": key.strip().lower(), "deleted": removed}


@registry.add(
    "search_memory",
    "Search past conversation history for anything relevant to a topic. "
    "Use this when the user refers to something they mentioned earlier.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
    tags=("memory",),
)
def search_memory(query: str, limit: int = 12) -> dict[str, Any]:
    rows = _mem().search_messages(query, int(limit))
    results = [
        {
            "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"])),
            "role": r["role"],
            "text": (r.get("snip") or r["content"])[:400],
        }
        for r in rows
    ]
    return {"query": query, "count": len(results), "results": results}


@registry.add(
    "take_note",
    "Write a note down for later, optionally tagged so it can be found by topic.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The note."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional lowercase topic tags.",
            },
        },
        "required": ["text"],
    },
    tags=("memory",),
)
def take_note(text: str, tags: list[str] | None = None) -> dict[str, Any]:
    note_id = _mem().add_note(text, tags or [])
    return {"note_id": note_id, "tags": tags or []}


@registry.add(
    "search_notes",
    "Search saved notes by keyword or tag.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword, or empty for all notes."},
            "tag": {"type": "string", "description": "Filter to one tag."},
            "limit": {"type": "integer"},
        },
    },
    tags=("memory",),
)
def search_notes(query: str = "", tag: str = "", limit: int = 20) -> dict[str, Any]:
    mem = _mem()
    rows = (
        mem.search_notes(query, int(limit))
        if query
        else mem.list_notes(int(limit), tag)
    )
    return {
        "count": len(rows),
        "notes": [
            {
                "id": r["id"],
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"])),
                "text": r["text"],
                "tags": r["tags"],
            }
            for r in rows
        ],
    }


@registry.add(
    "delete_note",
    "Delete a note by its numeric id.",
    {
        "type": "object",
        "properties": {"note_id": {"type": "integer"}},
        "required": ["note_id"],
    },
    tags=("memory",),
)
def delete_note(note_id: int) -> dict[str, Any]:
    return {"note_id": note_id, "deleted": _mem().delete_note(int(note_id))}


@registry.add(
    "list_facts",
    "List every durable fact you have saved.",
    {"type": "object", "properties": {}},
    tags=("memory",),
)
def list_facts() -> dict[str, Any]:
    facts = _mem().all_facts()
    return {"count": len(facts), "facts": [{"key": f["key"], "value": f["value"]} for f in facts]}


@registry.add(
    "forget_everything",
    "Erase the entire conversation history. Facts and notes are kept. "
    "Only call this when the user explicitly asks you to forget everything.",
    {"type": "object", "properties": {}},
    dangerous=True,
    tags=("memory",),
)
def forget_everything() -> dict[str, Any]:
    from . import system_tools

    system_tools._approve(
        "forget_everything", "You asked me to erase our whole conversation history."
    )
    removed = _mem().clear_messages()
    return {"messages_deleted": removed}
