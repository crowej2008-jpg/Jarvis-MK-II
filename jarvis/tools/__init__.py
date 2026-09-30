"""Tool registry: the bridge between what the model can say and what it can do.

Every tool declares a JSON Schema for its arguments, which is handed to Ollama
verbatim. Handlers receive plain Python types and return either a value or a
`ToolError`, which is fed back to the model as a recoverable observation rather
than aborting the turn.
"""

from __future__ import annotations

import inspect
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

log = logging.getLogger("jarvis.tools")


class ToolError(Exception):
    """A failure the model should see and be able to correct."""


class ToolConfirmationRequired(Exception):
    """Raised when a tool needs explicit human approval before running."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    dangerous: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Values a handler accepts that the schema does not advertise, as
    # {property: [values]}. A 3B model asked for five canonical power actions
    # will still answer "reboot" or "log off", and the handler already maps
    # those. Listing all 36 in the schema instead would cost ~150 prompt tokens
    # and hand the model a 36-way choice instead of a 5-way one, so the prompt
    # keeps the short list and the accepted vocabulary is recorded here for
    # validation only. Never sent to the model.
    accepts: dict[str, list[Any]] = field(default_factory=dict)

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.validate(tool)
        self._tools[tool.name] = tool
        return tool

    @staticmethod
    def validate(tool: Tool) -> None:
        """Check the schema is one Ollama will accept.

        The Ollama client validates every tool schema with pydantic before the
        request is sent, and one malformed schema fails the entire call. That
        takes out every tool at once, not just the broken one, and only at
        request time. Catching it at registration means the mistake is
        reported against the tool that caused it, at import.
        """
        params = tool.parameters
        if not isinstance(params, dict):
            raise ValueError(f"tool {tool.name}: parameters must be a dict")
        if params.get("type") != "object":
            raise ValueError(
                f"tool {tool.name}: parameters.type must be 'object', got "
                f"{params.get('type')!r}. Wrap the arguments in "
                '{"type": "object", "properties": {...}}.'
            )
        props = params.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError(f"tool {tool.name}: properties must be a dict")
        for key, spec in props.items():
            if not isinstance(spec, dict) or "type" not in spec:
                raise ValueError(
                    f"tool {tool.name}: property {key!r} needs a 'type'"
                )
        required = params.get("required", [])
        if not isinstance(required, list):
            raise ValueError(f"tool {tool.name}: required must be a list")
        unknown = [r for r in required if r not in props]
        if unknown:
            raise ValueError(
                f"tool {tool.name}: required names not in properties: {unknown}"
            )

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        dangerous: bool = False,
        tags: Iterable[str] = (),
        accepts: dict[str, list[Any]] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                Tool(name, description, parameters, fn, dangerous, tuple(tags),
                     dict(accepts or {}))
            )
            return fn

        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, tags: Iterable[str] | None = None) -> list[dict[str, Any]]:
        selected = list(self._tools.values())
        if tags is not None:
            wanted = set(tags)
            selected = [t for t in selected if wanted & set(t.tags)]
        return [t.spec() for t in selected]

    def specs_for(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Specs for a specific set of names, skipping any that do not exist."""
        out = []
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                out.append(tool.spec())
        return out

    def matching(
        self, query: str = "", tags: Iterable[str] = ()
    ) -> list[dict[str, Any]]:
        """Find tools by keyword or tag, for on-demand discovery.

        Sending every tool to a small model on CPU is slow: the schemas are
        prepended to every request and prefill is not cheap. So the model starts
        with a small working set and calls `list_more_tools` when it needs
        something else.
        """
        wanted_tags = {t.lower() for t in tags if t}
        needle = (query or "").strip().lower()
        found: list[dict[str, Any]] = []
        for tool in self._tools.values():
            if tool.name == "list_more_tools":
                continue
            if wanted_tags and not (wanted_tags & {t.lower() for t in tool.tags}):
                continue
            if needle:
                haystack = f"{tool.name} {tool.description} {' '.join(tool.tags)}"
                if needle not in haystack.lower():
                    continue
            found.append(
                {
                    "name": tool.name,
                    "description": tool.description[:110],
                    "tags": list(tool.tags),
                }
            )
        found.sort(key=lambda d: d["name"])
        return found

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(
                f"no such tool: {name!r}. Available: {', '.join(self.names())}"
            )
        args = dict(arguments or {})
        args = self._coerce(tool, args)
        self._enforce_enums(tool, args)
        try:
            inspect.signature(tool.handler).bind(**args)
        except TypeError as exc:
            raise ToolError(f"bad arguments for {name}: {exc}") from exc
        return tool.handler(**args)

    def call_checked(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Run a tool, converting any failure into a readable string."""
        try:
            result = self.call(name, arguments)
        except ToolConfirmationRequired as exc:
            return {"error": "confirmation_required", "detail": str(exc)}
        except ToolError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            log.warning("tool %s raised\n%s", name, traceback.format_exc())
            return {
                "error": f"{exc.__class__.__name__}: {exc}",
                "type": type(exc).__name__,
            }
        if result is None:
            return {"ok": True}
        return result

    @staticmethod
    def _normalise_enum_value(value: Any) -> Any:
        """Compare values the way a model means them, not byte for byte.

        It answers 'Left', ' left ' and 'LEFT' for the same button, and
        'shut  down' for 'shut down'. Rewriting to the canonical member matters
        more than the check: handlers index tables with the exact string.
        """
        if isinstance(value, str):
            return " ".join(value.split()).casefold()
        return value

    @classmethod
    def _match_enum(cls, value: Any, allowed: list[Any]) -> Any | None:
        """The canonical member equal to value, or None if there is none."""
        wanted = cls._normalise_enum_value(value)
        for member in allowed:
            if cls._normalise_enum_value(member) == wanted:
                return member
        return None

    @classmethod
    def _enforce_enums(cls, tool: Tool, args: dict[str, Any]) -> None:
        """Reject a value the schema never offered, before the handler runs.

        The schemas document enums but nothing checked them, so a model
        answering button='leftt' or action='shut' reached the handler
        unchanged. mouse.click then handed the string to pyautogui, which
        raised a bare ValueError *after* the authorisation check and after a
        'before' screenshot had been written to visual memory. A bad enum is a
        model mistake, so it should read as a clear, correctable error at the
        boundary rather than a confusing failure from inside a library.
        """
        accepts = tool.accepts or {}
        cls._walk_enums(tool, tool.parameters, args, "", accepts)

    @classmethod
    def _walk_enums(
        cls,
        tool: Tool,
        schema: dict[str, Any],
        value: Any,
        path: str,
        accepts: dict[str, list[Any]],
    ) -> Any:
        if not isinstance(schema, dict) or value is None:
            return value

        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            # `accepts` widens the vocabulary for this property only, and only
            # when the schema already declared an enum to widen.
            allowed = enum + list(accepts.get(path, []))
            match = cls._match_enum(value, allowed)
            if match is None:
                shown = ", ".join(repr(v) for v in enum[:12])
                if len(enum) > 12:
                    shown += f", ... ({len(enum)} values)"
                where = path or "argument"
                raise ToolError(
                    f"{tool.name}: {where}={value!r} is not one of the allowed "
                    f"values: {shown}. Call it again with one of those."
                )
            return match

        kind = schema.get("type")
        if kind == "object" and isinstance(value, dict):
            props = schema.get("properties", {})
            for key, sub in value.items():
                if key in props:
                    value[key] = cls._walk_enums(
                        tool, props[key], sub, f"{path}.{key}" if path else key,
                        accepts,
                    )
            return value
        if kind == "array" and isinstance(value, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for i, item in enumerate(value):
                    value[i] = cls._walk_enums(
                        tool, item_schema, item, f"{path}[{i}]", accepts,
                    )
            return value
        return value

    @staticmethod
    def _coerce(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
        """Nudge loose model output (string numbers, string lists) into shape."""
        props = tool.parameters.get("properties", {})
        out: dict[str, Any] = {}
        for key, value in args.items():
            spec = props.get(key)
            if spec is None or value is None:
                out[key] = value
                continue
            want = spec.get("type")
            try:
                if want == "number":
                    out[key] = value if isinstance(value, (int, float)) else float(value)
                elif want == "integer":
                    out[key] = int(value)
                elif want == "boolean":
                    if isinstance(value, str):
                        out[key] = value.strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        out[key] = bool(value)
                elif want == "array" and isinstance(value, str):
                    out[key] = ToolRegistry._parse_list(value)
                elif want == "string" and not isinstance(value, str):
                    # A model asked to type "42" or "true" may send the bare
                    # number or bool, and pyautogui.write() would choke on it.
                    out[key] = str(value)
                else:
                    out[key] = value
            except (TypeError, ValueError) as exc:
                raise ToolError(f"could not read {key}={value!r} as {want}: {exc}") from exc
        return out

    @staticmethod
    def _parse_list(value: str) -> list[str]:
        """Read a list a model sent as a string.

        JSON first, because a small model asked for a list of keystrokes often
        answers '["ctrl", "c"]' and splitting that on commas yields the literal
        fragments '["ctrl"' and '"c"]', which then fail inside the handler with
        a far less obvious error. A plain comma-separated string is still
        accepted, since that is what most callers actually mean.
        """
        import json

        text = value.strip()
        if text[:1] in "[(":
            try:
                parsed = json.loads(text.replace("(", "[").replace(")", "]"))
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return [str(p) for p in parsed]
        return [p.strip().strip("'\"") for p in text.split(",") if p.strip().strip("'\"")]


# The registry every tool module decorates onto.
registry = ToolRegistry()


@registry.add(
    "list_more_tools",
    "Search the tools you have not been given yet, by keyword or category, and "
    "load them. You start with only the common tools to keep replies fast, so "
    "call this when what you need is not in your current list. After this they "
    "stay available. Categories: sight, screen, act, mouse, keyboard, memory, "
    "windows, system, files, web, home, mqtt, audio, clipboard, power, shell, "
    "input, procedure, habits.",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Words describing the capability, e.g. 'weather' "
                "or 'window' or 'bluetooth'.",
            },
            "tag": {
                "type": "string",
                "description": "Or a category, e.g. 'files' or 'home'.",
            },
        },
    },
    tags=("discovery",),
)
def list_more_tools(query: str = "", tag: str = "") -> dict[str, Any]:
    matches = registry.matching(query=query, tags=(tag,) if tag else ())
    if not matches and not query and not tag:
        return {
            "found": 0,
            "hint": "Pass a query or a tag. Call it with no arguments to see "
            "every category.",
            "categories": sorted({t for tool in registry._tools.values() for t in tool.tags}),
        }
    return {
        "found": len(matches),
        # The names are what the brain loads; the descriptions let the model
        # choose between them.
        "tools": [m["name"] for m in matches],
        "detail": matches[:40],
        "note": "These are now loaded and you can call them directly.",
    }


TOOL_MODULES = (
    "system_tools",
    "screen_tools",
    "memory_tools",
    "iot_tools",
    "web_tools",
    "sight_tools",
    "mouse_tools",
    "game_tools",
)


def load_all() -> ToolRegistry:
    """Import every tool module so its handlers register themselves.

    Importing the submodules here (rather than at the top of this file) avoids
    a circular import: the submodules need `registry` to already exist.
    """
    from importlib import import_module

    for name in TOOL_MODULES:
        import_module(f"{__name__}.{name}")
    return registry


__all__ = [
    "Tool",
    "ToolError",
    "ToolConfirmationRequired",
    "ToolRegistry",
    "registry",
    "load_all",
]
