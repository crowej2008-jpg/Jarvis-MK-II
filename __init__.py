"""JARVIS - a local-first, always-listening voice assistant.

Runs entirely on your machine: Ollama for the brain, faster-whisper for ears,
a local TTS engine for the mouth, SQLite for memory.
"""

from __future__ import annotations

from .env import harden_imports

# Must happen before any submodule imports a desktop automation library.
harden_imports()

__version__ = "1.0.0"
__all__ = ["__version__"]
