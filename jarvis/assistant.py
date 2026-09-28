"""The assistant core: one place where a request becomes tool calls and a reply.

Both front ends (voice and text) go through `Assistant.respond`, so they behave
identically. Voice-specific concerns like wake words and microphone level live
in `jarvis.voice`.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .autonomy import Autonomy
from .brain import Brain, Reply, speakable
from .config import Config
from .embeddings import make_embedder
from .memory import Memory
from .locator import Locator
from .perception import Perception
from .vision_memory import VisionMemory
from .vlm import Vision
from .tools import iot_tools, load_all, memory_tools, mouse_tools, sight_tools, system_tools

log = logging.getLogger("jarvis.assistant")

# Things the user can say that should never reach the model.
INTERRUPTS = {
    "stop", "cancel", "never mind", "nevermind", "shut up", "be quiet",
    "silence", "quiet", "that's enough", "thats enough",
}
SLEEP_WORDS = {
    "go to sleep", "sleep now", "goodbye", "good bye", "bye", "power down",
    "standby", "that's all", "thats all",
}


@dataclass
class Outcome:
    reply: Reply
    spoken: str = ""
    action: str = "reply"  # reply | sleep | interrupt
    images: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.reply.text.strip())


class Assistant:
    def __init__(
        self,
        cfg: Config,
        memory: Memory,
        brain: Brain,
        speaker: Any = None,
        on_status: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.brain = brain
        self.speaker = speaker
        self.on_status = on_status or (lambda _m: None)
        self.asleep = False
        self._approver: Callable[[str, str], bool] | None = None

        registry = load_all()
        memory_tools.bind(memory)
        iot_tools.bind_config(cfg)
        system_tools.set_approver(self._confirm)
        self._registry = registry

        # The seeing stack: perception, visual memory, the vision model, the
        # locator, and the autonomy gate that sits in front of all actuation.
        self.perception = Perception(thumb_width=cfg.vision_thumb_width)
        self.vision_memory = VisionMemory(
            cfg.resolved_db_path.with_name(cfg.resolved_db_path.stem + "_vision.db"),
            embedder=make_embedder(cfg),
        )
        self.vision = Vision(cfg)
        self.autonomy = Autonomy(cfg, memory)
        self.locator = Locator(cfg, self.vision_memory, self.perception, self.vision)

        sight_tools.bind(cfg, self.vision_memory, memory, self.autonomy)
        mouse_tools.bind(cfg, self.vision_memory, self.autonomy, self.locator)
        system_tools.set_approver(self._confirm)
        # With confirmation switched off, decisions are auto-granted. Saying so
        # up front keeps them distinguishable from a person answering yes.
        system_tools.set_auto_approve(not cfg.confirm_destructive)

    # -- tool approval ---------------------------------------------------
    def set_approver(self, fn: Callable[[str, str], bool] | None) -> None:
        """Front ends install a real confirmation prompt here."""
        self._approver = fn
        system_tools.set_approver(self._confirm)

    def _confirm(self, tool_name: str, detail: str) -> bool:
        if not self.cfg.confirm_destructive:
            log.info("auto-approved %s: %s", tool_name, detail)
            return True
        if self._approver is None:
            # No prompt available: refuse rather than guess.
            log.warning("blocked %s, no confirmation handler", tool_name)
            return False
        return bool(self._approver(tool_name, detail))
    @property
    def tool_count(self) -> int:
        return len(self._registry)

    def tool_names(self) -> list[str]:
        return self._registry.names()

    # -- the main path ---------------------------------------------------
    def respond(
        self,
        text: str,
        images: Iterable[str] = (),
        on_token: Callable[[str], None] | None = None,
    ) -> Outcome:
        text = (text or "").strip()
        if not text:
            return Outcome(Reply(text=""), action="interrupt")

        lowered = re.sub(r"[^\w\s']", " ", text.lower()).strip()
        if lowered in INTERRUPTS:
            return Outcome(Reply(text=""), action="interrupt")
        if any(lowered == s or lowered.endswith(" " + s) for s in SLEEP_WORDS):
            self.asleep = True
            return Outcome(
                Reply(text="Going dormant. Say hey jarvis to wake me."),
                spoken="Going dormant.",
                action="sleep",
            )

        if self.asleep:
            self.asleep = False

        self.on_status("thinking")
        started = time.time()
        try:
            reply = self.brain.think(
                text, images=images, on_token=on_token, execute_tools=True
            )
        except Exception as exc:  # noqa: BLE001 - never die on one bad turn
            log.exception("brain failed")
            # Say something either way. A text-mode user sees reply.text, a
            # voice user hears spoken; leaving both empty just looks broken.
            message = self._explain_failure(exc)
            return Outcome(
                Reply(text=message, error=str(exc)),
                spoken=message,
                action="error",
            )

        spoken = speakable(reply.text)
        log.info(
            "turn ok in %.2fs (rounds=%d, tools=%d, chars=%d)",
            time.time() - started,
            reply.rounds,
            len(reply.tool_calls),
            len(reply.text),
        )
        self.on_status("idle")
        return Outcome(reply=reply, spoken=spoken, action="reply")

    # -- helpers for front ends -----------------------------------------
    @staticmethod
    def _explain_failure(exc: Exception) -> str:
        """Turn a brain failure into something a person can act on."""
        name = type(exc).__name__
        text = str(exc).lower()
        if "timeout" in name.lower() or "timed out" in text:
            return (
                "Ollama did not answer in time. It is probably still busy, or "
                "the request was too large. Try again, or ask me something shorter."
            )
        if "connection" in name.lower() or "connect" in text:
            return (
                "I could not reach Ollama. Check that it is running, then try again."
            )
        if "model" in text and "not found" in text:
            return f"The model is not installed. {exc}"
        return f"Something went wrong in my brain: {exc}"

    def close(self) -> None:
        """Release the visual memory database.

        Only this one matters: it is a second SQLite connection, and leaving it
        open on the way out can leave a stale WAL file next to the database.
        """
        try:
            self.vision_memory.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("vision memory close failed: %s", exc)

    def say(self, text: str) -> None:
        if self.speaker is not None and text.strip():
            self.speaker.speak(speakable(text))

    def stop_speaking(self) -> None:
        if self.speaker is not None:
            self.speaker.stop()

    def status_line(self) -> str:
        stats = self.memory.stats()
        return (
            f"model={self.brain.model} | tools={self.tool_count} | "
            f"turns={stats['messages'] // 2} | facts={stats['facts']} | "
            f"notes={stats['notes']}"
        )

    def vision_status(self) -> str:
        try:
            stats = self.vision_memory.stats()
        except Exception:  # noqa: BLE001
            return "vision memory unavailable"
        return (
            f"screens={stats['observations']} controls={stats['ui_map']} "
            f"procedures={stats['procedures']} actions={stats['actions']} "
            f"vision={self.vision.model if self.vision.available() else 'off'}"
        )

    def maintain(self) -> dict[str, Any]:
        """Prune old visual memory. Cheap, so it runs at startup."""
        try:
            removed = self.vision_memory.prune(
                self.cfg.vision_retention_days, self.cfg.vision_max_observations
            )
            if removed:
                log.info("pruned %s old observations", removed)
            return {"pruned": removed}
        except Exception as exc:  # noqa: BLE001
            log.warning("vision memory prune failed: %s", exc)
            return {"error": str(exc)}

    def describe_tools(self) -> str:
        return ", ".join(self.tool_names())
