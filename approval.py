"""Provenance and expiry for destructive-action approval.

The approval used to be a bare boolean: the front end returned True or False,
JARVIS acted, and nothing recorded what had actually been agreed to. That is
enough to run an action but not enough to answer afterwards "what did I approve,
when, and was that the same thing it went on to do?"

Three things are wrong with a bare boolean, and this module addresses each:

* **No provenance.** A decision is now a :class:`Decision` recording the tool,
  the exact wording shown to the human, the outcome, how it was reached, and
  when. ``history()`` is the audit trail, and it distinguishes a person saying
  yes from ``--confirmation off`` saying yes on their behalf.

* **Nothing binds the answer to the action.** The wording is fingerprinted, so a
  decision can be shown to have been about *this* tool and *these* details
  rather than some other destructive call that happened to be approved earlier
  in the session.

* **No expiry.** A front end that wants to remember an answer gets
  :meth:`ConfirmationLog.grant_token`, which is the fingerprint plus a
  deadline. :meth:`check_token` refuses it once stale, so "yes, don't ask again"
  cannot silently become "yes, forever". Without a token every action asks
  again, which is the default and the only behaviour that existed before.

Reuse within a window is available but not automatic. Skipping a prompt is a
convenience, and a convenience that is on by default is a convenience that
eventually skips the one prompt that mattered.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

# How a decision was reached. Recorded so an audit can tell a human answer from
# an automatic one.
PROMPT = "prompt"          # a person was asked and answered
AUTO = "auto"              # confirm_destructive was off
NO_HANDLER = "no-handler"  # nothing was wired up to ask, so it refused
TOKEN = "token"            # a still-valid token stood in for the prompt
CACHED = "cached"          # an identical, unexpired, granted decision


def fingerprint(tool: str, detail: str) -> str:
    """A short stable id for one specific tool-plus-detail pair."""
    payload = f"{tool}\x00{detail}".encode("utf-8", "surrogatepass")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class Decision:
    """One approval outcome, kept for the audit trail."""

    tool: str
    detail: str
    granted: bool
    source: str
    at: float
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint",
                               fingerprint(self.tool, self.detail))

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.at)

    def as_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "detail": self.detail,
            "granted": self.granted,
            "source": self.source,
            "at": self.at,
            "fingerprint": self.fingerprint,
        }


class ConfirmationLog:
    """Records every approval decision and validates reusable ones."""

    def __init__(self, ttl: float = 120.0, limit: int = 200) -> None:
        self.ttl = float(ttl)
        self.limit = max(1, int(limit))
        # Set by system_tools.set_auto_approve. When true, an approval is
        # granted without asking, and decisions are tagged AUTO so the trail
        # does not imply a person said yes.
        self.auto = False
        self._decisions: list[Decision] = []
        # fingerprint -> (granted_at, fingerprint). Only ever holds grants, and
        # only for as long as ttl.
        self._grants: dict[str, tuple[float, str]] = {}

    # -- deciding ---------------------------------------------------------

    def decide(
        self,
        tool: str,
        detail: str,
        asker: Callable[[], bool],
        *,
        source: str = PROMPT,
    ) -> Decision:
        """Ask (or not) and record the outcome. Returns the Decision."""
        try:
            granted = bool(asker())
        except Exception as exc:  # noqa: BLE001 - a broken prompt is a refusal
            self._record(Decision(tool, detail, False, f"error:{exc.__class__.__name__}"))
            raise
        decision = Decision(tool, detail, granted, source, time.time())
        self._record(decision)
        return decision

    def record_refusal(self, tool: str, detail: str, source: str) -> Decision:
        """Record a decision that was never put to anybody."""
        decision = Decision(tool, detail, False, source, time.time())
        self._record(decision)
        return decision

    def reuse(self, tool: str, detail: str) -> Decision | None:
        """A previous grant for the identical action that has not expired.

        Returns a Decision marked ``CACHED`` or None. Callers must still decide
        whether skipping the prompt is acceptable for what they are about to do.
        """
        self._expire()
        key = fingerprint(tool, detail)
        held = self._grants.get(key)
        if held is None:
            return None
        granted_at, fp = held
        decision = Decision(tool, detail, True, CACHED, granted_at, fp)
        self._record(decision)
        return decision

    # -- tokens -----------------------------------------------------------

    def grant_token(self, tool: str, detail: str,
                    ttl: float | None = None) -> str:
        """A time-limited proof that this exact action was approved.

        Returned to a front end that wants to honour "don't ask me again for
        this" without holding an unbounded boolean. Encodes no secret: it is a
        fingerprint and a deadline, so it can be logged and audited.
        """
        window = self.ttl if ttl is None else float(ttl)
        return f"{fingerprint(tool, detail)}:{time.time() + window:.3f}"

    def check_token(self, token: str, tool: str, detail: str) -> bool:
        """True only if the token is for this action and has not expired."""
        if not token or ":" not in token:
            return False
        given_fp, _, expires_raw = token.partition(":")
        try:
            expires = float(expires_raw)
        except ValueError:
            return False
        if time.time() > expires:
            return False
        return hmac.compare_digest(given_fp, fingerprint(tool, detail))

    # -- reading ----------------------------------------------------------

    def history(self, tool: str | None = None,
                limit: int | None = None) -> list[Decision]:
        """Decisions newest first, optionally for one tool."""
        items: Iterable[Decision] = reversed(self._decisions)
        if tool is not None:
            items = [d for d in items if d.tool == tool]
        out = list(items)
        return out[:limit] if limit else out

    def last(self, tool: str | None = None) -> Decision | None:
        found = self.history(tool, limit=1)
        return found[0] if found else None

    def granted_within(self, tool: str, detail: str,
                       ttl: float | None = None) -> bool:
        self._expire()
        held = self._grants.get(fingerprint(tool, detail))
        if held is None:
            return False
        return (time.time() - held[0]) <= (self.ttl if ttl is None else ttl)

    def forget(self) -> None:
        """Drop the reusable grants. The audit trail is kept."""
        self._grants.clear()

    def clear(self) -> None:
        self._decisions.clear()
        self._grants.clear()

    def __len__(self) -> int:
        return len(self._decisions)

    # -- internals --------------------------------------------------------

    def _record(self, decision: Decision) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > self.limit:
            del self._decisions[: len(self._decisions) - self.limit]
        if decision.granted:
            self._grants[decision.fingerprint] = (decision.at,
                                                   decision.fingerprint)
        self._expire()

    def _expire(self) -> None:
        if self.ttl <= 0:
            self._grants.clear()
            return
        cutoff = time.time() - self.ttl
        for key in [k for k, (when, _) in self._grants.items() if when < cutoff]:
            del self._grants[key]


# The process-wide log. Tools reach it through system_tools._approve, so every
# destructive action in the codebase lands here without its own bookkeeping.
LOG = ConfirmationLog()
