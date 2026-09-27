"""Autonomy: deciding whether JARVIS may act without asking.

The rule set is deliberately asymmetric. It is easy to earn trust for an app
(approve it once and it is remembered) and deliberately hard to lose it: the
denylist and the credential checks are not overridable by the allowlist.

Two things are checked before every click, keystroke, or typed string:

  1. Is this app trusted to act in, at all?
  2. Does what is on screen look like a credential or payment context?

The second check exists because an allowlisted browser can still be sitting on
a bank's login page. Trust in Chrome is not trust in a password field.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Titles that indicate a context where input is dangerous regardless of app.
DANGEROUS_TITLE = re.compile(
    r"(sign[ -]?in|log[ -]?in|log ?on|login|password|passcode|passphrase|credential|"
    r"unlock|authenticat|2fa|two[ -]factor|otp|verification code|security code|"
    r"\bpin\b|"
    r"payment|checkout|billing|card number|banking|transfer|wire|"
    r"account recovery|reset password|security question)",
    re.I,
)

# Words near a text field that mean the field wants a secret.
CREDENTIAL_CONTEXT = re.compile(
    r"(password|passcode|pin\b|secret|security code|verification code|otp|"
    r"recovery code|private key|api key|token)\b",
    re.I,
)

# Window classes Windows uses for privileged UI that must never be scripted.
DANGEROUS_CLASSES = frozenset(
    {
        "Credential Dialog Xaml Host",
        "CredentialUIBroker",
        "Logonui",
        "Consent",
        "SecurityHealthCheck",
    }
)


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    needs_approval: bool = False
    blocked: bool = False

    def __bool__(self) -> bool:
        return self.allowed


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def app_token(app: str) -> str:
    """Compare on the bare executable name, ignoring path and case."""
    name = _norm(app).replace("/", "\\")
    return name.rsplit("\\", 1)[-1] or name


class Autonomy:
    def __init__(self, cfg, memory=None):
        self.cfg = cfg
        self.memory = memory
        self.allow: set[str] = {app_token(a) for a in (cfg.autonomy_allowlist or [])}
        self.deny: set[str] = {app_token(d) for d in (cfg.autonomy_denylist or [])}
        self._granted: set[str] = set()
        self._loaded = False

    # -- persistent grants ----------------------------------------------
    def _load_grants(self) -> None:
        if self._loaded or self.memory is None:
            return
        self._loaded = True
        for fact in self.memory.all_facts():
            key = fact["key"]
            if key.startswith("autonomy_grant:"):
                token = key.split(":", 1)[1]
                if token:
                    self._granted.add(token)
            elif key.startswith("autonomy_deny:"):
                # A /deny outranks the config allowlist, so it has to survive
                # a restart; self.allow is rebuilt from config every launch.
                token = key.split(":", 1)[1]
                if token:
                    self.deny.add(token)
                    self.allow.discard(token)

    def _save_grant(self, token: str) -> None:
        if self.memory is not None:
            self.memory.remember(f"autonomy_grant:{token}", "granted")
            # Approving one app must not silently un-deny another.
            self.memory.forget(f"autonomy_deny:{token}")

    def grant(self, app: str) -> str:
        token = app_token(app)
        self._load_grants()
        if token in self.deny:
            # Denials are sticky. Refuse here rather than trusting each caller
            # to remember, so a new tool cannot bypass the gate.
            return token
        self._granted.add(token)
        self._save_grant(token)
        return token

    def trusted(self, app: str) -> bool:
        self._load_grants()
        token = app_token(app)
        if token in self.deny:
            return False
        return token in self.allow or token in self._granted

    def revoke(self, app: str) -> bool:
        token = app_token(app)
        self._load_grants()
        was = token in self._granted or token in self.allow
        self._granted.discard(token)
        self.allow.discard(token)
        # Persistent on purpose: /deny that reverts on the next launch would
        # give false confidence about an app handling credentials.
        self.deny.add(token)
        if self.memory is not None:
            self.memory.forget(f"autonomy_grant:{token}")
            self.memory.remember(f"autonomy_deny:{token}", "denied by you")
        return was

    def known_apps(self) -> dict[str, str]:
        self._load_grants()
        return {
            **{a: "built-in allowlist" for a in sorted(self.allow)},
            **{g: "approved by you" for g in sorted(self._granted)},
            **{d: "DENIED" for d in sorted(self.deny)},
        }

    # -- the decision ----------------------------------------------------
    def evaluate(
        self,
        app: str = "",
        window_title: str = "",
        elements: Sequence[Any] | None = None,
        action: str = "click",
    ) -> Decision:
        """May JARVIS perform `action` in this window right now?"""
        token = app_token(app)

        if token and token in self.deny:
            return Decision(
                False,
                f"{token} is on the denylist; it handles credentials or privileged UI",
                blocked=True,
            )

        if DANGEROUS_CLASSES.intersection({window_title} if window_title else set()):
            return Decision(
                False, f"{window_title} is privileged Windows UI", blocked=True
            )

        if window_title and DANGEROUS_TITLE.search(window_title):
            return Decision(
                False,
                f"the window title '{window_title}' suggests a sign-in, payment, "
                "or account-recovery screen",
                blocked=True,
            )

        # Typing next to a credential label is the sharpest edge here.
        if (
            getattr(self.cfg, "refuse_typing_near_credentials", True)
            and action in {"type", "hotkey", "key"}
        ):
            if self._credential_field(elements or []):
                return Decision(
                    False,
                    "there is a password or security-code field on screen; "
                    "I will not type into it",
                    blocked=True,
                )

        if self.trusted(token or app):
            return Decision(True, f"{token or 'this app'} is trusted")

        return Decision(
            False,
            f"{token or 'this app'} is not on the allowlist, so I need your OK first",
            needs_approval=True,
        )

    @staticmethod
    def _credential_field(elements: Iterable[Any]) -> bool:
        """True when a credential-looking label is visible anywhere on screen.

        This is deliberately blunt. The alternative is trying to prove the
        focused field is the credential field, but focus and field identity are
        not observable from a screenshot, and guessing wrong means typing a
        password somewhere visible. Over-blocking a screen that merely mentions
        "password" is a much cheaper mistake than that.
        """
        for element in elements:
            text = element.get("text") if isinstance(element, dict) else getattr(element, "text", "")
            if text and CREDENTIAL_CONTEXT.search(str(text)):
                return True
        return False

    # -- helpers used by the actuation tools -----------------------------
    def focus(self) -> tuple[str, str, list[dict[str, Any]]]:
        """(window_title, app_name, elements_on_screen) for the foreground window."""
        from .perception import Perception

        try:
            observation = Perception(ocr=True).observe(force=True)
        except Exception:  # noqa: BLE001
            return "", "", []
        elements = [e.as_dict() for e in observation.elements]
        return observation.window_title, observation.window_app, elements

    def can_act(
        self, action: str, elements: Sequence[Any] | None = None
    ) -> tuple[Decision, str, str]:
        """Evaluate the current screen for `action`.

        Returns (decision, window_title, app).
        """
        title, app, seen = self.focus()
        if elements is None:
            elements = seen
        return self.evaluate(app=app, window_title=title, elements=elements, action=action), title, app
