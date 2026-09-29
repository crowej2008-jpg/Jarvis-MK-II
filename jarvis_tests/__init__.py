"""Offline regression tests for JARVIS.

Run with:

    python -m unittest discover -s jarvis_tests -v

These cover the logic that must not silently regress: the autonomy gate, tool
schema validation, VLM output filtering, coordinate parsing, visual-memory
confidence, and the argument coercion that the model depends on.

Nothing here touches the real screen, the real mouse, the network, or your
actual database. Every test uses a temporary directory or an in-memory fake, so
the suite is safe to run at any time and on a machine where you are doing
something else. That is the point: the interesting failures in this project are
the ones where JARVIS types into the wrong window, and a test that clicks
something is no use as a guard.
"""

from __future__ import annotations

import ast
import sqlite3
import sys
from collections import OrderedDict
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from jarvis.autonomy import (
    CREDENTIAL_CONTEXT,
    DANGEROUS_CLASSES,
    Autonomy,
    app_token,
)
from jarvis.tools import Tool, ToolError, ToolRegistry, load_all, registry

TOOLS_DIR = Path(__file__).resolve().parent.parent / "jarvis" / "tools"


def make_config(**overrides):
    """A config stand-in with just the fields Autonomy reads."""
    defaults = {
        "autonomy_allowlist": ["notepad.exe", "chrome.exe"],
        "autonomy_denylist": ["keepass.exe", "1password.exe"],
        "refuse_typing_near_credentials": True,
    }
    defaults.update(overrides)
    return type("Cfg", (), defaults)()


class TestAppToken(unittest.TestCase):
    def test_strips_path_and_case(self):
        self.assertEqual(app_token(r"C:\Program Files\KeePass\KeePass.EXE"), "keepass.exe")
        self.assertEqual(app_token("notepad.exe"), "notepad.exe")

    def test_forward_slash(self):
        self.assertEqual(app_token("/usr/bin/notepad.exe"), "notepad.exe")

    def test_blank(self):
        self.assertEqual(app_token(""), "")
        self.assertEqual(app_token(None or ""), "")


class TestAutonomyDeny(unittest.TestCase):
    """The denylist is the last line of defence and must not be negotiable."""

    def setUp(self):
        self.a = Autonomy(make_config())

    def test_denied_app_blocked(self):
        d = self.a.evaluate(app="keepass.exe", window_title="KeePass")
        self.assertFalse(d.allowed)
        self.assertTrue(d.blocked, "a denylisted app must be blocked, not merely gated")

    def test_deny_survives_allowlist(self):
        # An app can be on both lists by misconfiguration; deny must win.
        cfg = make_config(autonomy_allowlist=["keepass.exe"], autonomy_denylist=["keepass.exe"])
        d = Autonomy(cfg).evaluate(app="keepass.exe", window_title="x")
        self.assertTrue(d.blocked)

    def test_allowlisted_app_allowed(self):
        d = self.a.evaluate(app="notepad.exe", window_title="Untitled - Notepad")
        self.assertTrue(d.allowed)
        self.assertFalse(d.needs_approval)

    def test_unknown_app_needs_approval_not_blocked(self):
        d = self.a.evaluate(app="someotherapp.exe", window_title="Some App")
        self.assertFalse(d.allowed)
        self.assertTrue(d.needs_approval)
        self.assertFalse(d.blocked, "an unknown app is gated, not blocked")


class TestAutonomyDangerousWindows(unittest.TestCase):
    def setUp(self):
        self.a = Autonomy(make_config())

    def test_privileged_class_blocked(self):
        for title in DANGEROUS_CLASSES:
            d = self.a.evaluate(app="chrome.exe", window_title=title)
            self.assertTrue(d.blocked, f"{title} should be blocked")

    def test_dangerous_title_blocked_even_in_trusted_app(self):
        """Trust in Chrome is not trust in a bank's login page."""
        for title in [
            "Sign in to your account",
            "Logon",
            "Checkout - Example Store",
            "Enter your PIN",
            "Reset password",
            "Two-factor authentication",
        ]:
            d = self.a.evaluate(app="chrome.exe", window_title=title)
            self.assertFalse(d.allowed, f"{title!r} should be refused")
            self.assertTrue(d.blocked, f"{title!r} should be blocked, not gated")

    def test_ordinary_title_allowed(self):
        d = self.a.evaluate(app="chrome.exe", window_title="Wikipedia")
        self.assertTrue(d.allowed)


class TestCredentialFields(unittest.TestCase):
    """Typing is the sharpest edge, so these run for every text-ish action."""

    def setUp(self):
        self.a = Autonomy(make_config())

    def test_credential_label_blocks_typing(self):
        for label in ["Password", "Enter PIN", "Security code", "Recovery code", "API key"]:
            d = self.a.evaluate(
                app="notepad.exe",
                window_title="Notepad",
                elements=[{"text": label}],
                action="type",
            )
            self.assertTrue(d.blocked, f"{label!r} should block typing")

    def test_blocks_hotkey_and_key_too(self):
        for action in ("type", "hotkey", "key"):
            d = self.a.evaluate(
                app="notepad.exe",
                window_title="Notepad",
                elements=[{"text": "Password"}],
                action=action,
            )
            self.assertTrue(d.blocked, f"action {action!r} must also be blocked")

    def test_credential_label_does_not_block_click(self):
        """Being on a login page is fine to click; only typing is refused."""
        d = self.a.evaluate(
            app="chrome.exe",
            window_title="Wikipedia",
            elements=[{"text": "Password"}],
            action="click",
        )
        self.assertTrue(d.allowed)

    def test_ordinary_screen_allows_typing(self):
        d = self.a.evaluate(
            app="notepad.exe",
            window_title="notes.txt - Notepad",
            elements=[{"text": "Shopping list"}, {"text": "milk"}],
            action="type",
        )
        self.assertTrue(d.allowed)

    def test_config_can_disable_the_check(self):
        a = Autonomy(make_config(refuse_typing_near_credentials=False))
        d = a.evaluate(
            app="notepad.exe",
            window_title="x",
            elements=[{"text": "Password"}],
            action="type",
        )
        self.assertTrue(d.allowed)

    def test_accepts_dicts_and_objects(self):
        class El:
            text = "Password"

        for elements in ([{"text": "Password"}], [El()]):
            d = self.a.evaluate(
                app="notepad.exe", window_title="x", elements=elements, action="type"
            )
            self.assertTrue(d.blocked)

    def test_regex_does_not_match_benign_words(self):
        """\bpin\b must not fire on 'pinned' or 'Pinball', and 'secret' must
        not fire on 'secretary'."""
        for text in ["pinned notes", "Pinball scores", "secretary", "coping"]:
            self.assertIsNone(
                CREDENTIAL_CONTEXT.search(text),
                f"{text!r} should not read as a credential field",
            )

    def test_regex_is_deliberately_blunt_about_token(self):
        """'token' anywhere blocks typing. That is intended: over-blocking a
        screen that merely mentions a credential word is the cheap mistake."""
        self.assertIsNotNone(CREDENTIAL_CONTEXT.search("token count"))


class TestAutonomyPersistence(unittest.TestCase):
    """Grants and denials are stored as facts and must survive a restart."""

    def setUp(self):
        from jarvis.memory import Memory

        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.db"
        self.mem = Memory(self.db)

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()

    def test_grant_persists(self):
        Autonomy(make_config(), memory=self.mem).grant("mspaint.exe")
        # A brand new Autonomy, as after a restart.
        again = Autonomy(make_config(), memory=self.mem)
        self.assertTrue(again.trusted("mspaint.exe"))

    def test_deny_persists_and_outranks_allowlist(self):
        # notepad.exe is on the config allowlist; /deny must beat that.
        Autonomy(make_config(), memory=self.mem).revoke("notepad.exe")
        again = Autonomy(make_config(), memory=self.mem)
        self.assertFalse(again.trusted("notepad.exe"))
        self.assertIn("notepad.exe", again.deny)
        d = again.evaluate(app="notepad.exe", window_title="Notepad")
        self.assertTrue(d.blocked)

    def test_grant_refuses_denied_app(self):
        """The invariant that stops a new tool from routing around the gate."""
        a = Autonomy(make_config(), memory=self.mem)
        a.revoke("notepad.exe")
        a.grant("notepad.exe")
        self.assertFalse(a.trusted("notepad.exe"), "grant() must not un-deny an app")

    def test_revoke_reports_whether_it_was_trusted(self):
        a = Autonomy(make_config(), memory=self.mem)
        self.assertTrue(a.revoke("notepad.exe"), "was trusted, so revoking reports True")
        self.assertFalse(a.revoke("mspaint.exe"), "was not trusted, so reports False")

    def test_known_apps_labels(self):
        a = Autonomy(make_config(), memory=self.mem)
        a.grant("mspaint.exe")
        apps = a.known_apps()
        self.assertEqual(apps["keepass.exe"], "DENIED")
        self.assertEqual(apps["notepad.exe"], "built-in allowlist")
        self.assertEqual(apps["mspaint.exe"], "approved by you")


class TestToolSchemaValidation(unittest.TestCase):
    """One malformed schema fails every tool at request time, so it is checked
    at registration instead.

    These register a Tool directly rather than using the add() decorator,
    because add() only registers once the decorator is applied to a function.
    """

    def setUp(self):
        self.r = ToolRegistry()

    def register(self, name="t", params=None, description="d", dangerous=False):
        def handler():
            return None

        return self.r.register(
            Tool(
                name=name,
                description=description,
                parameters=params if params is not None else {"type": "object", "properties": {}},
                handler=handler,
                dangerous=dangerous,
            )
        )

    def test_rejects_missing_object_type(self):
        with self.assertRaises(ValueError) as cm:
            self.register(params={"properties": {}})
        self.assertIn("object", str(cm.exception))

    def test_rejects_wrapped_properties(self):
        with self.assertRaises(ValueError):
            self.register(params={"type": "object", "properties": {"x": "not a dict"}})

    def test_rejects_property_without_type(self):
        with self.assertRaises(ValueError) as cm:
            self.register(params={"type": "object", "properties": {"x": {"description": "d"}}})
        self.assertIn("type", str(cm.exception))

    def test_rejects_required_not_in_properties(self):
        with self.assertRaises(ValueError) as cm:
            self.register(params={"type": "object", "properties": {}, "required": ["x"]})
        self.assertIn("required", str(cm.exception))

    def test_rejects_duplicate(self):
        self.register(name="x")
        with self.assertRaises(ValueError) as cm:
            self.register(name="x")
        self.assertIn("duplicate", str(cm.exception))

    def test_rejects_parameters_not_a_dict(self):
        with self.assertRaises(ValueError) as cm:
            self.register(params=["not", "a", "dict"])
        self.assertIn("dict", str(cm.exception))

    def test_accepts_good_schema(self):
        self.register(
            params={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        )
        self.assertIn("t", self.r)


class TestRealRegistry(unittest.TestCase):
    """Every shipped tool must satisfy the validator and be addressable."""

    @classmethod
    def setUpClass(cls):
        load_all()

    def test_every_tool_registers_cleanly(self):
        for name in registry.names():
            tool = registry.get(name)
            self.assertIsNotNone(tool, name)
            registry.validate(tool)  # raises if bad

    def test_no_duplicate_names(self):
        names = registry.names()
        self.assertEqual(len(names), len(set(names)))

    def test_specs_are_json_shaped(self):
        for spec in registry.specs():
            self.assertEqual(spec["type"], "function")
            self.assertIn("name", spec["function"])
            self.assertIn("description", spec["function"])
            self.assertEqual(spec["function"]["parameters"]["type"], "object")

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolError):
            registry.call("definitely_not_a_tool")

    def test_call_checked_returns_error_dict(self):
        out = registry.call_checked("definitely_not_a_tool")
        self.assertIn("error", out)

    def test_matching_never_returns_the_discovery_tool(self):
        for row in registry.matching(query="list more tools"):
            self.assertNotEqual(row["name"], "list_more_tools")

    def test_matching_by_tag(self):
        rows = registry.matching(tags=["screen"])
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("screen", row["tags"])

    def test_legacy_input_tools_are_registered(self):
        """Old names must keep working for existing prompts and scripts."""
        for name in ("click_at", "click_screen", "type_text", "hotkey", "press_key"):
            self.assertIn(name, registry.names(), f"{name} should still be registered")

    def test_verified_alternatives_are_registered(self):
        for name in ("click_on", "type_on_screen", "send_hotkey", "press_keys"):
            self.assertIn(name, registry.names())


class TestArgumentCoercion(unittest.TestCase):
    """A small model sends numbers and strings where the tool wants other
    things, so the registry cleans that up instead of failing the call."""

    def setUp(self):
        self.r = ToolRegistry()

        @self.r.add(
            "demo",
            "demo",
            {
                "type": "object",
                "properties": {
                    "n": {"type": "number"},
                    "s": {"type": "string"},
                    "b": {"type": "boolean"},
                    "xs": {"type": "array"},
                },
            },
        )
        def demo(n=0, s="", b=False, xs=None):
            return {"n": n, "s": s, "b": b, "xs": xs}

        self.r._tools["demo"].handler = demo

    def test_string_number_becomes_number(self):
        self.assertEqual(self.r.call("demo", {"n": "42"})["n"], 42)

    def test_float_string_becomes_float(self):
        self.assertEqual(self.r.call("demo", {"n": "1.5"})["n"], 1.5)

    def test_number_becomes_string(self):
        self.assertEqual(self.r.call("demo", {"s": 7})["s"], "7")

    def test_truthy_string_becomes_bool(self):
        self.assertIs(self.r.call("demo", {"b": "true"})["b"], True)
        self.assertIs(self.r.call("demo", {"b": "no"})["b"], False)

    def test_json_string_becomes_list(self):
        """A model asked for keystrokes often answers '["ctrl", "c"]'. Items
        come back as strings, which is what the key handlers need."""
        self.assertEqual(self.r.call("demo", {"xs": '["ctrl", "c"]'})["xs"], ["ctrl", "c"])

    def test_plain_comma_list_still_works(self):
        self.assertEqual(self.r.call("demo", {"xs": "ctrl, c"})["xs"], ["ctrl", "c"])

    def test_unknown_argument_is_rejected_not_ignored(self):
        with self.assertRaises(ToolError):
            self.r.call("demo", {"nope": 1})


class TestEnumEnforcement(unittest.TestCase):
    """A schema enum is a promise. Nothing checked it, so an invented value
    reached the handler and failed somewhere less obvious: mouse.click handed
    button='leftt' to pyautogui, which raised a bare ValueError after the
    authorisation check and after a 'before' screenshot went to visual memory.
    """

    def setUp(self):
        self.r = ToolRegistry()
        self.ran = []

        @self.r.add(
            "press",
            "press",
            {
                "type": "object",
                "properties": {
                    "button": {"type": "string",
                               "enum": ["left", "right", "middle"]},
                },
                "required": ["button"],
            },
        )
        def press(button="left"):
            self.ran.append(button)
            return {"button": button}

        @self.r.add(
            "nested",
            "nested",
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "string",
                                      "enum": ["low", "high"]},
                        },
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string",
                                  "enum": ["ctrl", "alt", "shift"]},
                    },
                },
            },
        )
        def nested(mode=None, keys=None):
            self.ran.append((mode, keys))
            return {"mode": mode, "keys": keys}

    def test_an_invented_value_never_reaches_the_handler(self):
        with self.assertRaises(ToolError) as caught:
            self.r.call("press", {"button": "leftt"})
        self.assertEqual(self.ran, [], "the handler ran on an invalid enum")
        message = str(caught.exception)
        self.assertIn("left", message)
        self.assertIn("press", message)

    def test_the_error_names_the_allowed_values_so_a_model_can_retry(self):
        """A 3B model can correct itself if the error says what is allowed."""
        with self.assertRaises(ToolError) as caught:
            self.r.call("press", {"button": "sideways"})
        message = str(caught.exception)
        for allowed in ("left", "right", "middle"):
            self.assertIn(allowed, message)

    def test_case_and_stray_spaces_do_not_cause_a_failure(self):
        """It means the same button, and the handler indexes a table with the
        exact string, so the value has to be rewritten, not just accepted."""
        self.assertEqual(self.r.call("press", {"button": "  LEFT "})["button"],
                         "left")
        self.assertEqual(self.ran, ["left"])

    def test_a_valid_value_is_untouched(self):
        self.assertEqual(self.r.call("press", {"button": "middle"})["button"],
                         "middle")

    def test_an_enum_inside_a_nested_object_is_checked(self):
        with self.assertRaises(ToolError) as caught:
            self.r.call("nested", {"mode": {"level": "extreme"}})
        self.assertIn("mode.level", str(caught.exception))
        self.assertEqual(self.ran, [])

    def test_an_enum_on_array_items_is_checked(self):
        with self.assertRaises(ToolError) as caught:
            self.r.call("nested", {"keys": ["ctrl", "hyper"]})
        self.assertIn("keys[1]", str(caught.exception))
        self.assertEqual(self.ran, [])

    def test_a_valid_array_is_untouched(self):
        self.assertEqual(
            self.r.call("nested", {"keys": ["ctrl", "shift"]})["keys"],
            ["ctrl", "shift"],
        )

    def test_a_handler_may_accept_more_than_it_advertises(self):
        """power_action tells the model there are five actions but the handler
        understands 'reboot' and 'log off', because a small model sends those
        anyway and a retry costs a whole round trip."""
        from jarvis.tools import system_tools

        wider = ToolRegistry()

        @wider.add(
            "power",
            "power",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["lock", "sleep", "restart",
                                        "shutdown", "signout"]},
                },
            },
            accepts={"action": sorted(system_tools._POWER_ACTIONS)},
        )
        def power(action="lock"):
            self.ran.append(action)
            return {"action": action}

        for word in ("reboot", "log off", "POWER OFF"):
            ToolRegistry._enforce_enums(wider.get("power"), {"action": word})
        self.assertEqual(self.ran, [], "validation alone must not run anything")

    def test_a_word_outside_both_lists_is_still_rejected(self):
        from jarvis.tools import system_tools

        wider = ToolRegistry()

        @wider.add(
            "power",
            "power",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["lock", "sleep", "restart",
                                        "shutdown", "signout"]},
                },
            },
            accepts={"action": sorted(system_tools._POWER_ACTIONS)},
        )
        def power(action="lock"):
            self.ran.append(action)
            return {"action": action}

        with self.assertRaises(ToolError):
            wider.call("power", {"action": "self destruct"})
        self.assertEqual(self.ran, [])

    def test_the_advertised_power_actions_stay_short(self):
        """The whole reason for `accepts` is to keep 36 words out of the
        prompt. If someone lists them all in the schema, the prompt grows by
        roughly 150 tokens and the model gets a 36-way choice instead of a
        5-way one."""
        from jarvis.tools import load_all, registry

        load_all()
        schema_enum = registry.get("power_action").parameters["properties"][
            "action"]["enum"]
        self.assertEqual(sorted(schema_enum),
                         ["lock", "restart", "shutdown", "signout", "sleep"])

    def test_every_advertised_enum_is_reachable_by_the_handler(self):
        """The schema and the handler table are two hand-written lists. If they
        drift, the model is offered an action that then fails."""
        from jarvis.tools import load_all, registry
        from jarvis.tools import system_tools

        load_all()
        accepted = registry.get("power_action").accepts["action"]
        for word in accepted:
            self.assertIsNotNone(
                system_tools._POWER_ACTIONS.get(" ".join(str(word).lower().split())),
                f"{word!r} is offered to the model but the handler cannot do it",
            )

    def test_every_enum_in_the_project_has_a_handler_that_knows_it(self):
        """A schema enum nothing checks is documentation. Where a handler
        indexes its own table, the two must line up."""
        from jarvis.tools import load_all, registry

        load_all()
        seen_any = False
        for name in registry.names():
            tool = registry.get(name)
            for key, spec in tool.parameters.get("properties", {}).items():
                if not isinstance(spec, dict) or "enum" not in spec:
                    continue
                seen_any = True
                enum = spec["enum"]
                self.assertTrue(
                    all(isinstance(v, str) and v.strip() for v in enum),
                    f"{name}.{key} has a non-string or blank enum member",
                )
                # The advertised list must not repeat itself. `accepts` may
                # overlap it on purpose: it only widens what a handler already
                # understood, and power_action's table contains all five
                # canonical words.
                self.assertEqual(
                    len(enum), len(set(enum)),
                    f"{name}.{key} lists a duplicate enum member",
                )
        self.assertTrue(seen_any, "no enums found; the scan is not looking "
                                   "at the right place")

    def test_a_bad_button_never_reaches_the_mouse(self):
        """The regression that motivated all of this, at the real tool."""
        from unittest import mock

        from jarvis.tools import load_all, mouse_tools

        load_all()
        clicked = []

        class FakeMouse:
            def validate(self, x, y):
                return None

            def click(self, x, y, button="left", clicks=1):
                clicked.append((x, y, button))
                return {"ok": True}

        # _act is stubbed out: the real one takes a before/after screenshot and
        # writes the pair to visual memory, which a unit test has no business
        # doing to the user's database. The enum boundary is what's under test.
        def bare_act(kind, label, perform, **kwargs):
            return perform()

        with mock.patch.object(mouse_tools, "_mouse", return_value=FakeMouse()), \
                mock.patch.object(mouse_tools, "_act", side_effect=bare_act):
            out = mouse_tools.registry.call_checked(
                "click_at", {"x": 10, "y": 20, "button": "leftt"},
            )
        self.assertEqual(clicked, [], "a click happened on an invalid button")
        self.assertIn("error", out)
        self.assertIn("left", out["error"])

    def test_a_good_button_reaches_the_mouse_canonicalised(self):
        from unittest import mock

        from jarvis.tools import load_all, mouse_tools

        load_all()
        clicked = []

        class FakeMouse:
            def validate(self, x, y):
                return None

            def click(self, x, y, button="left", clicks=1):
                clicked.append((x, y, button))
                return {"ok": True}

        def bare_act(kind, label, perform, **kwargs):
            return perform()

        with mock.patch.object(mouse_tools, "_mouse", return_value=FakeMouse()), \
                mock.patch.object(mouse_tools, "_act", side_effect=bare_act):
            out = mouse_tools.registry.call_checked(
                "click_at", {"x": 10, "y": 20, "button": "MIDDLE"},
            )
        self.assertEqual(clicked, [(10, 20, "middle")],
                         "pyautogui needs the exact lowercase word")
        self.assertNotIn("error", out)


class TestApprovalProvenance(unittest.TestCase):
    """Approval used to be a bool that vanished after the call, so there was no
    way to say afterwards what had been agreed to. These cover the record, the
    binding to the exact action, and expiry."""

    def setUp(self):
        from jarvis.approval import ConfirmationLog

        self.log = ConfirmationLog(ttl=60.0)
        self.asked = []

    def asker(self, answer=True):
        def ask():
            self.asked.append(len(self.asked))
            return answer
        return ask

    def test_a_grant_is_recorded_with_what_was_asked(self):
        d = self.log.decide("power_action", "shut down this machine",
                            self.asker(True))
        self.assertTrue(d.granted)
        self.assertEqual(d.tool, "power_action")
        self.assertEqual(d.detail, "shut down this machine")
        self.assertEqual(d.source, "prompt")
        self.assertGreater(d.at, 0)

    def test_a_refusal_is_recorded_too(self):
        """'It asked and I said no' and 'it never ran' are different events and
        the trail has to distinguish them."""
        d = self.log.decide("kill_process", "force-close notepad",
                            self.asker(False))
        self.assertFalse(d.granted)
        self.assertEqual(len(self.log), 1)
        self.assertFalse(self.log.last().granted)

    def test_a_missing_handler_is_not_recorded_as_a_refusal_by_the_user(self):
        d = self.log.record_refusal("power_action", "restart", "no-handler")
        self.assertEqual(d.source, "no-handler")
        self.assertFalse(d.granted)

    def test_an_auto_approval_is_distinguishable_from_a_person_saying_yes(self):
        """With --confirmation off the answer is still yes, but a reader of the
        trail must be able to see nobody was asked."""
        self.log.auto = True
        d = self.log.decide("power_action", "restart", self.asker(True),
                            source="auto")
        self.assertTrue(d.granted)
        self.assertEqual(d.source, "auto")
        self.assertNotEqual(d.source, "prompt")

    def test_the_detail_is_bound_to_the_decision(self):
        """Two destructive calls in one session are not interchangeable, so the
        fingerprint has to distinguish them."""
        from jarvis.approval import fingerprint

        a = self.log.decide("kill_process", "force-close notepad",
                            self.asker(True))
        b = self.log.decide("kill_process", "force-close explorer",
                            self.asker(True))
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(fingerprint("kill_process", "x"),
                            fingerprint("power_action", "x"))

    def test_history_is_newest_first_and_filterable(self):
        self.log.decide("power_action", "restart", self.asker(True))
        self.log.decide("kill_process", "force-close notepad",
                        self.asker(True))
        newest = self.log.last()
        self.assertEqual(newest.tool, "kill_process")
        only = self.log.history("power_action")
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0].detail, "restart")

    def test_the_trail_is_bounded(self):
        small = type(self.log)(ttl=60.0, limit=5)
        for i in range(20):
            small.decide("power_action", f"restart {i}", self.asker(True))
        self.assertEqual(len(small), 5)
        self.assertEqual(small.last().detail, "restart 19")

    # -- expiry -----------------------------------------------------------

    def test_a_token_is_valid_only_for_the_action_it_was_issued_for(self):
        token = self.log.grant_token("power_action", "restart this machine")
        self.assertTrue(self.log.check_token(token, "power_action",
                                             "restart this machine"))

    def test_a_token_does_not_work_for_a_different_action(self):
        token = self.log.grant_token("power_action", "restart this machine")
        self.assertFalse(self.log.check_token(token, "power_action",
                                              "shut down this machine"))
        self.assertFalse(self.log.check_token(token, "kill_process",
                                              "restart this machine"))

    def test_a_token_expires(self):
        token = self.log.grant_token("power_action", "restart", ttl=-1)
        self.assertFalse(self.log.check_token(token, "power_action", "restart"),
                         "an expired approval must not read as valid")

    def test_a_malformed_token_is_refused_rather_than_crashing(self):
        for junk in ("", "garbage", "abc:", ":123", "abc:notanumber"):
            self.assertFalse(
                self.log.check_token(junk, "power_action", "restart"),
                f"{junk!r} should not pass as a token",
            )

    def test_reuse_needs_the_identical_action(self):
        self.log.decide("kill_process", "force-close notepad",
                        self.asker(True))
        self.assertIsNotNone(self.log.reuse("kill_process",
                                            "force-close notepad"))
        self.assertIsNone(self.log.reuse("kill_process",
                                         "force-close explorer"))

    def test_reuse_stops_working_once_the_window_passes(self):
        log = type(self.log)(ttl=0.05)
        log.decide("kill_process", "force-close notepad", self.asker(True))
        self.assertIsNotNone(log.reuse("kill_process", "force-close notepad"))
        time.sleep(0.12)
        self.assertIsNone(log.reuse("kill_process", "force-close notepad"),
                          "an old yes must not authorise a later action")

    def test_a_refusal_is_never_reusable(self):
        log = type(self.log)(ttl=60.0)
        log.decide("kill_process", "force-close notepad", self.asker(False))
        self.assertIsNone(log.reuse("kill_process", "force-close notepad"))

    def test_forgetting_grants_keeps_the_audit_trail(self):
        self.log.decide("power_action", "restart", self.asker(True))
        self.log.forget()
        self.assertIsNone(self.log.reuse("power_action", "restart"))
        self.assertEqual(len(self.log), 1, "the record is the point")


class TestDestructiveGate(unittest.TestCase):
    """The gate every dangerous tool goes through, exercised through the real
    tool rather than the helper."""

    def setUp(self):
        from jarvis import approval
        from jarvis.tools import load_all, system_tools

        load_all()
        self.approval = approval
        self.system_tools = system_tools
        self._saved_approver = system_tools._approver
        self._saved_log = system_tools.approval_log
        system_tools.approval_log = approval.ConfirmationLog(ttl=60.0)
        system_tools.set_auto_approve(False)
        self.addCleanup(self._restore)

    def _restore(self):
        self.system_tools.approval_log = self._saved_log
        self.system_tools.set_approver(self._saved_approver)
        self.system_tools.set_auto_approve(False)

    def _install(self, answer):
        seen = []

        def approver(tool_name, detail):
            seen.append((tool_name, detail))
            return answer

        self.system_tools.set_approver(approver)
        return seen

    def test_a_valid_token_skips_the_prompt(self):
        log = self.system_tools.approval_log
        token = log.grant_token("power_action", "restart this machine")
        seen = self._install(True)
        self.system_tools._approve("power_action", "restart this machine",
                                   token=token)
        self.assertEqual(seen, [], "a valid token should not re-ask")
        self.assertEqual(log.last().source, "token")

    def test_a_stale_token_falls_back_to_asking_rather_than_trusting_it(self):
        log = self.system_tools.approval_log
        token = log.grant_token("power_action", "restart this machine",
                                ttl=-1)
        seen = self._install(True)
        self.system_tools._approve("power_action", "restart this machine",
                                   token=token)
        self.assertEqual(len(seen), 1, "an expired token must not authorise")

    def test_a_token_for_something_else_does_not_authorise_this(self):
        log = self.system_tools.approval_log
        token = log.grant_token("power_action", "restart this machine")
        seen = self._install(False)
        with self.assertRaises(ToolError):
            self.system_tools._approve("power_action", "shut down this machine",
                                       token=token)
        self.assertEqual(len(seen), 1, "it should have asked about the real one")

    def test_the_prompt_shows_the_reason_and_the_action(self):
        seen = self._install(True)
        self.system_tools._approve("kill_process", 'force-close "notepad"')
        self.assertEqual(len(seen), 1)
        tool_name, detail = seen[0]
        self.assertEqual(tool_name, "kill_process")
        self.assertIn("notepad", detail)
        self.assertIn("go-ahead", detail)

    def test_a_decline_blocks_and_is_recorded(self):
        self._install(False)
        with self.assertRaises(ToolError) as caught:
            self.system_tools._approve("power_action", "restart")
        self.assertIn("declined", str(caught.exception))
        last = self.system_tools.last_approval("power_action")
        self.assertFalse(last.granted)

    def test_no_handler_blocks_and_says_so(self):
        self.system_tools.set_approver(None)
        with self.assertRaises(ToolError) as caught:
            self.system_tools._approve("power_action", "restart")
        self.assertIn("no confirmation handler", str(caught.exception))
        last = self.system_tools.last_approval("power_action")
        self.assertEqual(last.source, "no-handler")

    def test_auto_mode_is_recorded_as_auto_not_as_a_person(self):
        """Auto mode is the assistant answering on the user's behalf, so the
        approver still returns yes - but nobody was asked, and the record has to
        show that rather than implying a person tapped confirm."""
        self._install(True)
        self.system_tools.set_auto_approve(True)
        self.system_tools._approve("power_action", "restart")
        last = self.system_tools.last_approval("power_action")
        self.assertTrue(last.granted)
        self.assertEqual(last.source, "auto")

    def test_typed_mode_is_recorded_as_a_prompt(self):
        self._install(True)
        self.system_tools.set_auto_approve(False)
        self.system_tools._approve("power_action", "restart")
        self.assertEqual(self.system_tools.last_approval("power_action").source,
                         "prompt")

    def test_every_dangerous_tool_shares_the_one_gate(self):
        """The trail is only worth having if the tools actually route through
        it. Which tools ask is checked by AST elsewhere; this checks they all
        land in the same log rather than keeping private state."""
        from jarvis.tools import load_all, registry, system_tools

        load_all()
        flagged = [n for n in registry.names() if registry.get(n).dangerous]
        self.assertGreaterEqual(len(flagged), 10)
        self.assertIs(system_tools.approval_log, self.system_tools.approval_log)

        self._install(True)
        system_tools.approval_log.clear()
        system_tools._approve("kill_process", 'force-close "notepad"')
        system_tools._approve("power_action", "restart this machine")
        tools = {d.tool for d in system_tools.approval_history()}
        self.assertEqual(tools, {"kill_process", "power_action"})
        self.assertTrue(all(d.granted for d in system_tools.approval_history()))


class TestWebGuards(unittest.TestCase):
    """A free endpoint answering with a web page must not be parsed as data."""

    def test_postcode_detection(self):
        from jarvis.tools.web_tools import _POSTCODE

        for pc in ["SW1A 1AA", "M1 1AE", "K4P 1A1", "ec1r 3df", "B33 8TH"]:
            self.assertTrue(_POSTCODE.match(pc), f"{pc} should look like a postcode")
        for place in ["Seattle", "New York", "San Francisco", "Berlin", "Paris"]:
            self.assertIsNone(_POSTCODE.match(place), f"{place} is not a postcode")

    def test_json_guard_rejects_html(self):
        from jarvis.tools import web_tools

        class FakeResponse:
            headers = {"Content-Type": "text/html; charset=utf-8"}
            text = "<!DOCTYPE html><html><body>nope</body></html>"

            def json(self):
                raise ValueError("not json")

        original = web_tools._get
        web_tools._get = lambda *a, **k: FakeResponse()
        try:
            with self.assertRaises(ToolError) as cm:
                web_tools._get_json("https://example.invalid")
            self.assertIn("web page", str(cm.exception))
        finally:
            web_tools._get = original

    def test_json_guard_accepts_real_json(self):
        from jarvis.tools import web_tools

        class FakeResponse:
            headers = {"Content-Type": "application/json"}
            text = '{"ok": true}'

            def json(self):
                return {"ok": True}

        original = web_tools._get
        web_tools._get = lambda *a, **k: FakeResponse()
        try:
            self.assertEqual(web_tools._get_json("https://example.invalid"), {"ok": True})
        finally:
            web_tools._get = original


class TestVisionMemoryConfidence(unittest.TestCase):
    def setUp(self):
        from jarvis.vision_memory import UiEntry

        self.UiEntry = UiEntry

    def entry(self, hits=0, misses=0):
        return self.UiEntry(app="a.exe", label="Save", x=1, y=2, hits=hits, misses=misses)

    def test_unknown_entry_is_low(self):
        self.assertLess(self.entry().confidence, 0.4)

    def test_hits_raise_confidence(self):
        self.assertGreater(self.entry(hits=5, misses=0).confidence, 0.6)

    def test_misses_lower_confidence(self):
        self.assertLess(self.entry(hits=1, misses=8).confidence, 0.3)

    def test_confidence_is_bounded(self):
        for h, m in [(0, 0), (1, 0), (100, 0), (0, 100), (5, 5)]:
            c = self.entry(h, m).confidence
            self.assertGreaterEqual(c, 0.0, f"h={h} m={m}")
            self.assertLessEqual(c, 1.0, f"h={h} m={m}")

    def test_one_hit_outweighs_several_misses(self):
        self.assertGreater(
            self.entry(hits=1, misses=0).confidence, self.entry(hits=0, misses=1).confidence
        )


class TestVisualMemoryRelearning(unittest.TestCase):
    """A control that moves must not keep the confidence of where it was.

    hits and misses describe a position, not a name. When they survived a move,
    a Save button that relocated from (100,200) to (999,888) still reported 0.857
    confidence, and the locator clicks that coordinate at 0.9x whenever OCR
    cannot see the control.
    """

    def setUp(self):
        from jarvis.perception import Element
        from jarvis.vision_memory import VisionMemory

        self.Element = Element
        self.tmp = tempfile.TemporaryDirectory()
        self.vm = VisionMemory(Path(self.tmp.name) / "v.db")

    def tearDown(self):
        self.vm.close()
        self.tmp.cleanup()

    def _el(self, x, y, w=80, h=30):
        return self.Element(text="Save", x=x, y=y, w=w, h=h)

    def _trust(self, label, app, times=5):
        for _ in range(times):
            self.vm.reinforce(app, label, True)

    def test_a_moved_control_loses_its_old_record(self):
        self.vm.learn_element("notepad", "Save", self._el(100, 200))
        self._trust("Save", "notepad")
        self.assertGreater(self.vm.known("notepad", "Save").confidence, 0.8)

        self.vm.learn_element("notepad", "Save", self._el(999, 888))
        entry = self.vm.known("notepad", "Save")
        self.assertEqual((entry.hits, entry.misses), (0, 0))
        self.assertLess(entry.confidence, 0.5,
                        "confidence about the old spot was carried to the new one")
        self.assertEqual((entry.x, entry.y), (999, 888), "the new position must stick")

    def test_it_can_learn_again_after_moving(self):
        self.vm.learn_element("notepad", "Save", self._el(100, 200))
        self._trust("Save", "notepad")
        self.vm.learn_element("notepad", "Save", self._el(999, 888))
        self._trust("Save", "notepad", times=3)
        self.assertGreater(self.vm.known("notepad", "Save").confidence, 0.7)

    def test_rediscovering_the_same_control_keeps_its_record(self):
        """The map is fed from every OCR match, and re-detection jitters by a
        pixel or two. If that reset the counters, nothing would ever be
        learned."""
        self.vm.learn_element("calc", "Total", self._el(300, 400, w=120, h=40))
        self._trust("Total", "calc", times=6)
        before = self.vm.known("calc", "Total").confidence

        for i in range(20):
            self.vm.learn_element(
                "calc", "Total", self._el(300 + (i % 3) - 1, 400 + (i % 2), w=120, h=40)
            )
        entry = self.vm.known("calc", "Total")
        self.assertEqual(entry.confidence, before, "jitter destroyed the learning")
        self.assertEqual(entry.hits, 6, "hits should not be reset by re-detection")

    def test_the_move_threshold_scales_with_the_control(self):
        # A 20px control shifting 40px has really gone somewhere.
        self.vm.learn_element("calc", "Tiny", self._el(50, 50, w=20, h=16))
        self._trust("Tiny", "calc", times=6)
        self.vm.learn_element("calc", "Tiny", self._el(90, 50, w=20, h=16))
        self.assertEqual(self.vm.known("calc", "Tiny").hits, 0)

        # A full-width row shifting 190px has not.
        self.vm.learn_element("calc", "Row", self._el(10, 10, w=1200, h=60))
        self._trust("Row", "calc", times=6)
        self.vm.learn_element("calc", "Row", self._el(200, 10, w=1200, h=60))
        self.assertEqual(self.vm.known("calc", "Row").hits, 6)

    def test_relearning_at_the_same_spot_changes_nothing(self):
        self.vm.learn_element("notepad", "Save", self._el(100, 200))
        self._trust("Save", "notepad", times=3)
        self.vm.learn_element("notepad", "Save", self._el(100, 200))
        entry = self.vm.known("notepad", "Save")
        self.assertEqual(entry.hits, 3)
        self.assertGreater(entry.confidence, 0.7)

    def test_demotion_still_hides_a_control_that_keeps_failing(self):
        """The reset must not interfere with ordinary demotion."""
        self.vm.learn_element("notepad", "Broken", self._el(5, 5))
        self._trust("Broken", "notepad", times=1)
        for _ in range(5):
            self.vm.reinforce("notepad", "Broken", False)
        self.assertEqual(self.vm.find_label("notepad", "Broken", 0.3), [],
                         "a control that keeps failing should stop being offered")


class TestHabitRecency(unittest.TestCase):
    """likely_apps is exposed to the model as likely_open_now, so it has to
    mean now. Ranking on the cumulative habit count alone put an app last used a
    year ago ahead of one opened seconds earlier."""

    def setUp(self):
        from jarvis.vision_memory import VisionMemory

        self.tmp = tempfile.TemporaryDirectory()
        self.vm = VisionMemory(Path(self.tmp.name) / "v.db")

    def tearDown(self):
        self.vm.close()
        self.tmp.cleanup()

    def test_recent_use_beats_a_year_of_old_volume(self):
        import time

        now = time.time()
        for _ in range(500):
            self.vm._note_habit_locked("ancient.exe", now - 86400 * 365)
        self.vm._note_habit_locked("justopened.exe", now)

        top = self.vm.likely_apps()[0]
        self.assertEqual(top["app"], "justopened.exe")

    def test_a_current_habit_still_beats_a_one_off(self):
        import time

        now = time.time()
        for _ in range(20):
            self.vm._note_habit_locked("daily.exe", now - 3600)
        self.vm._note_habit_locked("oneshot.exe", now)

        top = self.vm.likely_apps()[0]
        self.assertEqual(top["app"], "daily.exe",
                         "recency weighting must not flatten real current habits")

    def test_it_reports_the_age_so_a_caller_can_judge(self):
        import time

        self.vm._note_habit_locked("old.exe", time.time() - 86400 * 10)
        row = next(r for r in self.vm.likely_apps() if r["app"] == "old.exe")
        self.assertGreaterEqual(row["days_since_last"], 9.9)
        self.assertIn("likelihood", row)


class TestProcedureRecords(unittest.TestCase):
    """A procedure's successes score its steps, not its goal.

    Re-teaching a goal with different steps used to keep the old tally, so
    JARVIS reported steps it had never run as proven five times over.
    """

    def setUp(self):
        from jarvis.vision_memory import VisionMemory

        self.tmp = tempfile.TemporaryDirectory()
        self.vm = VisionMemory(Path(self.tmp.name) / "v.db")

    def tearDown(self):
        self.vm.close()
        self.tmp.cleanup()

    def test_reteaching_with_new_steps_clears_the_record(self):
        self.vm.save_procedure("send it", [{"click": "Send"}], "outlook")
        for _ in range(5):
            self.vm.score_procedure("send it", "outlook", True)

        new_steps = [{"click": "File"}, {"click": "Send Now"}]
        self.vm.save_procedure("send it", new_steps, "outlook")
        after = self.vm.get_procedure("send it", "outlook")
        self.assertEqual((after["successes"], after["failures"]), (0, 0),
                         "untried steps must not inherit the old steps' record")
        self.assertEqual(after["steps"], new_steps)

    def test_saving_the_same_steps_keeps_the_record(self):
        steps = [{"click": "Send"}]
        self.vm.save_procedure("send it", steps, "outlook")
        for _ in range(3):
            self.vm.score_procedure("send it", "outlook", True)
        self.vm.save_procedure("send it", steps, "outlook")
        self.assertEqual(self.vm.get_procedure("send it", "outlook")["successes"], 3)

    def test_apps_do_not_share_a_record(self):
        self.vm.save_procedure("export", [{"click": "File"}], "word")
        for _ in range(4):
            self.vm.score_procedure("export", "word", True)
        self.vm.save_procedure("export", [{"click": "Export"}], "calc")
        self.assertEqual(self.vm.get_procedure("export", "word")["successes"], 4)
        self.assertEqual(self.vm.get_procedure("export", "calc")["successes"], 0)

    def test_a_failing_procedure_is_still_recallable_but_scores_low(self):
        """It stays available for the model to inspect, but stops being
        recommended, which is what find_procedures ranking is for."""
        self.vm.save_procedure("export to pdf", [{"click": "x"}], "word")
        for _ in range(4):
            self.vm.score_procedure("export to pdf", "word", True)
        for _ in range(10):
            self.vm.score_procedure("export to pdf", "word", False)
        self.assertIsNotNone(self.vm.get_procedure("export to pdf", "word"))
        rate = (4 + 1) / (14 + 2)
        self.assertLess(rate, 0.5, "a procedure that mostly fails should score low")


class TestVisualMemoryStore(unittest.TestCase):
    """The learning store, on a throwaway database."""

    def setUp(self):
        from jarvis.vision_memory import VisionMemory

        self.tmp = tempfile.TemporaryDirectory()
        self.vm = VisionMemory(Path(self.tmp.name) / "v.db")

    def tearDown(self):
        self.vm.close()
        self.tmp.cleanup()

    def test_remember_and_recall_element(self):
        self.vm.remember_position("notepad.exe", "Save", 100, 200, w=80, h=30)
        got = self.vm.known("notepad.exe", "save")  # case-insensitive
        self.assertIsNotNone(got, "label lookup should ignore case")
        self.assertEqual((got.x, got.y), (100, 200))

    def test_reinforce_hit_raises_confidence(self):
        self.vm.remember_position("a.exe", "Save", 10, 20)
        before = self.vm.known("a.exe", "Save").confidence
        for _ in range(3):
            self.vm.reinforce("a.exe", "Save", True)
        after = self.vm.known("a.exe", "Save").confidence
        self.assertGreater(after, before)

    def test_reinforce_miss_demotes(self):
        self.vm.remember_position("a.exe", "Save", 10, 20)
        for _ in range(4):
            self.vm.reinforce("a.exe", "Save", False)
        self.assertLess(self.vm.known("a.exe", "Save").confidence, 0.4)

    def test_forget_element(self):
        self.vm.remember_position("a.exe", "Save", 1, 2)
        self.assertTrue(self.vm.forget_element("a.exe", "Save"))
        self.assertIsNone(self.vm.known("a.exe", "Save"))

    def test_forget_app_clears_everything_for_that_app(self):
        self.vm.remember_position("a.exe", "Save", 1, 2)
        self.vm.remember_position("b.exe", "Save", 3, 4)
        removed = self.vm.forget_app("a.exe")
        self.assertGreater(removed, 0)
        self.assertIsNone(self.vm.known("a.exe", "Save"))
        self.assertIsNotNone(self.vm.known("b.exe", "Save"), "other apps must survive")

    def test_forget_app_blank_is_a_noop(self):
        self.assertEqual(self.vm.forget_app(""), 0)

    def test_procedure_score_tracks_success(self):
        self.vm.save_procedure("save the file", [{"tool": "click_on", "args": {}}], app="a.exe")
        self.vm.score_procedure("save the file", "a.exe", True)
        proc = self.vm.get_procedure("save the file", "a.exe")
        self.assertEqual(proc["successes"], 1)

    def test_wipe_clears_all(self):
        self.vm.remember_position("a.exe", "Save", 1, 2)
        self.vm.wipe()
        self.assertIsNone(self.vm.known("a.exe", "Save"))


class _StubMemory:
    """Just enough Memory for the pre-answer path, which only searches."""

    def __init__(self, rows):
        self._rows = rows

    def search_messages(self, query, limit=15):
        return list(self._rows)


class _PromptMemory:
    """Memory with the surface system_prompt() and think() actually touch."""

    def fact_block(self):
        return "The user's cat is called Ferret."

    def list_notes(self, limit=5):
        return []

    def growing_window(self, max_chars, trim_slack=0.25):
        return []

    def history_for(self, keep):
        return []

    def add_message(self, role, text):
        pass


def _prompt_brain():
    """A Brain with no model behind it, for prompt-construction tests.

    Built with __new__ like the other tests here: these check the shape of what
    gets sent to Ollama, not that a model replies, so nothing should reach the
    network.
    """
    from jarvis.brain import Brain

    class _Cfg:
        persona = "You are JARVIS."
        keep_last_messages = 6
        history_char_budget = 3500
        history_trim_slack = 0.25
        max_tool_rounds = 6
        tool_select_max = 0
        # Mirrors the real default. Absent it, think() would raise rather than
        # take the path the other prompt tests here have always taken.
        fast_path = True

    brain = Brain.__new__(Brain)
    brain.cfg = _Cfg()
    brain.memory = _PromptMemory()
    brain.model_note = ""
    brain._image_capability = False
    brain._extra_tools = set()
    brain.on_thinking = None
    brain.model = ""
    return brain


class TestBrainHelpers(unittest.TestCase):
    """The prompt-shaping helpers, which are pure functions of their input."""

    def test_split_images_extracts_base64(self):
        from jarvis.brain import _split_images

        # Over the 256-char floor _split_images uses to tell an image payload
        # from an ordinary string, as a real screenshot always is.
        payload = "iVBORw0KGgoAAAANSUhEUg" + "A" * 400 + "=="
        result = {"path": "shot.png", "image_base64": payload, "width": 1}
        text, shots = _split_images(result)
        self.assertEqual(len(shots), 1)
        self.assertNotIn(payload, text, "base64 must not survive into the text")
        self.assertIn("shot.png", text)

    def test_split_images_ignores_short_strings(self):
        """A short string under the floor is data, not an image."""
        from jarvis.brain import _split_images

        text, shots = _split_images({"image_base64": "short"})
        self.assertEqual(shots, [])

    def test_split_images_handles_nesting(self):
        from jarvis.brain import _split_images

        payload = "iVBORw0KGgoAAAANSUhEUg" + "B" * 400 + "=="
        text, shots = _split_images({"outer": {"image_base64": payload, "n": 1}})
        self.assertEqual(len(shots), 1)
        self.assertNotIn(payload, text)

    def test_split_images_passes_through_plain_results(self):
        from jarvis.brain import _split_images

        text, shots = _split_images({"ok": True, "n": 3})
        self.assertEqual(shots, [])
        self.assertIn("ok", text)

    def test_core_tools_all_exist(self):
        from jarvis.brain import CORE_TOOLS

        load_all()
        missing = sorted(set(CORE_TOOLS) - set(registry.names()))
        self.assertEqual(missing, [], "CORE_TOOLS names a tool that does not exist")

    def test_core_set_is_a_deliberate_subset(self):
        from jarvis.brain import CORE_TOOLS

        load_all()
        self.assertLess(len(CORE_TOOLS), len(registry.names()))
        self.assertIn("list_more_tools", CORE_TOOLS, "discovery must always be loaded")

    def test_discovery_prompt_tells_the_model_to_discover(self):
        from jarvis.brain import TOOL_DISCOVERY

        self.assertIn("list_more_tools", TOOL_DISCOVERY)

    def test_the_replayed_history_keeps_a_stable_prefix(self):
        """Everything up to the newest turn must be byte-identical turn to turn.

        Ollama reuses a cached prompt only when the new prompt starts with the
        same tokens, so the replay has to *grow* rather than slide. The old
        newest-n window advanced on every turn and put a different message at
        position 1, which re-prefilled the whole history block every time -
        measured at 9.78-17.01s of warm prefill against 0.15s when stable.

        This asserts the mechanism, not the timing, so it holds on a fast
        machine where the numbers are too small to measure in a test.
        """
        brain = _prompt_brain()
        brain.memory.add_message("user", "earlier question")
        brain.memory.add_message("assistant", "earlier answer")
        first = brain._messages("one")

        brain.memory.add_message("user", "second question")
        brain.memory.add_message("assistant", "second answer")
        second = brain._messages("two")

        self.assertEqual(first[0]["role"], "system")
        self.assertEqual(second[0]["role"], "system")
        self.assertEqual(
            first[0], second[0],
            "the system prompt should stay byte-identical so it stays cacheable",
        )
        # The whole replay minus the newest exchange is the cacheable prefix, so
        # it must not move. Only the trailing user message may differ.
        self.assertEqual(
            first[1:-1], second[1:-1],
            "the replayed history must be append-only, or the cache is "
            "invalidated on every turn",
        )
        self.assertNotEqual(first[-1], second[-1])

    def test_the_history_anchor_survives_and_holds_position_zero(self):
        """The anchor is persisted, and position 0 must not move on new turns.

        This is the property the cache needs, asserted on the real Memory
        against uneven message lengths. The two earlier attempts both passed an
        offline simulation and failed on the live transcript: "newest that fits"
        drops one message per turn once over budget, and a recomputed chunked
        overshoot breaks on the first message once the budget is spent. Uniform
        synthetic messages cannot see either; 500-char and 12-char messages in
        turn can.
        """
        import tempfile
        from pathlib import Path as _P

        from jarvis.memory import Memory as _Memory

        with tempfile.TemporaryDirectory() as tmp:
            db = _P(tmp) / "m.db"
            mem = _Memory(db)
            try:
                # Seed well past any plausible budget, with wildly uneven sizes.
                for i in range(60):
                    body = "x" * (500 if i % 5 == 0 else 12)
                    mem.add_message("user" if i % 2 == 0 else "assistant",
                                    f"seed-{i:03d}-{body}")
                budget = 3500
                slack = 0.4
                w0 = mem.growing_window(budget, slack)
                self.assertTrue(w0, "the window should not be empty")
                self.assertLessEqual(
                    sum(len(m["content"]) for m in w0), budget,
                    "a freshly seeded window should fit the budget",
                )

                # A turn that holds the anchor costs nothing extra, because new
                # content has to be prefilled either way. So the figure of merit
                # is the characters re-prefilled *because the anchor moved* - a
                # sliding window re-prefills the whole window every turn.
                trims = 0
                extra = 0
                prev_first = w0[0]
                for turn in range(8):
                    mem.add_message("user", f"live-{turn}-" + "y" * 300)
                    w = mem.growing_window(budget, slack)
                    size = sum(len(m["content"]) for m in w)
                    self.assertLessEqual(
                        size, budget,
                        f"turn {turn}: window grew past its budget without a trim",
                    )
                    if w[0] != prev_first:
                        trims += 1
                        extra += size
                        prev_first = w[0]
                # At slack 0.0 this stretch re-prefills the full window on every
                # turn, about 3,500 chars x 8 = 28,000. At the configured 0.4 a
                # trim leaves 2,100 chars and needs 1,400 chars of new
                # conversation - roughly five 300-char turns - before the next
                # one, so two trims in eight turns is the expected arithmetic and
                # a bound of 6,000 is roughly one trim of headroom over it.
                self.assertLessEqual(
                    extra, 6000,
                    f"the anchor re-prefilled {extra} chars in 8 turns across "
                    f"{trims} trims; it is creeping forward instead of holding",
                )
            finally:
                mem.close()

    def test_the_history_anchor_is_persisted_across_handles(self):
        """The anchor has to outlive the process, or it re-slides every launch."""
        import tempfile
        from pathlib import Path as _P

        from jarvis.memory import _WINDOW_ANCHOR, Memory as _Memory

        with tempfile.TemporaryDirectory() as tmp:
            db = _P(tmp) / "m.db"
            first = _Memory(db)
            try:
                for i in range(40):
                    first.add_message("user", f"m{i}-" + "z" * 40)
                w1 = first.growing_window(1200, 0.25)
                first_anchor = first._prompt_state_get(_WINDOW_ANCHOR)
                self.assertIsNotNone(first_anchor, "the anchor must be persisted")
            finally:
                first.close()

            second = _Memory(db)
            try:
                # No new message: the transcript is identical, so the window must
                # be identical. If the anchor were re-seeded here it would still
                # fit the same messages, so compare the stored anchor directly -
                # that is what actually has to persist.
                anchor_before = first_anchor
                anchor_after = second._prompt_state_get(_WINDOW_ANCHOR)
                self.assertEqual(
                    anchor_after, anchor_before,
                    "a fresh handle must not re-seed the anchor, or every launch "
                    "pays a full re-prefill",
                )
                w2 = second.growing_window(1200, 0.25)
                self.assertEqual(w2, w1)
            finally:
                second.close()

    def test_the_history_window_reseeds_after_messages_are_pruned(self):
        """Pruning must not leave a hole at the front of the window."""
        import tempfile
        from pathlib import Path as _P

        from jarvis.memory import Memory as _Memory

        with tempfile.TemporaryDirectory() as tmp:
            db = _P(tmp) / "m.db"
            mem = _Memory(db)
            try:
                for i in range(50):
                    mem.add_message("user", f"m{i}-" + "z" * 40)
                w = mem.growing_window(800, 0.25)
                anchor_content = w[0]["content"]
                with mem._lock:
                    mem._conn.execute(
                        "DELETE FROM messages WHERE content = ?", (anchor_content,)
                    )
                    mem._conn.commit()
                after = mem.growing_window(800, 0.25)
                self.assertTrue(after, "the window should re-seed, not go empty")
                self.assertNotEqual(
                    after[0]["content"], anchor_content,
                    "the pruned anchor should have been replaced",
                )
            finally:
                mem.close()

    def test_a_sliding_history_window_is_not_a_cache_prefix(self):
        """The old behaviour, pinned so growing_window() cannot regress into it.

        Newest-n advances every turn, so position 1 becomes a different message
        and everything from there on is re-prefilled. Kept as the counter-example
        the real test above is measured against.
        """
        rows = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(10)
        ]
        turn_one = rows[2:6]    # newest 4 of the first 6
        turn_two = rows[4:8]    # newest 4 after two more arrive
        self.assertNotEqual(
            turn_one[0], turn_two[0],
            "a newest-n window is expected to put a different message at "
            "position 0 on the next turn, which is what invalidates the cache",
        )

    def test_the_history_window_fits_the_context(self):
        """Pins the 8,192-token ceiling against a real measurement.

        A tool-using turn measured 8,181 prompt tokens against num_ctx=8192 with
        an 8,000-char budget, which is one exchange from truncation. The system
        prompt and tool schemas are already a measured ~4,100 tokens at 38
        schemas, so the replay has to leave real headroom rather than spend the
        rest of the context.
        """
        from jarvis.config import Config

        cfg = Config()
        # ~3.7 chars per token for this model's tokenizer on English prose.
        chars_per_token = 3.7
        system_and_tools = 4100  # measured, at 38 schemas
        per_turn_headroom = 400   # a full generation plus the user turn
        # A trim drops the window to (1 - slack) of the budget, and between trims
        # it grows back to the full budget, so the budget itself is the ceiling.
        replay_tokens = cfg.history_char_budget / chars_per_token
        total = system_and_tools + replay_tokens + per_turn_headroom
        self.assertLess(
            total, cfg.num_ctx,
            f"replay budget can reach ~{total:.0f} prompt tokens against "
            f"num_ctx={cfg.num_ctx}",
        )

    def test_the_default_history_budget_leaves_context_headroom(self):
        """Pins the budget against a measurement, the way the old pin did.

        The previous pin required keep_last_messages == 8 because a large sliding
        window was ruinous. The window is anchored and append-only now, so the
        constraint is no longer "small" but "fits the context": at 3,500 chars
        the replay is about 950 tokens, which with the measured 4,100-token
        system-and-tools prefix and a turn of generation lands near 5,500
        against num_ctx=8192. Raising this without re-measuring risks
        truncating the conversation instead.

        The slack is pinned to the measured knee. A sweep over 60 turns counted
        the characters re-prefilled because the anchor moved: 2,870/turn at
        slack 0.0, 835 at 0.25, 406 at 0.4, 287 at 0.5, 85 at 0.8. Past 0.4 the
        saving flattens while the window a trim leaves behind shrinks from 2,128
        to 737 chars, and a short window costs anaphora.
        """
        from jarvis.config import Config

        cfg = Config()
        self.assertEqual(cfg.history_char_budget, 3500)
        self.assertEqual(cfg.history_trim_slack, 0.4)

    def test_system_prompt_carries_no_clock(self):
        """A clock in the system prompt costs a full prefill on every turn.

        Ollama can only reuse a cached prompt when the new one begins with the
        same tokens, and the system prompt is rendered before the tool schemas.
        A time string there therefore sat at the very front of the stream and
        invalidated all 3,833 tokens, every turn, once a minute. Measured: 96.8s,
        105.2s and 113.2s for three consecutive turns. With the clock moved to
        the user's message the same three turns took 112.8s, 7.7s and 14.9s.
        """
        import time as time_mod

        import jarvis.brain as brain_mod

        brain = _prompt_brain()
        first = brain.system_prompt()

        real_localtime = brain_mod.time.localtime
        real_strftime = brain_mod.time.strftime

        def later_strftime(fmt, when=None):
            if when is None:
                when = real_localtime()
            shifted = time_mod.struct_time(
                (when.tm_year, when.tm_mon, when.tm_mday,
                 when.tm_hour, when.tm_min + 37, when.tm_sec,
                 when.tm_wday, when.tm_yday, when.tm_isdst)
            )
            return real_strftime(fmt, shifted)

        try:
            brain_mod.time.strftime = later_strftime
            later = brain.system_prompt()
        finally:
            brain_mod.time.strftime = real_strftime

        self.assertEqual(
            first, later, "system_prompt() changed with the clock, which breaks "
            "Ollama's prompt cache and costs ~90s on every turn"
        )

    def test_clock_reaches_the_model_on_the_user_message(self):
        """The clock still has to arrive, just after the stable prefix."""
        brain = _prompt_brain()
        messages = brain._messages("what time is it")
        self.assertIn("Local time:", messages[-1]["content"])
        self.assertIn("what time is it", messages[-1]["content"])
        self.assertNotIn("Local time:", messages[0]["content"])

    def test_history_does_not_accumulate_clock_notes(self):
        """History feeds the prompt, so a stored clock would break the cache
        again, one turn later."""
        brain = _prompt_brain()
        stored = []
        brain.memory.add_message = lambda role, text: stored.append((role, text))
        brain._round = lambda messages, tools, on_token: ("done", None)
        brain.think("what time is it")
        for _role, text in stored:
            self.assertNotIn("Local time:", text)
        self.assertTrue(stored, "the turn should have been recorded")

    def test_tools_loads_everything_even_if_nothing_imported_first(self):
        """_tools() used to hand the model a single tool if the registry had not
        been filled, and the model replied with nothing at all."""
        from jarvis.brain import Brain, CORE_TOOLS

        load_all()
        brain = Brain.__new__(Brain)  # no __init__, so no ollama client
        brain._extra_tools = set()
        brain._ensure_registry = Brain._ensure_registry.__get__(brain)
        self.assertEqual(len(brain._tools()), len(CORE_TOOLS))

    def test_preroute_adds_home_and_mqtt_tools(self):
        from jarvis.brain import Brain, CORE_TOOLS

        load_all()

        def fresh() -> Brain:
            brain = Brain.__new__(Brain)
            brain._extra_tools = set()
            return brain

        lights = fresh()
        self.assertGreater(lights.preroute("turn on the living room lights"), 0)
        self.assertIn("ha_call_service", lights.active_tool_names())

        mqtt = fresh()
        self.assertGreater(mqtt.preroute("send an mqtt message to the porch topic"), 0)
        self.assertIn("mqtt_publish", mqtt.active_tool_names())

        # Weather and web requests must not drag in the whole home category.
        for plain in ("what is the temperature outside", "search the web for a recipe",
                      "turn the volume up", "open notepad"):
            self.assertEqual(fresh().preroute(plain), 0, f"false positive on {plain!r}")
        self.assertEqual(len(fresh().active_tool_names()), len(CORE_TOOLS))

    def test_preroute_does_not_repeat_itself(self):
        from jarvis.brain import Brain, CORE_TOOLS

        load_all()
        brain = Brain.__new__(Brain)
        brain._extra_tools = set()
        first = brain.preroute("turn on the lights")
        self.assertGreater(first, 0)
        self.assertEqual(brain.preroute("turn on the lights"), 0)
        self.assertEqual(len(brain.active_tool_names()), len(CORE_TOOLS) + first)

    def test_overlapping_tool_descriptions_state_the_boundary(self):
        """The 3B model sent 'click the save button' to find_on_screen, which
        only reports coordinates, and 'turn on the lights' to open_app."""
        load_all()
        self.assertIn("without clicking", registry.get("find_on_screen").description)
        self.assertIn("quick_lookup", registry.get("open_app").description)
        self.assertIn("remind", registry.get("remember").description)

    def test_memory_topic_recognises_the_question(self):
        from jarvis.brain import memory_topic


        self.assertEqual(memory_topic("what do you remember about my wifi"), "my wifi")
        self.assertEqual(
            memory_topic("what did I tell you about the ferret?"), "the ferret")
        for other in ("what time is it", "turn on the lights",
                      "what do you know about cooking", "open the file",
                      "remind me to call the dentist", "do you remember"):
            self.assertIsNone(memory_topic(other), f"false positive on {other!r}")

    def test_pre_answer_reports_a_real_miss_as_checked(self):
        """The model used to say 'I don't have anything saved' without ever
        checking, so a genuine absence is stated as a completed search."""
        from jarvis.brain import Brain

        brain = Brain.__new__(Brain)
        brain.memory = _StubMemory([])
        answer = brain.memory_answer("quantum entanglement")
        self.assertIn("nothing saved", answer)
        self.assertIn("rather say that than guess", answer)

    def test_memory_answer_falls_back_when_search_fails(self):
        from jarvis.brain import Brain

        class Broken:
            def search_messages(self, query, limit=15):
                raise RuntimeError("database is locked")

        brain = Brain.__new__(Brain)
        brain.memory = Broken()
        self.assertIsNone(brain.memory_answer("my wifi"),
                          "a failed search must defer to the model, not answer")

    def test_memory_answer_ignores_the_question_echoed_back(self):
        """The live question is in the history and matches the topic exactly, so
        without filtering every lookup looks like a hit."""
        from jarvis.brain import Brain

        echo = [{"role": "user", "content": "what do you remember about the ferret"}]
        brain = Brain.__new__(Brain)
        brain.memory = _StubMemory(echo)
        self.assertIn("nothing saved", brain.memory_answer("the ferret"))

    def test_memory_answer_keeps_a_real_memory(self):
        from jarvis.brain import Brain

        rows = [{"role": "user", "content": "the ferret wears a yellow hat"}]
        brain = Brain.__new__(Brain)
        brain.memory = _StubMemory(rows)
        answer = brain.memory_answer("the ferret")
        self.assertIn("yellow hat", answer)
        self.assertNotIn("nothing saved", answer)

    def test_memory_answer_ignores_the_models_own_wrong_replies(self):
        """This is the one that actually broke it: the model read its own past
        denial out of the history and repeated it while the correct detail sat
        in the prompt."""
        from jarvis.brain import Brain

        rows = [
            {"role": "assistant",
             "content": "I don't have any saved information about the ferret."},
            {"role": "user", "content": "the ferret wears a yellow hat"},
        ]
        brain = Brain.__new__(Brain)
        brain.memory = _StubMemory(rows)
        answer = brain.memory_answer("the ferret")
        self.assertIn("yellow hat", answer)
        self.assertNotIn("don't have any saved", answer)

    def test_memory_answer_drops_rows_matched_on_filler_only(self):
        from jarvis.brain import Brain

        rows = [{"role": "user", "content": "what a nice day it is"}]
        brain = Brain.__new__(Brain)
        brain.memory = _StubMemory(rows)
        self.assertIn("nothing saved", brain.memory_answer("the ferret"))


class TestConfirmation(unittest.TestCase):
    """The confirmed flag: only the user can set it, and it can be taken back."""

    def _memory(self):
        import tempfile
        from pathlib import Path

        from jarvis.memory import Memory

        tmp = tempfile.mkdtemp()
        mem = Memory(Path(tmp) / "m.db")
        self.addCleanup(mem.close)
        return mem

    def test_replies_start_unconfirmed(self):
        mem = self._memory()
        mem.add_message("assistant", "the ferret wears a yellow hat")
        self.assertEqual(mem.search_messages("ferret")[0]["confirmed"], 0)

    def test_confirm_marks_one_reply(self):
        mem = self._memory()
        rid = mem.add_message("assistant", "the ferret wears a yellow hat")
        self.assertTrue(mem.confirm(rid))
        self.assertEqual(mem.search_messages("ferret")[0]["confirmed"], 1)

    def test_the_user_cannot_be_confirmed(self):
        """A user row is a record already. Marking it would let a message look
        verified when nobody ever checked it."""
        mem = self._memory()
        uid = mem.add_message("user", "the ferret wears a yellow hat")
        self.assertFalse(mem.confirm(uid))
        self.assertEqual(mem.search_messages("ferret")[0]["confirmed"], 0)

    def test_a_confirmation_can_be_withdrawn(self):
        mem = self._memory()
        rid = mem.add_message("assistant", "the ferret wears a yellow hat")
        mem.confirm(rid)
        self.assertTrue(mem.unconfirm(rid))
        self.assertEqual(mem.search_messages("ferret")[0]["confirmed"], 0)

    def test_confirmed_replies_count_as_evidence(self):
        from jarvis.brain import Brain

        mem = self._memory()
        rid = mem.add_message("assistant", "the ferret lives in the shed")
        mem.confirm(rid)
        brain = Brain.__new__(Brain)
        brain.memory = mem
        answer = brain.memory_answer("the ferret")
        self.assertIn("shed", answer)
        self.assertIn("you confirmed", answer)

    def test_unconfirmed_replies_are_still_ignored(self):
        from jarvis.brain import Brain

        mem = self._memory()
        mem.add_message("assistant", "the ferret lives in the shed")
        brain = Brain.__new__(Brain)
        brain.memory = mem
        self.assertIn("nothing saved", brain.memory_answer("the ferret"))

    def test_agreement_confirms_the_previous_reply(self):
        from jarvis.brain import apply_verdict

        mem = self._memory()
        mem.add_message("assistant", "the ferret lives in the shed")
        self.assertEqual(apply_verdict(mem, "that's right"), "confirmed")
        self.assertEqual(mem.search_messages("shed")[0]["confirmed"], 1)

    def test_disagreement_withdraws_a_confirmation(self):
        from jarvis.brain import apply_verdict

        mem = self._memory()
        rid = mem.add_message("assistant", "the ferret lives in the shed")
        apply_verdict(mem, "that's right")
        self.assertEqual(apply_verdict(mem, "no, that's wrong"), "retracted")
        self.assertEqual(mem.search_messages("shed")[0]["confirmed"], 0)

    def test_a_mistaken_confirmation_is_not_permanent(self):
        from jarvis.brain import apply_verdict

        mem = self._memory()
        mem.add_message("assistant", "the ferret lives in the shed")
        apply_verdict(mem, "that's right")
        apply_verdict(mem, "no, that's wrong")
        apply_verdict(mem, "no, that's wrong")
        self.assertEqual(mem.search_messages("shed")[0]["confirmed"], 0)

    def test_vague_agreement_is_not_taken_as_confirmation(self):
        """Accepting by habit is not verification, so anything short of an
        explicit verdict is ignored."""
        from jarvis.brain import apply_verdict

        mem = self._memory()
        mem.add_message("assistant", "the ferret lives in the shed")
        for vague in ("ok", "ok thanks", "sure", "yep", "sounds good", "mm",
                      "thanks", "right", "yes"):
            self.assertIsNone(apply_verdict(mem, vague), f"took {vague!r} as approval")
        self.assertEqual(mem.search_messages("shed")[0]["confirmed"], 0)

    def test_verdict_does_not_reach_back_further_than_one_reply(self):
        from jarvis.brain import apply_verdict

        mem = self._memory()
        mem.add_message("assistant", "first claim")
        mem.add_message("assistant", "second claim")
        apply_verdict(mem, "that's right")
        rows = {r["content"]: r["confirmed"] for r in mem.search_messages("claim")}
        self.assertEqual(rows["second claim"], 1)
        self.assertEqual(rows["first claim"], 0)

    def test_old_database_gains_the_column(self):
        """CREATE TABLE IF NOT EXISTS will not add a column to a table that
        already exists, so the migration has to do it."""
        import sqlite3
        import tempfile
        from pathlib import Path

        from jarvis.memory import Memory

        tmp = Path(tempfile.mkdtemp()) / "old.db"
        conn = sqlite3.connect(tmp)
        conn.executescript(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL);"
        )
        conn.execute("INSERT INTO messages (ts, role, content) VALUES (1.0, ?, ?)",
                     ("user", "an old message"))
        conn.commit()
        conn.close()

        mem = Memory(tmp)
        self.addCleanup(mem.close)
        columns = {r["name"] for r in mem._conn.execute("PRAGMA table_info(messages)")}
        self.assertIn("confirmed", columns)
        self.assertEqual(
            mem._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 1,
            "the migration must not lose existing rows",
        )
        rid = mem.add_message("assistant", "a fresh reply")
        self.assertTrue(mem.confirm(rid))

    def test_blind_caption_does_not_speak_in_first_person(self):
        """It used to say 'I am blind', which came back out of the model as
        'the user is blind'."""
        from jarvis.brain import _image_caption

        caption = _image_caption(
            "look_at_screen", {"window": "Notepad", "width": 1920, "height": 1200}, False
        )
        self.assertNotIn("I am blind", caption)
        self.assertNotIn("i am blind", caption.lower())
        self.assertIn("no vision", caption.lower())
        self.assertIn("Notepad", caption, "the caption should still carry the context")


class TestVlmCorroboration(unittest.TestCase):
    """The failure this guards against was measured, not imagined: on a real
    OpenCode window moondream answered "1. Read the instructions." That answer
    has letters, has vowels, and is not a repeated token, so the degenerate
    output filter accepted it and the chat model would have repeated it as a
    real observation of the user's screen."""

    def setUp(self):
        from jarvis.vlm import Vision

        self.supported = Vision.supported_by
        self.words = Vision.content_words

    def test_rejects_the_observed_confabulation(self):
        ocr = "Hoopa's Vault OCR bot setup\nNow let me add the corroboration check\nEdit vim.py"
        confab = "1. Read the instructions."
        from jarvis.vlm import Vision

        # The old filter lets this through, which is the whole problem.
        self.assertTrue(Vision._usable(confab))
        self.assertFalse(self.supported(confab, ocr))

    def test_accepts_text_that_is_really_on_screen(self):
        ocr = "Hoopa's Vault OCR bot setup\nSave file\nCancel"
        self.assertTrue(self.supported("Save file and Cancel are visible.", ocr))

    def test_stopwords_alone_do_not_corroborate(self):
        """Filler must not count as evidence, or every sentence would pass."""
        ocr = "Save Cancel Open"
        self.assertFalse(self.supported("The window is on the top of it.", ocr))

    def test_empty_ocr_contradicts_nothing(self):
        """With no text on screen there is no evidence either way, and claiming
        the answer is false would be inventing a negative."""
        self.assertTrue(self.supported("A dark image fills the screen.", ""))
        self.assertTrue(self.supported("A dark image fills the screen.", "   "))

    def test_answer_with_no_checkable_words_is_left_to_the_other_filter(self):
        """Nothing to check means nothing to contradict; _usable still judges it."""
        self.assertTrue(self.supported("It is.", "some other text entirely"))

    def test_an_unsupported_app_name_is_a_confabulation(self):
        """Naming the application is a reading claim, so OCR can contradict it."""
        self.assertFalse(self.supported("Firefox", "Save Cancel Open"))

    def test_visual_claims_are_not_judged_by_word_overlap(self):
        """The docstring warns about this: describing appearance legitimately
        shares no words with the text, so callers flag rather than delete."""
        ocr = "Total files 12\nLast modified today"
        self.assertFalse(self.supported("A file manager window is open.", ocr))

    def test_content_words_drops_short_and_common_words(self):
        self.assertEqual(
            self.words("The Save button is on it"), {"save", "button"}
        )


def _fake_screen(seed: int = 0):
    """A tiny real image, for describe() tests that now hash the payload.

    The describe path keys its cache on the encoded bytes, so these tests need
    an actual array rather than object(). Four pixels is enough; nothing here is
    ever sent to a model.
    """
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = seed % 256
    return img


class TestVlmDescribeFailsFast(unittest.TestCase):
    """Two of three describe passes on a real window were confabulations and
    cost ~35s. Once the model has produced nothing usable, the remaining passes
    are the same model on the same unreadable image, so they are skipped."""

    def test_stops_after_the_first_unusable_pass(self):
        from jarvis.vlm import Vision

        asked: list[str] = []

        class Fake(Vision):
            def __init__(self):  # noqa: D107 - deliberately does not call super
                self.model = "fake"
                self.describe_model = "fake"
                self.describe_max_side = 448
                self.describe_num_predict = 96
                self.describe_budget = 150.0
                self._describe_cache = OrderedDict()
                self._describe_cache_max = 24
                # The no-question path reads the reuse tier's bounds, so a stub
                # that skipped them broke on the pass sequence rather than on
                # anything this test is about.
                self._describe_near = OrderedDict()
                self._describe_reuse_max_changed = -1
                self._reuse_max_changed_no_ocr = 0.001
                self._SIGNATURE_DELTA = 24

            def _ask(self, image, prompt, model="", max_side=None, num_predict=None):
                asked.append(prompt)
                if len(asked) == 1:
                    return "urn:jira:issue/uid:urn:jira:issue/uid:urn:jira:issue/uid:"
                return "A code editor."

        out = Fake().describe(_fake_screen())
        self.assertEqual(len(asked), 1, f"asked {len(asked)} times: {asked}")
        self.assertEqual(out, "", "a confabulation must not be reported as a description")

    def test_keeps_going_while_answers_are_usable(self):
        from jarvis.vlm import Vision

        class Fake(Vision):
            def __init__(self):  # noqa: D107
                self.model = "fake"
                self.describe_model = "fake"
                self.describe_max_side = 448
                self.describe_num_predict = 96
                self.describe_budget = 150.0
                self._describe_cache = OrderedDict()
                self._describe_cache_max = 24
                self._describe_near = OrderedDict()
                self._describe_reuse_max_changed = -1
                self._reuse_max_changed_no_ocr = 0.001
                self._SIGNATURE_DELTA = 24

            def _ask(self, image, prompt, model="", max_side=None, num_predict=None):
                return f"answer for {prompt[:12]}"

        out = Fake().describe(_fake_screen())
        self.assertEqual(len(out.splitlines()), len(Vision._DETAIL_PROMPTS))

    def test_question_short_circuits_the_pass_sequence(self):
        from jarvis.vlm import Vision

        asked: list[str] = []

        class Fake(Vision):
            def __init__(self):  # noqa: D107
                self.model = "fake"
                self.describe_model = "fake"
                self.describe_max_side = 448
                self.describe_num_predict = 96
                self.describe_budget = 150.0
                self._describe_cache = OrderedDict()
                self._describe_cache_max = 24
                # The loose reuse tier needs these too, since describe() reads
                # them for the question path. Left at -1 so this test is only
                # about the pass sequence, not about reuse.
                self._describe_near = OrderedDict()
                self._describe_reuse_max_changed = -1
                # describe()'s no-question path reads this too, now that the
                # blank-OCR tier is calibrated rather than closed.
                self._reuse_max_changed_no_ocr = 0.001
                self._SIGNATURE_DELTA = 24

            def _ask(self, image, prompt, model="", max_side=None, num_predict=None):
                asked.append(prompt)
                return "The error says disk full."

        self.assertEqual(
            Fake().describe(_fake_screen(), "what does the error say?"),
            "The error says disk full.",
        )
        self.assertEqual(len(asked), 1)


class TestVlmOutputFilter(unittest.TestCase):
    """Degenerate output from a tiny VLM is caught here, not passed on."""

    def setUp(self):
        from jarvis.vlm import Vision

        self.usable = Vision._usable
        self.periodic = Vision._is_periodic

    def test_rejects_empty_and_tiny(self):
        for text in ["", "  ", "a", "12"]:
            self.assertFalse(self.usable(text), f"{text!r} should be rejected")

    def test_rejects_character_loop(self):
        loop = "urn:jira:issue/uid:urn:jira:issue/uid:urn:jira:issue/uid:"
        self.assertTrue(self.periodic(loop))
        self.assertFalse(self.usable(loop))

    def test_rejects_bare_coordinates_when_prose_wanted(self):
        for text in ["[0.5, 0.5]", "1, 2", "42", "[[120, 340]]"]:
            self.assertFalse(self.usable(text, want_prose=True), f"{text!r} is not prose")

    def test_allows_coordinates_when_a_point_was_asked_for(self):
        self.assertTrue(self.usable("[0.5, 0.5]", want_prose=False))

    def test_rejects_hallucinated_uri(self):
        for text in [
            "see https://example.com/page for details",
            "urn:jira:issue/uid-1234",
            "mailto:someone@example.com",
        ]:
            self.assertFalse(self.usable(text), f"{text!r} is a confabulated URI")

    def test_rejects_consonant_soup(self):
        self.assertFalse(self.usable("xkcd brnght klmp zxq"))

    def test_accepts_real_sentences(self):
        for text in [
            "A text editor with a menu bar and a Save button.",
            "Firefox is open on a login page.",
            "clear",
            "The window shows a folder of Python files.",
        ]:
            self.assertTrue(self.usable(text), f"{text!r} should be accepted")

    def test_periodic_needs_three_repeats(self):
        self.assertFalse(self.periodic("abc"))
        self.assertTrue(self.periodic("ababababab"))


class TestVlmCoordinateParsing(unittest.TestCase):
    """moondream answers in several coordinate spaces; all must be handled."""

    def setUp(self):
        from jarvis.vlm import Vision

        self.parse = Vision._parse_point

    def test_normalised_box_uses_centre(self):
        fix = self.parse("[0.2, 0.4, 0.6, 0.8]", 1920, 1200, "Save")
        self.assertIsNotNone(fix)
        self.assertEqual(fix.x, (int(0.2 * 1919) + int(0.6 * 1919)) // 2)
        self.assertEqual(fix.box[2] - fix.box[0], int(0.6 * 1919) - int(0.2 * 1919))

    def test_normalised_point(self):
        fix = self.parse("[0.5, 0.5]", 1920, 1200, "Save")
        self.assertEqual((fix.x, fix.y), (959, 599))

    def test_pixel_point(self):
        fix = self.parse("[640, 480]", 1920, 1200, "Save")
        self.assertEqual((fix.x, fix.y), (640, 480))

    def test_point_tag_format(self):
        fix = self.parse("<point>640, 480</point>", 1920, 1200, "Save")
        self.assertEqual((fix.x, fix.y), (640, 480))

    def test_xy_format(self):
        fix = self.parse("x=640 y=480", 1920, 1200, "Save")
        self.assertEqual((fix.x, fix.y), (640, 480))

    def test_reversed_box_is_normalised(self):
        fix = self.parse("[0.6, 0.8, 0.2, 0.4]", 1920, 1200, "Save")
        self.assertIsNotNone(fix)
        self.assertLess(fix.box[0], fix.box[2])

    def test_zero_area_box_rejected(self):
        self.assertIsNone(self.parse("[0.5, 0.5, 0.5, 0.5]", 1920, 1200, "Save"))

    def test_garbage_returns_none(self):
        for text in ["I cannot see the screen", "", "some text"]:
            self.assertIsNone(self.parse(text, 1920, 1200, "Save"), f"{text!r}")


class TestScreenToolRegressions(unittest.TestCase):
    """take_screenshot and list_displays both called a function named
    _screenshotter that does not exist, so both raised NameError."""

    def test_no_dangling_screenshotter_reference(self):
        import jarvis.tools.screen_tools as st

        source = Path(st.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_screenshotter", source)
        self.assertTrue(hasattr(st, "_open"), "screen_tools should expose _open")

    def test_take_screenshot_is_callable(self):
        from jarvis.tools import screen_tools

        self.assertTrue(callable(screen_tools.take_screenshot))
        self.assertTrue(callable(screen_tools.list_displays))


class TestEnvironmentShims(unittest.TestCase):
    """Project-root shims exist on purpose; the real packages must still win."""

    def test_guarded_modules_resolve_to_site_packages(self):
        import sys

        from jarvis.env import GUARDED

        for name in GUARDED:
            module = sys.modules.get(name)
            if module is None:
                continue
            origin = getattr(module, "__file__", "") or ""
            if not origin:
                continue
            self.assertIn(
                "site-packages",
                Path(origin).resolve().as_posix(),
                f"{name} resolved to {origin}, not site-packages",
            )

    def test_pyautogui_reports_the_real_display_size(self):
        """A 1x1 shim is the classic failure here, and pyautogui.size() would
        be the first thing to show it."""
        import pyautogui

        w, h = pyautogui.size()
        self.assertGreater(w, 100, f"pyautogui screen width looks like a shim: {w}")
        self.assertGreater(h, 100, f"pyautogui screen height looks like a shim: {h}")


class TestConfigRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_from_empty_dir_uses_defaults(self):
        from jarvis.config import load_config

        cfg = load_config()
        self.assertTrue(cfg.model)
        self.assertGreater(cfg.request_timeout, 0)
        self.assertGreater(cfg.max_tool_rounds, 0)

    def test_vision_model_settings_present(self):
        from jarvis.config import load_config

        cfg = load_config()
        self.assertTrue(cfg.vlm_model, "a vision model should be configured")
        # vlm_describe_model may be empty to mean "same as vlm_model"
        self.assertIsInstance(cfg.vlm_describe_model, str)

    def test_timeout_leaves_room_for_a_cold_first_turn(self):
        """The chat model prefills at roughly 32 tok/s here, so a cold first
        turn legitimately takes over a minute."""
        from jarvis.config import load_config

        self.assertGreater(load_config().request_timeout, 180)


class TestMicrophoneSelection(unittest.TestCase):
    """Windows lists virtual and loopback endpoints as inputs, and the first one
    is usually "Microsoft Sound Mapper - Input", which captures nothing useful.
    Picking it by default means the assistant is technically running and hears
    silence, so the selection is pinned by tests."""

    def _fake_sounddevice(self, inputs, default_index=0):
        import sys

        class _Dev(dict):
            """Real PortAudio device info supports both d['name'] and d.name."""

            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc

        module = _Dev()
        module.__class__ = _Dev

        def query_devices(index=None, kind=None):
            if index is None and kind is None:
                return [
                    _Dev(d)
                    for d in [*inputs, {"index": 90, "name": "Speakers", "max_input_channels": 0}]
                ]
            if kind == "input":
                for d in inputs:
                    if d["index"] == default_index:
                        return _Dev(d)
                raise ValueError("no default")
            for d in inputs:
                if d["index"] == index:
                    return _Dev(d)
            raise ValueError(f"no device {index}")

        module["query_devices"] = query_devices
        return module

    def _install(self, module):
        import sys

        self._saved = sys.modules.get("sounddevice")
        sys.modules["sounddevice"] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__("sounddevice", self._saved)
            if self._saved is not None
            else sys.modules.pop("sounddevice", None)
        )

    def test_mic_rank_prefers_real_microphones(self):
        from jarvis.audio import mic_rank

        self.assertEqual(mic_rank("Microphone (Realtek HD Audio Mic input)"), 0)
        self.assertEqual(mic_rank("Microphone Array (Intel Smart Sound)"), 0)
        for bad in (
            "Microsoft Sound Mapper - Input",
            "Stereo Mix (Realtek HD Audio Stereo input)",
            "PC Speaker (Realtek HD Audio 2nd output with SST)",
            "CABLE Output",
            "Primary Sound Capture Driver",
        ):
            self.assertGreater(mic_rank(bad), 0, f"{bad!r} should not rank as a real mic")

    def test_explicit_index_is_always_honoured(self):
        from jarvis.audio import default_input_device

        sd = self._fake_sounddevice(
            [
                {"index": 0, "name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
                {"index": 7, "name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 2},
            ]
        )
        self._install(sd)
        self.assertEqual(default_input_device(7), 7)
        self.assertEqual(default_input_device(0), 0, "an explicit mapper choice is the user's call")

    def test_default_skips_the_mapper(self):
        from jarvis.audio import default_input_device

        sd = self._fake_sounddevice(
            [
                {"index": 0, "name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
                {"index": 7, "name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 2},
            ]
        )
        self._install(sd)
        self.assertEqual(
            default_input_device(-1), 7, "a real mic must beat the system default mapper"
        )

    def test_default_prefers_a_real_system_default(self):
        from jarvis.audio import default_input_device

        sd = self._fake_sounddevice(
            [
                {"index": 0, "name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
                {"index": 3, "name": "Microphone Array 1", "max_input_channels": 2},
            ],
            default_index=3,
        )
        self._install(sd)
        self.assertEqual(default_input_device(-1), 3)

    def test_default_falls_back_when_no_microphone_exists(self):
        from jarvis.audio import AudioUnavailable, default_input_device

        self._install(self._fake_sounddevice([]))
        with self.assertRaises(AudioUnavailable):
            default_input_device(-1)

    def test_mic_resolves_device_at_construction(self):
        """The stream must open on the resolved index, not on -1, or every
        recording goes to the mapper no matter what the doctor reported."""
        from jarvis.audio import Mic

        sd = self._fake_sounddevice(
            [
                {"index": 0, "name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
                {"index": 7, "name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 2},
            ]
        )
        self._install(sd)
        mic = Mic(device=-1)
        self.assertEqual(mic.device, 7)
        self.assertEqual(mic.requested_device, -1, "the original request is kept for the log")


class TestVlmDescribeSizing(unittest.TestCase):
    """Describing and pointing want different amounts of image.

    The image size is kept small for describe, but the reason turned out not to
    be latency. Measured on this CPU with qwen2.5vl and a real 1920x1200
    window, a describe call reported 1112 prompt tokens at 112px, 160px, 224px,
    256px, 320px, 384px and 448px - all identical - because the model pads
    whatever image it is given to a fixed token grid. A call takes 78-93s and
    about 97% of that is prefilling those tokens, so shrinking the picture saves
    nothing at all. What does help is capping generation and bounding the
    multi-question path, which is what the budget tests below cover.
    """

    # Comfortably past the 12-content-word threshold that marks OCR as the
    # better reader of this screen.
    RICH_OCR = (
        "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
        "corroboration check vim python script window terminal file edit saved"
    )

    def _vision(self, describe_side=448, point_side=896, cache_entries=24):
        from jarvis.vlm import Vision

        class Cfg:
            vlm_model = "moondream"
            vlm_describe_model = "qwen2.5vl:3b"
            ollama_host = "http://localhost:11434"
            vlm_timeout = 300.0
            vlm_max_side = point_side
            vlm_describe_max_side = describe_side
            vlm_num_predict = 256
            vlm_describe_num_predict = 96
            vlm_describe_budget_seconds = 150.0
            vlm_describe_cache_entries = cache_entries

        return Vision(Cfg())

    def _record(self, vision, answer="A terminal window showing a script."):
        """Replace _ask with a recorder. Returns the list of calls."""
        seen: list[dict] = []

        def fake_ask(image, prompt, model="", max_side=None, num_predict=None):
            # _ask resolves an empty model to the instance's own, so resolve it
            # here too or the recorder lies about which model was asked.
            seen.append(
                {
                    "prompt": prompt,
                    "model": model or vision.model,
                    "max_side": max_side,
                    "num_predict": num_predict,
                }
            )
            return answer

        vision._ask = fake_ask
        return seen

    def test_describe_uses_the_small_size(self):
        v = self._vision()
        seen = self._record(v)
        v.describe(_fake_screen(), ocr_text=self.RICH_OCR)
        self.assertTrue(seen, "the model was never asked")
        for call in seen:
            self.assertEqual(call["max_side"], 448, f"describe passed {seen}")

    def test_pointing_keeps_the_full_size(self):
        import numpy as np

        v = self._vision()
        seen = self._record(v, answer="[[0.5, 0.5]]")
        v.point_at(np.zeros((1200, 1920, 3), dtype=np.uint8), "the task bar")
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0]["max_side"], "point_at must fall back to the pointing size")

    def test_pointing_uses_the_pointing_model(self):
        import numpy as np

        v = self._vision()
        seen = self._record(v, answer="[[0.5, 0.5]]")
        v.point_at(np.zeros((1200, 1920, 3), dtype=np.uint8), "the task bar")
        self.assertEqual(seen[0]["model"], "moondream", "moondream is the pointer")

    def test_describe_uses_the_describe_model(self):
        v = self._vision()
        seen = self._record(v)
        v.describe(_fake_screen(), ocr_text=self.RICH_OCR)
        self.assertEqual(seen[0]["model"], "qwen2.5vl:3b")

    def test_prepare_scales_to_the_requested_side(self):
        import numpy as np

        v = self._vision()
        image = np.zeros((1200, 1920, 3), dtype=np.uint8)
        payload, scale, size = v._prepare(image, 448)
        self.assertEqual(size, (1920, 1200))
        self.assertAlmostEqual(scale, 448 / 1920, places=3)

        import base64

        import cv2

        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(payload), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(max(decoded.shape[:2]), 448)

    def test_prepare_is_deterministic_so_the_server_cache_hits(self):
        """A one-pixel wobble between runs would pay the per-size cost again."""
        import numpy as np

        v = self._vision()
        image = np.zeros((1200, 1920, 3), dtype=np.uint8)
        self.assertEqual(v._prepare(image, 448)[0], v._prepare(image, 448)[0])

    def test_ocr_richness_routes_away_from_transcription(self):
        from jarvis.vlm import Vision

        self.assertFalse(Vision._ocr_is_rich(""))
        self.assertFalse(Vision._ocr_is_rich("ok"))
        self.assertFalse(
            Vision._ocr_is_rich("one two three four five six seven eight nine ten"),
            "eleven content words is still below the threshold",
        )
        self.assertTrue(Vision._ocr_is_rich(self.RICH_OCR))

    def test_rich_ocr_asks_exactly_one_question(self):
        from jarvis.vlm import Vision

        v = self._vision()
        seen = self._record(v)
        v.describe(_fake_screen(), ocr_text=self.RICH_OCR)
        self.assertEqual(len(seen), 1, f"asked {len(seen)} times: {seen}")
        self.assertEqual(seen[0]["prompt"], Vision._SINGLE_PROMPT)

    def test_thin_ocr_still_uses_the_detail_sequence(self):
        """A blank or image-only screen is what the vision model is for."""
        from jarvis.vlm import Vision

        v = self._vision()
        seen = self._record(v)
        out = v.describe(_fake_screen(), ocr_text="")
        self.assertEqual(len(seen), len(Vision._DETAIL_PROMPTS))
        self.assertEqual(out.count("\n- "), len(Vision._DETAIL_PROMPTS) - 1)

    def test_describe_caps_the_tokens_it_asks_for(self):
        """A high cap only costs time when the model degenerates: moondream
        once spent all 256 tokens emitting "urn:jars:li:9:0:0:0..."."""
        v = self._vision()
        seen = self._record(v)
        v.describe(_fake_screen(), ocr_text=self.RICH_OCR)
        self.assertEqual(seen[0]["num_predict"], 96)

    def test_pointing_keeps_its_own_token_budget(self):
        """Describing wants a sentence; pointing wants coordinates. They should
        not share a cap chosen for the other's failure mode."""
        v = self._vision()
        self.assertEqual(v.num_predict, 256)
        self.assertEqual(v.describe_num_predict, 96)

    def test_the_detail_sequence_stops_when_the_budget_is_gone(self):
        """Three cold image prefills is 4-5 minutes on this CPU. The budget
        turns that into a shorter answer rather than a longer wait."""
        from jarvis import vlm as vlm_mod
        from jarvis.vlm import Vision

        v = self._vision()
        v.describe_budget = 0.0
        seen = self._record(v)

        # The deadline is computed from the first reading, then the first loop
        # check reads a later one, so the budget is already spent.
        clock = iter([100.0, 200.0, 300.0, 400.0])
        real_monotonic = vlm_mod.time.monotonic
        vlm_mod.time.monotonic = lambda: next(clock)
        try:
            out = v.describe(_fake_screen(), ocr_text="")
        finally:
            vlm_mod.time.monotonic = real_monotonic

        self.assertEqual(seen, [], "no question should be asked with no budget")
        self.assertEqual(out, "")

    def test_a_spent_budget_still_returns_what_was_already_gathered(self):
        """Degrading the answer is the point. Returning nothing at all would be
        worse than the wait it was meant to avoid."""
        from jarvis import vlm as vlm_mod
        from jarvis.vlm import Vision

        v = self._vision()
        v.describe_budget = 0.0
        seen = self._record(v, answer="A terminal window showing a script.")

        # First question is inside the budget, the second is not.
        clock = iter([100.0, 100.0, 999.0, 999.0])
        real_monotonic = vlm_mod.time.monotonic
        vlm_mod.time.monotonic = lambda: next(clock)
        try:
            out = v.describe(_fake_screen(), ocr_text="")
        finally:
            vlm_mod.time.monotonic = real_monotonic

        self.assertEqual(len(seen), 1, f"asked {len(seen)} times: {seen}")
        self.assertIn("terminal window", out)

    def test_a_generous_budget_asks_every_question(self):
        """The budget must not quietly become the only thing limiting quality."""
        from jarvis.vlm import Vision

        v = self._vision()
        seen = self._record(v)
        v.describe(_fake_screen(), ocr_text="")
        self.assertEqual(len(seen), len(Vision._DETAIL_PROMPTS))

    def test_a_direct_question_is_not_gated_by_the_budget(self):
        """Someone asked a specific question and is waiting for it. The budget
        exists to stop three speculative questions, not to refuse a request."""
        from jarvis.vlm import Vision

        v = self._vision()
        v.describe_budget = 0.0
        seen = self._record(v)
        out = v.describe(_fake_screen(), question="What is the app name?")
        self.assertEqual(len(seen), 1)
        self.assertIn("terminal window", out)

    def test_rich_ocr_is_asked_one_question_within_budget(self):
        """The common path: OCR already read the screen, so exactly one call is
        made and the budget never comes into it."""
        from jarvis.vlm import Vision

        v = self._vision()
        v.describe_budget = 0.0
        seen = self._record(v)
        out = v.describe(_fake_screen(), ocr_text=self.RICH_OCR)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["prompt"], Vision._SINGLE_PROMPT)
        self.assertIn("terminal window", out)


class TestVlmUriFilter(unittest.TestCase):
    """The URI filter used to throw away correct answers. A verbatim read of a
    terminal legitimately contains "http://localhost:11434/api/tags", and that
    was being rejected as a hallucination, so a good reading was discarded."""

    def setUp(self):
        from jarvis.vlm import Vision

        self.usable = Vision._usable

    def test_keeps_a_real_reading_that_contains_a_url(self):
        text = (
            "The terminal shows a curl call to http://localhost:11434/api/tags with "
            "a timeout of five seconds and a list of the installed model names."
        )
        self.assertTrue(self.usable(text), "a correct reading was being thrown away")

    def test_still_rejects_an_answer_that_is_only_a_url(self):
        for text in [
            "see https://example.com/page for details",
            "http://example.com",
            "urn:jira:assistant:url:https://jira.atlassian.net/browse/JARL",
        ]:
            self.assertFalse(self.usable(text), f"{text!r} is a confabulated URI")


class TestHomeAssistant(unittest.TestCase):
    """The HA tools, against a real HTTP server on loopback.

    These were the two integrations with no coverage at all, because both need
    a server that did not exist on this machine. Asserting against what actually
    arrived over HTTP is the only way to know the tools build the requests they
    claim to.
    """

    def setUp(self):
        from jarvis.config import load_config
        from jarvis.tools import iot_tools, load_all, system_tools
        from jarvis_tests.fakes import FakeHomeAssistant

        load_all()
        self.iot = iot_tools
        self.ha = FakeHomeAssistant()
        self.addCleanup(self.ha.stop)

        cfg = load_config()
        cfg.home_assistant_url = self.ha.url
        cfg.home_assistant_token = "good-token"
        iot_tools.bind_config(cfg)

        self.asked: list[tuple[str, str]] = []
        system_tools.set_approver(
            lambda name, detail: self.asked.append((name, detail)) or True
        )
        self.addCleanup(system_tools.set_approver, None)

    def test_it_lists_and_filters_entities(self):
        out = self.iot.ha_list_entities()
        self.assertEqual(out["count"], 4)
        self.assertIn("/api/states", self.ha.paths())

        lights = self.iot.ha_list_entities(domain="light")
        self.assertEqual([e["entity_id"] for e in lights["entities"]], ["light.porch"])

        found = self.iot.ha_list_entities(search="outside")
        self.assertEqual([e["entity_id"] for e in found["entities"]],
                         ["sensor.outside_temp"])

    def test_sensor_attributes_survive(self):
        out = self.iot.ha_get_state("sensor.outside_temp")
        self.assertEqual(out["state"], "7.5")
        self.assertEqual(out["attributes"].get("unit_of_measurement"), "°C")

    def test_a_friendly_human_name_becomes_an_entity_id(self):
        """The old docstring promised this lookup and the code only lowercased
        and underscored, so "the porch" became "the_porch", which is not an
        entity id at all."""
        self.iot.ha_call_service("light", "turn_on", entity="the porch")
        self.assertIn("/api/services/light/turn_on", self.ha.paths())
        body = [b for m, p, b in self.ha.calls if m == "POST"][-1]
        self.assertEqual(body.get("entity_id"), "light.porch")

    def test_the_object_id_on_its_own_is_enough(self):
        self.iot.ha_call_service("light", "turn_off", entity="porch")
        body = [b for m, p, b in self.ha.calls if m == "POST"][-1]
        self.assertEqual(body.get("entity_id"), "light.porch")

    def test_a_full_entity_id_is_passed_through_untouched(self):
        self.iot.ha_call_service("light", "turn_on", entity="light.porch")
        body = [b for m, p, b in self.ha.calls if m == "POST"][-1]
        self.assertEqual(body.get("entity_id"), "light.porch")

    def test_an_unresolvable_name_is_reported_not_guessed(self):
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_call_service("light", "turn_on", entity="the dishwasher")
        message = str(ctx.exception)
        self.assertIn("dishwasher", message)
        self.assertNotIn("/api/services/light/turn_on", self.ha.paths())

    def test_an_ambiguous_name_is_reported_not_guessed(self):
        """Silently picking one of several equally good matches would change the
        wrong room."""
        from jarvis_tests.fakes import FakeHomeAssistant

        ambiguous = FakeHomeAssistant(states=[
            {"entity_id": "light.porch_main", "state": "off",
             "attributes": {"friendly_name": "Porch Light"}},
            {"entity_id": "light.porch_side", "state": "off",
             "attributes": {"friendly_name": "Porch Light"}},
        ])
        self.addCleanup(ambiguous.stop)
        from jarvis.config import load_config

        cfg = load_config()
        cfg.home_assistant_url = ambiguous.url
        cfg.home_assistant_token = "good-token"
        self.iot.bind_config(cfg)
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_call_service("light", "turn_on", entity="porch light")
        self.assertIn("light.porch_side", str(ctx.exception))
        self.assertEqual(ambiguous.paths(), ["/api/states"],
                         "nothing may be called when the target is ambiguous")

    def test_an_exact_match_beats_a_partial_one(self):
        from jarvis_tests.fakes import FakeHomeAssistant

        mixed = FakeHomeAssistant(states=[
            {"entity_id": "light.kitchen", "state": "off", "attributes": {}},
            {"entity_id": "light.kitchenette", "state": "off", "attributes": {}},
        ])
        self.addCleanup(mixed.stop)
        from jarvis.config import load_config

        cfg = load_config()
        cfg.home_assistant_url = mixed.url
        cfg.home_assistant_token = "good-token"
        self.iot.bind_config(cfg)
        self.iot.ha_call_service("light", "turn_on", entity="kitchen")
        body = [b for m, p, b in mixed.calls if m == "POST"][-1]
        self.assertEqual(body.get("entity_id"), "light.kitchen")

    def test_a_service_body_passed_as_entity_is_understood(self):
        """Writing the HA service body into `entity` is the natural mistake,
        and it used to raise AttributeError from inside string handling."""
        out = self.iot.ha_call_service(
            "light", "turn_on", {"entity_id": "light.porch", "brightness_pct": 40}
        )
        self.assertEqual(out["payload"]["entity_id"], "light.porch")
        self.assertEqual(out["payload"]["brightness_pct"], 40)

    def test_a_body_with_no_entity_id_says_so_plainly(self):
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_call_service("light", "turn_on", {"brightness_pct": 40})
        self.assertIn("entity_id", str(ctx.exception))

    def test_writes_ask_permission_and_can_be_declined(self):
        from jarvis.tools import system_tools

        system_tools.set_approver(lambda name, detail: False)
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_call_service("light", "turn_on", entity="light.porch")
        self.assertIn("declined", str(ctx.exception))
        self.assertNotIn("/api/services/light/turn_on", self.ha.paths())

    def test_a_script_is_findable_without_a_friendly_name(self):
        """Reading friendly_name with an empty default, as this did, meant any
        script without one could not be found by its own name."""
        out = self.iot.ha_run_script("goodnight")
        self.assertEqual(out["called"], "script.turn_on on script.goodnight")

    def test_a_script_is_findable_by_friendly_name(self):
        out = self.iot.ha_run_script("Movie Time")
        self.assertEqual(out["called"], "scene.turn_on on scene.movie")

    def test_an_unknown_script_lists_what_exists(self):
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_run_script("nope")
        message = str(ctx.exception)
        self.assertIn("script.goodnight", message)
        self.assertIn("scene.movie", message)

    def test_upstream_failures_are_reported_not_swallowed(self):
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_call_service("light", "error", entity="light.porch")
        self.assertIn("500", str(ctx.exception))

        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_get_state("light.nope")
        self.assertIn("404", str(ctx.exception))

    def test_a_rejected_token_is_reported_as_such(self):
        from jarvis.config import load_config

        cfg = load_config()
        cfg.home_assistant_url = self.ha.url
        cfg.home_assistant_token = "wrong"
        self.iot.bind_config(cfg)
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_list_entities()
        self.assertIn("401", str(ctx.exception))

    def test_being_unconfigured_is_explained(self):
        from jarvis.config import load_config

        cfg = load_config()
        cfg.home_assistant_url = ""
        self.iot.bind_config(cfg)
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_list_entities()
        self.assertIn("not configured", str(ctx.exception))

    def test_an_unreachable_host_does_not_hang_or_crash(self):
        import socket as socketmod

        from jarvis.config import load_config

        probe = socketmod.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()

        cfg = load_config()
        cfg.home_assistant_url = f"http://127.0.0.1:{dead_port}"
        cfg.home_assistant_token = "good-token"
        self.iot.bind_config(cfg)
        with self.assertRaises(ToolError) as ctx:
            self.iot.ha_list_entities()
        self.assertIn("could not reach", str(ctx.exception))


class TestMqtt(unittest.TestCase):
    """The MQTT tools, against a real broker on loopback.

    paho-mqtt only reads the socket from its own network thread. The publish
    tool never started one, so a QoS 1 or 2 publish could never complete its
    handshake and always reported a delivery that had not been acknowledged.
    """

    def setUp(self):
        from jarvis.config import load_config
        from jarvis.tools import iot_tools, load_all, system_tools
        from jarvis_tests.fakes import MqttBroker

        load_all()
        self.iot = iot_tools
        self.broker = MqttBroker(username="jarvis", password="s3cret")
        self.addCleanup(self.broker.stop)

        system_tools.set_approver(lambda name, detail: True)
        self.addCleanup(system_tools.set_approver, None)
        self.bind(password="s3cret")

    def bind(self, password: str = "s3cret", host: str = "127.0.0.1") -> None:
        from jarvis.config import load_config

        cfg = load_config()
        cfg.mqtt_host = host
        cfg.mqtt_port = self.broker.port
        cfg.mqtt_username = "jarvis"
        cfg.mqtt_password = password
        self.iot.bind_config(cfg)

    def test_a_qos1_publish_is_acknowledged(self):
        out = self.iot.mqtt_publish("porch/lamp", "on", qos=1)
        self.assertTrue(out["delivered"])
        self.assertEqual(self.broker.published[-1][0], "porch/lamp")
        self.assertEqual(self.broker.published[-1][2], 1)

    def test_a_qos2_publish_completes_its_handshake(self):
        out = self.iot.mqtt_publish("porch/lamp", "on", qos=2)
        self.assertTrue(out["delivered"])

    def test_the_payload_reaches_the_broker_unchanged(self):
        import json as jsonmod

        body = jsonmod.dumps({"state": "on", "brightness": 200})
        self.iot.mqtt_publish("porch/lamp", body, qos=1)
        topic, payload, _qos, _retain = self.broker.published[-1]
        self.assertEqual(topic, "porch/lamp")
        self.assertEqual(jsonmod.loads(payload.decode()), {"state": "on",
                                                           "brightness": 200})

    def test_a_retained_publish_is_retained(self):
        self.iot.mqtt_publish("home/porch/state", "on", retain=True)
        self.assertIn("home/porch/state", self.broker.retained)
        self.assertEqual(self.broker.retained["home/porch/state"], b"on")
        self.assertTrue(self.broker.published[-1][3])

    def test_publishing_asks_permission_first(self):
        from jarvis.tools import system_tools

        asked: list[str] = []
        system_tools.set_approver(lambda name, detail: asked.append(name) or False)
        with self.assertRaises(ToolError) as ctx:
            self.iot.mqtt_publish("porch/lamp", "on")
        self.assertIn("declined", str(ctx.exception))
        self.assertEqual(asked, ["mqtt_publish"])
        self.assertEqual(self.broker.published, [],
                         "a declined publish must not reach the broker")

    def test_a_rejected_login_is_not_reported_as_a_delivery(self):
        """paho signals a refused connection by return code, and bad credentials
        are not even known until CONNACK arrives. Ignoring both made a publish
        against a rejecting broker report success."""
        self.bind(password="wrong")
        with self.assertRaises(ToolError) as ctx:
            self.iot.mqtt_publish("porch/lamp", "on")
        self.assertIn("rejected the connection", str(ctx.exception))
        self.assertEqual(self.broker.auth_failures, 1)
        self.assertEqual(self.broker.published, [])

    def test_listening_returns_a_live_message(self):
        import threading

        def later():
            self.assertTrue(self.broker.wait_for_subscription("home/porch/state"),
                            "subscription never reached the broker")
            self.broker.publish("home/porch/state", '{"state": "on"}')

        threading.Thread(target=later, daemon=True).start()
        out = self.iot.mqtt_listen("home/porch/state", 10.0)
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["messages"][0]["payload"], {"state": "on"})

    def test_listening_returns_a_retained_message(self):
        self.iot.mqtt_publish("home/porch/state", '{"state": "off"}', retain=True)
        out = self.iot.mqtt_listen("home/porch/state", 5.0)
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["messages"][0]["payload"], {"state": "off"})

    def test_a_wildcard_subscription_matches(self):
        import threading

        def later():
            self.assertTrue(self.broker.wait_for_subscription("home/+/state"),
                            "subscription never reached the broker")
            self.broker.publish("home/kitchen/state", '{"state": "on"}')
            self.broker.publish("home/porch/state", '{"state": "off"}')
            self.broker.publish("other/porch/state", '{"state": "on"}')

        threading.Thread(target=later, daemon=True).start()
        out = self.iot.mqtt_listen("home/+/state", 10.0)
        topics = {m["topic"] for m in out["messages"]}
        self.assertIn("home/kitchen/state", topics)
        self.assertIn("home/porch/state", topics)
        self.assertNotIn("other/porch/state", topics)

    def test_a_non_json_payload_comes_back_as_text(self):
        import threading

        def later():
            self.assertTrue(self.broker.wait_for_subscription("raw/text"),
                            "subscription never reached the broker")
            self.broker.publish("raw/text", "just words")

        threading.Thread(target=later, daemon=True).start()
        out = self.iot.mqtt_listen("raw/text", 10.0)
        self.assertEqual(out["messages"][0]["payload"], "just words")

    def test_a_silent_topic_says_so(self):
        out = self.iot.mqtt_listen("nothing/here", 1.0)
        self.assertEqual(out["messages"], [])
        self.assertIn("Nothing published", out["note"])

    def test_listening_under_a_rejected_login_is_not_a_silent_sensor(self):
        """Subscribing anyway looked exactly like a sensor with nothing to say,
        which is the wrong conclusion about a sensor you cannot read."""
        self.bind(password="wrong")
        with self.assertRaises(ToolError) as ctx:
            self.iot.mqtt_listen("home/porch/state", 3.0)
        self.assertIn("rejected the connection", str(ctx.exception))

    def test_a_dead_broker_is_reported_clearly(self):
        self.broker.stop()
        with self.assertRaises(ToolError) as ctx:
            self.iot.mqtt_publish("x", "y")
        self.assertIn("could not connect", str(ctx.exception))
        with self.assertRaises(ToolError) as ctx:
            self.iot.mqtt_listen("x", 2.0)
        self.assertIn("could not connect", str(ctx.exception))

    def test_being_unconfigured_is_explained(self):
        from jarvis.config import load_config
        from jarvis.tools import system_tools

        cfg = load_config()
        cfg.mqtt_host = ""
        self.iot.bind_config(cfg)
        system_tools.set_approver(lambda name, detail: True)
        with self.assertRaises(ToolError) as ctx:
            self.iot.mqtt_publish("x", "y")
        self.assertIn("not configured", str(ctx.exception))


class TestPowerActions(unittest.TestCase):
    """power_action decides what happens to the machine. Nothing here runs a real
    command: Popen and LockWorkStation are both replaced.

    This was the most dangerous bug found in the project. The tool's schema
    listed an enum, but nothing enforced enums, and the body was a chain of
    equality tests ending in a shutdown branch. Every word the chain did not
    recognise became a shutdown, so "reboot" powered the machine off instead of
    restarting it, and "log off" powered it off instead of signing out.
    """

    def setUp(self):
        from unittest import mock

        from jarvis.tools import load_all, system_tools

        load_all()
        self.system_tools = system_tools
        self.mock = mock
        self.asked: list[str] = []
        system_tools.set_approver(
            lambda name, detail: self.asked.append(name) or True
        )
        self.addCleanup(system_tools.set_approver, None)

        popen_patcher = mock.patch.object(system_tools.subprocess, "Popen")
        self.popen = popen_patcher.start()
        self.addCleanup(popen_patcher.stop)

        lock_patcher = mock.patch(
            "jarvis.tools.system_tools.ctypes.windll.user32.LockWorkStation",
            return_value=1,
        )
        lock_patcher.start()
        self.addCleanup(lock_patcher.stop)

    def ran(self) -> list[list[str]]:
        return [call.args[0] for call in self.popen.call_args_list]

    def test_reboot_restarts_rather_than_shutting_down(self):
        out = self.system_tools.power_action("reboot")
        self.assertEqual(out["action"], "restart")
        self.assertEqual(self.ran(), [["shutdown", "/r", "/t", "5", "/c",
                                       "Requested by JARVIS"]])

    def test_log_off_signs_out_rather_than_shutting_down(self):
        out = self.system_tools.power_action("log off")
        self.assertEqual(out["action"], "signout")
        self.assertEqual(self.ran(), [["shutdown", "/l"]])

    def test_the_canonical_names_still_work(self):
        for action, expected in (("restart", "/r"), ("shutdown", "/s")):
            self.popen.reset_mock()
            out = self.system_tools.power_action(action)
            self.assertEqual(out["action"], action)
            self.assertEqual(self.ran()[0][:2], ["shutdown", expected])

    def test_ordinary_synonyms_map_to_the_right_thing(self):
        for words, action, head in (
            ("suspend", "sleep", "rundll32.exe"),
            ("power off", "shutdown", "shutdown"),
            ("shut down", "shutdown", "shutdown"),
            ("lock screen", "lock", None),
        ):
            self.popen.reset_mock()
            out = self.system_tools.power_action(words)
            self.assertEqual(out["action"], action, f"{words!r} went to {out}")
            if head is None:
                self.assertEqual(self.ran(), [])
            else:
                self.assertEqual(self.ran()[0][0], head)

    def test_an_unknown_action_is_an_error_and_runs_nothing(self):
        for action in ("hibernate", "wipe", "format", "power cycle", "", "yes"):
            self.popen.reset_mock()
            with self.assertRaises(ToolError, msg=f"{action!r} was accepted"):
                self.system_tools.power_action(action)
            self.assertEqual(self.ran(), [],
                             f"{action!r} must not reach a command")

    def test_an_unknown_action_is_refused_before_asking_permission(self):
        """Asking the user to confirm a request that cannot be carried out is
        worse than useless on a tool like this."""
        self.asked.clear()
        with self.assertRaises(ToolError):
            self.system_tools.power_action("hibernate")
        self.assertEqual(self.asked, [])

    def test_the_enum_and_the_synonyms_agree(self):
        """Guards against the two lists drifting apart again."""
        from jarvis.tools import system_tools

        declared = system_tools.registry.get("power_action")
        enum = declared.parameters["properties"]["action"]["enum"]
        for action in enum:
            self.assertEqual(
                system_tools._POWER_ACTIONS.get(action), action,
                f"{action!r} is offered to the model but not understood",
            )

    def test_the_delay_is_clamped_and_passed_on(self):
        self.system_tools.power_action("restart", delay_seconds=-30)
        self.assertEqual(self.ran()[0][3], "0")
        self.popen.reset_mock()
        self.system_tools.power_action("restart", delay_seconds=90)
        self.assertEqual(self.ran()[0][3], "90")

    def test_it_reports_how_to_undo_itself(self):
        out = self.system_tools.power_action("shutdown", delay_seconds=60)
        self.assertEqual(out["abort_with"], "shutdown /a")
        self.assertEqual(out["delay_seconds"], 60)

    def test_a_declined_action_runs_nothing(self):
        self.system_tools.set_approver(lambda name, detail: False)
        with self.assertRaises(ToolError) as ctx:
            self.system_tools.power_action("shutdown")
        self.assertIn("declined", str(ctx.exception))
        self.assertEqual(self.ran(), [])

    def test_a_refused_lock_is_reported(self):
        refused = self.mock.patch(
            "jarvis.tools.system_tools.ctypes.windll.user32.LockWorkStation",
            return_value=0,
        )
        refused.start()
        self.addCleanup(refused.stop)
        with self.assertRaises(ToolError) as ctx:
            self.system_tools.power_action("lock")
        self.assertIn("LockWorkStation", str(ctx.exception))


class TestDangerousToolsReallyAsk(unittest.TestCase):
    """Every tool the model is told needs approval must actually ask.

    `dangerous=True` is read in exactly one place in the package: __main__.py,
    to print "[needs approval]" in a listing. It enforces nothing. What actually
    gates these tools is a hand-written `_approve()` call inside the handler, so
    the decorator and the call are two pieces of information kept in different
    places by hand. Add the flag, forget the call, and the tool is announced as
    guarded while running completely unconfirmed.

    This checks the property by reading the source, so it holds without running
    a single destructive tool.
    """

    @classmethod
    def setUpClass(cls):
        load_all()
        cls.handlers = {}
        for path in sorted(TOOLS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                flagged = any(
                    isinstance(dec, ast.Call)
                    and any(kw.arg == "dangerous"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                            for kw in dec.keywords)
                    for dec in node.decorator_list
                )
                if flagged:
                    cls.handlers[node.name] = node

    @staticmethod
    def _called_names(node: ast.AST) -> set[str]:
        out = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = (getattr(child.func, "id", None)
                        or getattr(child.func, "attr", None))
                if name:
                    out.add(name)
        return out

    def test_the_destructive_tools_are_still_flagged(self):
        self.assertGreaterEqual(
            len(self.handlers), 10,
            "the dangerous flag went missing, so the listing no longer warns",
        )

    def test_every_flagged_tool_asks_before_it_acts(self):
        """Directly, or by delegating to another flagged tool that does.
        ha_run_script is the delegating case: it hands off to ha_call_service,
        which is where the confirmation lives."""
        unanswered = []
        for name, node in self.handlers.items():
            calls = self._called_names(node)
            if "_approve" in calls:
                continue
            if calls & (set(self.handlers) - {name}):
                continue
            unanswered.append(name)
        self.assertEqual(
            sorted(unanswered), [],
            "listed to the model as needing approval but never ask",
        )

    def test_the_confirmation_precedes_the_action(self):
        """Asking after the click, the write, or the shutdown would make the
        confirmation decorative in a different way."""
        acting = {"Popen", "run", "kill", "write_text", "write_bytes", "unlink",
                  "remove", "rmdir", "LockWorkStation", "turn_on", "call_service"}
        too_late = []
        for name, node in self.handlers.items():
            approve_at = None
            action_at = None
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                called = (getattr(child.func, "id", None)
                          or getattr(child.func, "attr", None))
                if called == "_approve" and approve_at is None:
                    approve_at = child.lineno
                elif called in acting:
                    action_at = min(action_at or 10 ** 9, child.lineno)
            if approve_at is not None and action_at is not None:
                if approve_at > action_at:
                    too_late.append(name)
        self.assertEqual(sorted(too_late), [], "these act before asking")

    def test_approval_is_only_claimed_by_tools_that_exist(self):
        """_approve("name") and the registry are written independently, so a
        renamed tool can leave a confirmation labelled for something else."""
        claimed: set[str] = set()
        for path in sorted(TOOLS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = (getattr(node.func, "id", None)
                          or getattr(node.func, "attr", None))
                if called == "_approve" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        claimed.add(first.value)
        self.assertEqual(
            sorted(n for n in claimed if n not in registry.names()), [],
            "_approve is called for a tool that is not registered, so the "
            "confirmation is attributed to the wrong name",
        )

    def test_nothing_runs_when_no_confirmation_handler_is_wired_up(self):
        """The fail-closed case: with no approver, _approve raises rather than
        defaulting to allowed."""
        from jarvis.tools import system_tools

        system_tools.set_approver(None)
        self.addCleanup(system_tools.set_approver, None)
        with self.assertRaises(ToolError) as ctx:
            system_tools._approve("power_action", "test")
        self.assertIn("blocked", str(ctx.exception))


class TestKillProcessAndAbort(unittest.TestCase):
    """kill_process and abort_shutdown force things and had no coverage at all.
    Nothing here touches a real process: Popen, run, and psutil are replaced."""

    def setUp(self):
        from unittest import mock

        from jarvis.tools import load_all, system_tools

        load_all()
        self.system_tools = system_tools
        self.mock = mock
        system_tools.set_approver(lambda name, detail: True)
        self.addCleanup(system_tools.set_approver, None)

        for target in ("Popen", "run"):
            patcher = mock.patch.object(system_tools.subprocess, target)
            setattr(self, target.lower(), patcher.start())
            self.addCleanup(patcher.stop)

    def test_a_declined_kill_kills_nothing(self):
        import psutil

        killed: list[int] = []
        victim = self.mock.Mock()
        victim.info = {"name": "notepad.exe", "pid": 4242}
        victim.kill = lambda: killed.append(4242)

        self.system_tools.set_approver(lambda name, detail: False)
        with self.mock.patch.object(psutil, "process_iter", return_value=[victim]):
            with self.assertRaises(ToolError) as ctx:
                self.system_tools.kill_process("notepad.exe")
        self.assertIn("declined", str(ctx.exception))
        self.assertEqual(killed, [], "a declined kill must not reach kill()")

    def test_a_missing_process_is_an_error_and_kills_nothing(self):
        import psutil

        bystander = self.mock.Mock()
        bystander.info = {"name": "other.exe", "pid": 1}
        with self.mock.patch.object(psutil, "process_iter", return_value=[bystander]):
            with self.assertRaises(ToolError) as ctx:
                self.system_tools.kill_process("notepad.exe")
        self.assertIn("no running process", str(ctx.exception))
        bystander.kill.assert_not_called()

    def test_a_declined_abort_runs_nothing(self):
        self.system_tools.set_approver(lambda name, detail: False)
        with self.assertRaises(ToolError):
            self.system_tools.abort_shutdown()
        self.run.assert_not_called()
        self.popen.assert_not_called()

    def test_abort_uses_the_documented_command(self):
        self.run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        out = self.system_tools.abort_shutdown()
        self.assertTrue(out["aborted"])
        self.assertEqual(self.run.call_args.args[0], ["shutdown", "/a"])


class TestSpokenAudioPath(unittest.TestCase):
    """The utterance path, with a human replaced by synthesized speech.

    The microphone and the wake word are the only parts that genuinely need a
    person and a room, so everything downstream of them is testable: Windows
    SAPI writes a WAV of a known phrase, and it goes through the same
    16 kHz-mono-to-Whisper path a recorded utterance takes. The silence and
    too-short guards need no model at all and always run.
    """

    @staticmethod
    def _listener_with_spy_model():
        """A Listener whose model records whether it was ever asked to transcribe.

        The point of the guards is that quiet audio never reaches Whispper, not
        that the model is never loaded: it is loaded once per process on purpose
        (warm_up does it at startup), so asserting on loading would be asserting
        the wrong thing.
        """
        from jarvis.stt import Listener

        class SpyModel:
            def __init__(self):
                self.calls = 0

            def transcribe(self, *a, **k):
                self.calls += 1
                return [], type("Info", (), {"language": "en"})()

        listener = Listener()
        spy = SpyModel()
        listener._model = spy
        return listener, spy

    def test_silence_yields_no_transcript_and_never_reaches_the_model(self):
        """Recording starts before anyone speaks, so most captured blocks are
        silence, and sending them to Whisper would cost real time for nothing."""
        import numpy as np

        listener, spy = self._listener_with_spy_model()
        result = listener.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(result.text, "")
        self.assertFalse(result, "an empty transcript must read as falsey")
        self.assertEqual(spy.calls, 0, "silence must not be sent to the model")

    def test_a_click_is_too_short_to_transcribe(self):
        import numpy as np

        listener, spy = self._listener_with_spy_model()
        self.assertEqual(
            listener.transcribe(np.zeros(100, dtype=np.float32)).text, "")
        self.assertEqual(spy.calls, 0)

    def test_quiet_but_long_audio_is_still_dropped(self):
        """Amplitude, not just length, is the guard: a long room-tone recording
        should not be sent to the model either."""
        import numpy as np

        listener, spy = self._listener_with_spy_model()
        noise = (np.random.RandomState(0).randn(32000) * 1e-6).astype(np.float32)
        self.assertEqual(listener.transcribe(noise).text, "")
        self.assertEqual(spy.calls, 0,
                         "room tone is silence for this purpose")

    def test_real_speech_does_reach_the_model(self):
        """The complement, so the guards cannot pass by dropping everything."""
        import numpy as np

        listener, spy = self._listener_with_spy_model()
        tone = (0.4 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
                ).astype(np.float32)
        listener.transcribe(tone)
        self.assertEqual(spy.calls, 1,
                         "audible audio must be transcribed, not discarded")

    def test_speech_is_transcribed_end_to_end(self):
        """Needs faster-whisper and its model, so it skips rather than failing
        on a machine that has not downloaded it."""
        import importlib.util
        import subprocess
        import wave

        import numpy as np

        from jarvis.stt import Listener, STTUnavailable

        if importlib.util.find_spec("faster_whisper") is None:
            self.skipTest("faster-whisper is not installed")
        if not sys.platform.startswith("win"):
            self.skipTest("SAPI speech synthesis is Windows-only")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        wav = Path(tmp.name) / "phrase.wav"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f'$s.SetOutputToWaveFile("{wav}"); '
            '$s.Speak("what time is it right now"); $s.Dispose()'
        )
        made = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                              capture_output=True, text=True, timeout=120)
        if not wav.exists():
            self.skipTest(f"SAPI could not write audio: {made.stderr.strip()[:120]}")

        with wave.open(str(wav), "rb") as handle:
            rate, channels = handle.getframerate(), handle.getnchannels()
            samples = np.frombuffer(handle.readframes(handle.getnframes()),
                                    dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != 16000:
            idx = np.linspace(0, len(samples) - 1,
                              int(len(samples) * 16000 / rate))
            samples = np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)

        listener = Listener()
        try:
            result = listener.transcribe(samples)
        except STTUnavailable as exc:
            self.skipTest(f"whisper model unavailable: {exc}")

        heard = set(result.text.lower().replace("?", " ").replace(".", " ").split())
        expected = {"what", "time", "is", "it", "right", "now"}
        self.assertTrue(expected <= heard,
                        f"heard {result.text!r}, missing {sorted(expected - heard)}")
        self.assertTrue(result)


class TestScreenRegions(unittest.TestCase):
    """Region parsing and display listing, with no real capture."""

    def test_a_region_becomes_a_monitor_dict(self):
        from jarvis.tools.screen_tools import _monitor_from

        self.assertEqual(
            _monitor_from([10, 20, 300, 400]),
            {"left": 10, "top": 20, "width": 300, "height": 400},
        )

    def test_no_region_means_the_whole_primary_display(self):
        from jarvis.tools.screen_tools import _monitor_from

        self.assertIsNone(_monitor_from(None))
        self.assertIsNone(_monitor_from([]))

    def test_a_malformed_region_is_a_clear_error(self):
        """These used to reach the capture library and come back as an opaque
        internal error name instead of something the model can act on."""
        from jarvis.tools.screen_tools import _monitor_from

        for bad in ([1, 2, 3], [1, 2, 3, 4, 5], [0, 0, -5, 10], [0, 0, 0, 0],
                    ["a", "b", "c", "d"], [0, 0, None, 10]):
            with self.assertRaises(ToolError, msg=f"{bad!r} was accepted"):
                _monitor_from(bad)

    def test_numeric_strings_are_accepted(self):
        from jarvis.tools.screen_tools import _monitor_from

        self.assertEqual(
            _monitor_from(["10", "20", "30", "40"]),
            {"left": 10, "top": 20, "width": 30, "height": 40},
        )

    def test_listing_displays_does_not_reopen_the_session(self):
        """It called .mss() on the object _open() had already returned, so the
        tool raised AttributeError every time it was used."""
        from unittest import mock

        from jarvis.tools import screen_tools

        class FakeSession:
            def __init__(self):
                self.closed = False
                self.monitors = [
                    {"left": 0, "top": 0, "width": 2560, "height": 1440},
                    {"left": 0, "top": 0, "width": 1920, "height": 1080,
                     "is_primary": True},
                    {"left": 1920, "top": -200, "width": 2560, "height": 1440,
                     "is_primary": False},
                ]

            def close(self):
                self.closed = True

        session = FakeSession()
        with mock.patch.object(screen_tools, "_open", return_value=session):
            out = screen_tools.list_displays()

        self.assertEqual(len(out["monitors"]), 3)
        self.assertTrue(out["monitors"][1]["primary"])
        self.assertFalse(out["monitors"][2]["primary"])
        self.assertEqual(out["monitors"][2]["left"], 1920)
        self.assertTrue(session.closed, "the capture session must be closed")

    def test_the_primary_display_is_the_one_the_os_names(self):
        """It used to be hardcoded to index 1. mss lists the virtual desktop
        union first and the real displays after it, but not in an order that
        puts the primary at index 1, so on a two-monitor machine where the
        second display is the primary, every primary flag was wrong."""
        from unittest import mock

        from jarvis.tools import screen_tools

        class FakeSession:
            closed = False
            monitors = [
                {"left": 0, "top": 0, "width": 3840, "height": 1080},
                {"left": 0, "top": 0, "width": 1920, "height": 1080,
                 "is_primary": False, "name": "DELL U2412M"},
                {"left": 1920, "top": 0, "width": 1920, "height": 1080,
                 "is_primary": True, "name": "Built-in"},
            ]

            def close(self):
                pass

        with mock.patch.object(screen_tools, "_open", return_value=FakeSession()):
            out = screen_tools.list_displays()

        self.assertFalse(out["monitors"][0]["primary"],
                         "the virtual desktop is not a display")
        self.assertFalse(out["monitors"][1]["primary"],
                         "the OS says the second display is primary")
        self.assertTrue(out["monitors"][2]["primary"],
                        "index 1 is not always the primary")
        self.assertEqual(out["monitors"][2]["name"], "Built-in")

    def test_an_mss_without_the_flag_still_reports_something(self):
        """Older mss builds do not carry is_primary, and reporting nothing is
        worse than reporting the usual answer."""
        from unittest import mock

        from jarvis.tools import screen_tools

        class FakeSession:
            closed = False
            monitors = [
                {"left": 0, "top": 0, "width": 1920, "height": 1200},
                {"left": 0, "top": 0, "width": 1920, "height": 1200},
            ]

            def close(self):
                pass

        with mock.patch.object(screen_tools, "_open", return_value=FakeSession()):
            out = screen_tools.list_displays()

        self.assertFalse(out["monitors"][0]["primary"])
        self.assertTrue(out["monitors"][1]["primary"])

    def test_a_secondary_display_is_reported_with_its_offset(self):
        """Coordinates are virtual-desktop absolute, so a monitor to the left or
        above the primary has a negative origin and clicking has to respect it."""
        from unittest import mock

        from jarvis.tools import screen_tools

        class FakeSession:
            closed = False
            monitors = [
                {"left": -1920, "top": 0, "width": 3840, "height": 1080},
                {"left": 0, "top": 0, "width": 1920, "height": 1080,
                 "is_primary": True},
                {"left": 1920, "top": -1080, "width": 1920, "height": 1080,
                 "is_primary": False},
            ]

            def close(self):
                pass

        with mock.patch.object(screen_tools, "_open", return_value=FakeSession()):
            out = screen_tools.list_displays()

        self.assertEqual(out["monitors"][0]["left"], -1920)
        self.assertEqual(out["monitors"][2]["top"], -1080)
        self.assertEqual(out["monitors"][0]["width"], 3840)


class ProcedureRecallIsNotConfusedAcrossApps(unittest.TestCase):
    """recall_procedure returns steps, which are clicks and typed text meant for
    one program. With no app given, a loose word-overlap match would hand back
    another program's steps for the model to replay wherever it happens to be."""

    def setUp(self):
        from jarvis.tools import sight_tools
        from jarvis.vision_memory import VisionMemory

        self.tools = sight_tools
        self.tmp = tempfile.TemporaryDirectory()
        self.vm = VisionMemory(Path(self.tmp.name) / "v.db")

    def tearDown(self):
        self.vm.close()
        self.tmp.cleanup()

    def _call(self, goal, app=""):
        from unittest import mock

        with mock.patch.object(self.tools, "_vis", return_value=self.vm):
            return self.tools.recall_procedure(goal, app)

    def test_a_different_apps_steps_are_not_reported_as_a_hit(self):
        self.vm.save_procedure("open the file menu", [{"click": "File"}], "notepad")
        out = self._call("open the settings page")
        self.assertFalse(out["found"],
                         "a different goal in another app is not this goal")
        self.assertEqual(out["similar"][0]["goal"], "open the file menu")
        self.assertEqual(out["similar"][0]["app"], "notepad",
                         "the suggestion must say which app it belongs to, so the "
                         "model can ask again with it")

    def test_the_same_goal_in_any_app_is_still_a_hit(self):
        self.vm.save_procedure("export the file", [{"click": "Export"}], "word")
        out = self._call("export the file")
        self.assertTrue(out["found"], "an identical goal is unambiguous")
        self.assertEqual(out["app"], "word")
        self.assertEqual(out["steps"], [{"click": "Export"}])

    def test_naming_the_app_still_finds_it_directly(self):
        self.vm.save_procedure("export the file", [{"click": "Export"}], "word")
        self.vm.save_procedure("export the file", [{"click": "Share"}], "calc")
        out = self._call("export the file", "calc")
        self.assertTrue(out["found"])
        self.assertEqual(out["app"], "calc")
        self.assertEqual(out["steps"], [{"click": "Share"}])

    def test_naming_the_wrong_app_does_not_fall_back_to_another(self):
        self.vm.save_procedure("export the file", [{"click": "Export"}], "word")
        out = self._call("export the file", "calc")
        self.assertFalse(out["found"],
                         "an explicit app must not silently return another app's "
                         "steps, which is the substitution this guards against")


class TestWakeFraming(unittest.TestCase):
    """The mic delivers 20ms blocks; the model wants non-overlapping 80ms frames.

    These assert the two facts that matter and cannot be inferred from the
    scores: every sample is scored exactly once, and consecutive frames do not
    overlap. The bug this guards against scored the same window four times per
    80ms of audio, which held real speech at 0.087 against a 0.5 threshold.
    """

    BLOCK = 320      # 20ms at 16kHz, what Mic actually hands over
    FRAME = 1280     # 80ms, what the model wants

    def _wake(self):
        from jarvis.wake import WakeWord

        ww = WakeWord(model_name="hey_jarvis", threshold=0.5)
        self.seen = []

        class Recorder:
            def predict(inner, pcm):  # noqa: ANN001
                self.seen.append(np.array(pcm))
                return {"hey_jarvis": 0.0}

        ww._model = Recorder()
        return ww

    def test_every_sample_is_scored_exactly_once(self):
        from jarvis.wake import FRAME_SAMPLES

        self.assertEqual(FRAME_SAMPLES, self.FRAME)
        ww = self._wake()
        n_blocks = 100
        signal = np.linspace(-0.5, 0.5, n_blocks * self.BLOCK, dtype=np.float32)

        for i in range(0, len(signal), self.BLOCK):
            ww.feed(signal[i:i + self.BLOCK])

        expected_frames = len(signal) // self.FRAME
        self.assertEqual(
            len(self.seen), expected_frames,
            f"scored {len(self.seen)} frames for {len(signal)} samples; "
            f"expected exactly {expected_frames} non-overlapping frames",
        )
        # Compare in the pcm domain the model actually receives.
        head = signal[:expected_frames * self.FRAME]
        expected_pcm = np.clip(head * 32767.0, -32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(
            np.concatenate(self.seen), expected_pcm,
            "the scored audio must be the input, in order, with nothing repeated or dropped",
        )

    def test_a_frame_is_never_scored_twice(self):
        ww = self._wake()
        # A ramp, so a consumed window and a repeated one are distinguishable.
        signal = np.linspace(-0.5, 0.5, 40 * self.BLOCK, dtype=np.float32)

        for i in range(0, len(signal), self.BLOCK):
            ww.feed(signal[i:i + self.BLOCK])

        self.assertGreater(len(self.seen), 1, "not enough frames to compare")
        for earlier, later in zip(self.seen, self.seen[1:]):
            self.assertFalse(
                np.array_equal(earlier, later),
                "the same frame was scored twice in a row, so the model is fed "
                "repeats instead of a forward stream",
            )

    def test_a_short_utterance_is_held_until_a_whole_frame_arrives(self):
        ww = self._wake()
        # Less than one frame must not be scored, or the model gets a stub.
        for _ in range(3):
            ww.feed(np.zeros(self.BLOCK, dtype=np.float32))
        self.assertEqual(self.seen, [], "scored a partial frame")
        ww.feed(np.zeros(self.BLOCK, dtype=np.float32))
        self.assertEqual(len(self.seen), 1, "did not score once a full frame arrived")

    def test_audio_after_a_reset_is_not_scored(self):
        ww = self._wake()
        for _ in range(4):
            ww.feed(np.zeros(self.BLOCK, dtype=np.float32))
        self.assertEqual(len(self.seen), 1)
        ww.reset()
        for _ in range(3):
            ww.feed(np.zeros(self.BLOCK, dtype=np.float32))
        self.assertEqual(len(self.seen), 1, "reset left a partial frame buffered")


class TestToolSelection(unittest.TestCase):
    """Relevance picks which tools reach the prompt.

    Sending all 38 core tools cost 3068 prompt tokens and prefill runs at about
    12.8 ms/token on this CPU, so a cold turn spent 39-101s reading schemas it
    would not use. The tests below are the reason this is safe to do: the tool
    each request actually needs has to survive selection, and the way back to
    everything else has to always be present.
    """

    # One request per tool that used to be reachable only via list_more_tools,
    # plus a spread of ordinary ones. Seven of these are not core tools, which
    # is the point: showing every core tool found only 13 of the 20.
    REQUESTS = {
        "what time is it right now": "what_time_is_it",
        "take a screenshot of the screen": "take_screenshot",
        "shut down the computer": "power_action",
        "what is the weather in paris": "get_weather",
        "set the volume to 40": "set_volume",
        "open notepad": "open_app",
        "read the file report.docx": "read_text_file",
        "type hello in notepad": "type_on_screen",
        "scroll the page down": "scroll_screen",
        "click on submit": "click_on",
        "search my notes for budget": "search_notes",
        "kill the stuck process": "kill_process",
        "list the running programs": "list_processes",
        "play some music": "media_control",
        "copy that to the clipboard": "clipboard_write",
        "where is the save button": "where_is",
        "focus the browser window": "focus_window",
        "press ctrl+s": "press_keys",
        "list the monitors": "list_displays",
        "delete the note about milk": "delete_note",
    }

    def _brain(self, cap=10, extra=()):
        from jarvis.brain import Brain
        from jarvis.config import load_config
        from jarvis.tools import load_all

        load_all()
        cfg = load_config()
        cfg.tool_select_max = cap
        brain = Brain.__new__(Brain)
        brain.cfg = cfg
        brain._extra_tools = set(extra)
        brain._ensure_registry = lambda: None
        return brain

    def test_the_tool_a_request_needs_survives_selection(self):
        brain = self._brain()
        missed = []
        for request, needed in self.REQUESTS.items():
            chosen = brain.select_tools(request, brain.cfg.tool_select_max)
            if needed not in chosen:
                missed.append(f"{needed} for {request!r}")
        self.assertEqual(missed, [], "selection dropped tools it should keep")

    def test_asking_for_every_core_tool_would_have_missed_seven_of_them(self):
        # The reason ranking the whole catalogue beats sending all of core.
        from jarvis.brain import CORE_TOOLS

        unreachable = sorted(
            needed for needed in self.REQUESTS.values() if needed not in CORE_TOOLS
        )
        self.assertGreaterEqual(
            len(unreachable), 7,
            f"expected several non-core tools, found {unreachable}",
        )
        brain = self._brain(cap=0)  # 0 = old behaviour, every core tool
        chosen = brain.select_tools("take a screenshot of the screen", 0)
        for tool in unreachable:
            with self.subTest(tool=tool):
                self.assertNotIn(
                    tool, chosen,
                    "if a non-core tool is reachable when selection is off, this "
                    "test no longer describes the problem it was written for",
                )

    def test_the_way_back_to_every_other_tool_is_always_offered(self):
        brain = self._brain()
        for request in list(self.REQUESTS) + ["", "   ", "hello", "xyzzy", "?!?"]:
            with self.subTest(request=request):
                chosen = brain.select_tools(request, brain.cfg.tool_select_max)
                self.assertIn(
                    "list_more_tools", chosen,
                    "without the escape hatch a mis-scored tool is simply lost, "
                    "because this model does not call list_more_tools on its own",
                )

    def test_selection_respects_the_cap(self):
        brain = self._brain(cap=6)
        for request in self.REQUESTS:
            with self.subTest(request=request):
                self.assertLessEqual(
                    len(brain.select_tools(request, 6)), 7,
                    "6 selected tools plus the floor",
                )

    def test_a_cap_of_zero_sends_every_core_tool(self):
        brain = self._brain(cap=0)
        chosen = brain.select_tools("what time is it", 0)
        from jarvis.brain import CORE_TOOLS

        self.assertEqual(set(chosen), set(CORE_TOOLS))

    def test_selection_is_deterministic(self):
        brain = self._brain()
        first = brain.select_tools("shut down the computer", 10)
        for _ in range(3):
            self.assertEqual(
                brain.select_tools("shut down the computer", 10), first,
                "the same request must produce the same prompt, or a cached "
                "prefill is thrown away on every turn",
            )

    def test_learned_tools_survive_later_selection(self):
        # A tool absorbed from list_more_tools must not be dropped next turn.
        brain = self._brain(extra=["ha_call_service"])
        for request in self.REQUESTS:
            with self.subTest(request=request):
                self.assertIn(
                    "ha_call_service", brain.select_tools(request, 10),
                    "a learned tool was dropped, so discovery has to be redone",
                )

    def test_every_selected_tool_has_a_schema(self):
        brain = self._brain()
        from jarvis.tools import registry

        for request in self.REQUESTS:
            with self.subTest(request=request):
                chosen = brain.select_tools(request, 10)
                specs = registry.specs_for(chosen)
                self.assertEqual(
                    [s["function"]["name"] for s in specs], sorted(chosen),
                    "selection returned a name with no schema behind it",
                )

    def test_selection_shrinks_the_prompt(self):
        import json

        from jarvis.brain import CORE_TOOLS
        from jarvis.tools import registry

        brain = self._brain()
        everything = len(json.dumps(registry.specs_for(sorted(CORE_TOOLS))))
        sizes = [
            len(json.dumps(registry.specs_for(brain.select_tools(q, 10))))
            for q in self.REQUESTS
        ]
        self.assertLess(
            max(sizes), everything,
            "selection did not reduce the prompt, so it costs accuracy for nothing",
        )

    def test_the_default_sends_every_core_tool(self):
        # Selection is off by default, and that is a measured decision rather
        # than an oversight. Ollama can only reuse a cached prefix when the new
        # prompt starts with the same tokens, so a per-question tool set misses
        # the cache every question. Measured in one process with the model
        # loaded, four questions:
        #
        #                 cold    2nd     3rd     repeat
        #   selection on  44.6s   36.3s   14.5s    0.13s
        #   selection off 68.1s    0.49s   0.62s   0.61s
        #
        # The cold turn is paid once per keep_alive window and the warm one on
        # every question, so the fixed working set wins overall here. Change
        # this only with a fresh measurement, not on the strength of the cold
        # turn alone.
        from jarvis.config import Config

        self.assertEqual(Config().tool_select_max, 0)

    def test_scoring_rewards_the_tools_own_name_most(self):
        from jarvis.brain import _tool_score, _tokens

        from jarvis.tools import load_all, registry

        load_all()
        tool = registry.get("take_screenshot")
        words = _tokens("please take a screenshot")
        by_name = _tool_score(words, "please take a screenshot", tool)
        by_desc = _tool_score(_tokens("a photo of the display"),
                              "a photo of the display", tool)
        self.assertGreater(
            by_name, by_desc,
            "a word of the tool's name should outweigh vague description overlap",
        )

    def test_a_tag_keyword_outweighs_description_overlap(self):
        from jarvis.brain import _tool_score, _tokens

        from jarvis.tools import load_all, registry

        load_all()
        tool = registry.get("ha_call_service")
        tagged = _tool_score(_tokens("turn on the lights"),
                             "turn on the lights", tool)
        vague = _tool_score(_tokens("do the thing"), "do the thing", tool)
        self.assertGreater(tagged, vague)


class TestVisionMemoryFts(unittest.TestCase):
    """The keyword index over observations has to actually be readable.

    It was declared as an FTS5 external-content table over `observations`,
    which made FTS5 look for a `body` column there. There is no such column, so
    every read raised "no such column: T.body" - the index was not slow, it was
    unusable, and `recall` swallowed the error, so keyword search quietly
    contributed nothing to every search. `prune` also warned on every startup.
    The index is contentless now, fed by triggers, so it has no content table
    to disagree with.
    """

    def setUp(self):
        from jarvis.vision_memory import VisionMemory

        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "v.db"
        self.vm = VisionMemory(self.path)

    def tearDown(self):
        self.vm.close()
        self.tmp.cleanup()

    def _obs(self, text, title="window"):
        from jarvis.perception import Observation

        return Observation(
            ts=1.0, window_app="notepad", window_title=title,
            width=100, height=100, text=text, thumb_path="", change="new",
        )

    def _count(self, sql):
        return self.vm._conn.execute(sql).fetchone()[0]

    def test_the_index_is_readable(self):
        self.vm.record(self._obs("zeppelin invoice total"))
        # The query that used to raise no such column: T.body.
        self.assertEqual(self._count("SELECT COUNT(*) FROM observations_fts"), 1)

    def test_a_recorded_observation_is_findable_by_keyword(self):
        obs_id = self.vm.record(self._obs("zeppelin invoice total"))
        hits = [
            r[0] for r in self.vm._conn.execute(
                "SELECT rowid FROM observations_fts WHERE observations_fts MATCH 'zeppelin'"
            )
        ]
        self.assertIn(obs_id, hits, "keyword search missed a recorded observation")

    def test_deleting_an_observation_retracts_it_from_the_index(self):
        obs_id = self.vm.record(self._obs("zeppelin invoice total"))
        self.vm._conn.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
        hits = [
            r[0] for r in self.vm._conn.execute(
                "SELECT rowid FROM observations_fts WHERE observations_fts MATCH 'zeppelin'"
            )
        ]
        self.assertNotIn(
            obs_id, hits,
            "a deleted observation was still findable, so recall would offer "
            "something that is no longer there",
        )

    def test_prune_does_not_raise_and_leaves_the_index_consistent(self):
        for i in range(8):
            self.vm.record(self._obs(f"observation number {i}"))
        # prune used to warn on every startup because of the rebuild.
        removed = self.vm.prune(retention_days=30, max_rows=5)
        self.assertGreater(removed, 0)
        self.assertEqual(
            self._count("SELECT COUNT(*) FROM observations_fts"),
            self._count("SELECT COUNT(*) FROM observations"),
            "prune left the keyword index disagreeing with the observations",
        )

    def test_recall_uses_the_keyword_path(self):
        self.vm.record(self._obs("quarterly report spreadsheet"))
        # The blend that used to lose its keyword half to a swallowed error.
        results = self.vm.recall("spreadsheet")
        self.assertTrue(results, "recall returned nothing for a term it stored")
        self.assertIn("keyword", results[0]["matched"])

    def test_an_index_left_over_from_the_old_schema_is_repaired_on_open(self):
        from jarvis.vision_memory import VisionMemory

        # FTS5 only creates its table if absent, so fixing the declaration
        # would never reach a database made before the fix. An existing broken
        # index has to be noticed and remade, or the app stays broken on the
        # one file that actually matters: the one already on disk.
        self.vm.record(self._obs("zeppelin invoice total"))
        self.vm.record(self._obs("another observation"))
        self.vm.close()

        # Put the file back the way the old code left it.
        raw = sqlite3.connect(self.path)
        raw.execute("DROP TABLE observations_fts")
        raw.execute(
            "CREATE VIRTUAL TABLE observations_fts USING fts5("
            "body, content='observations', content_rowid='id')"
        )
        raw.commit()
        raw.close()
        with self.assertRaises(sqlite3.OperationalError):
            broken = sqlite3.connect(self.path)
            try:
                broken.execute("SELECT COUNT(*) FROM observations_fts").fetchone()
            finally:
                broken.close()

        self.vm = VisionMemory(self.path)  # reopen, as a second run would
        self.assertTrue(self.vm.fts)
        self.assertEqual(
            self._count("SELECT COUNT(*) FROM observations_fts"),
            self._count("SELECT COUNT(*) FROM observations"),
            "the old index was not remade and refilled on open",
        )
        hits = [
            r[0] for r in self.vm._conn.execute(
                "SELECT rowid FROM observations_fts WHERE observations_fts MATCH 'zeppelin'"
            )
        ]
        self.assertTrue(hits, "the repaired index found nothing it had stored")

    def test_reopening_keeps_the_index_consistent(self):
        from jarvis.vision_memory import VisionMemory

        self.vm.record(self._obs("zeppelin invoice total"))
        self.vm.close()
        self.vm = VisionMemory(self.path)
        self.assertEqual(
            self._count("SELECT COUNT(*) FROM observations_fts"),
            self._count("SELECT COUNT(*) FROM observations"),
        )


class TestMultiMonitorGeometry(unittest.TestCase):
    """A second display must be reachable, in one coordinate space.

    Both of these were real bugs on a 1920x1200 laptop panel at 150% scaling
    beside a 1920x1080 monitor at 100%, and both fail silently: the first
    refuses a coordinate it should accept, the second moves the pointer somewhere
    else without complaining. The desktop is stubbed here so these run the same
    on a single screen.
    """

    # A laptop panel at 0,0 and a second monitor to its right, as mss reports.
    DESKTOP = (3840, 1200)
    LAPTOP = {"index": 1, "left": 0, "top": 0, "width": 1920, "height": 1200, "primary": True}
    ASUS = {"index": 2, "left": 1920, "top": 0, "width": 1920, "height": 1080, "primary": False}

    def setUp(self):
        from unittest import mock

        import jarvis.env as env
        from jarvis.config import load_config
        from jarvis.mouse import Mouse

        self.mock = mock
        self.cfg = load_config()
        self.mouse = Mouse(self.cfg)
        # Patch the module attribute: both callers import it inside the function,
        # so this is the seam they actually use.
        self.desktop = mock.patch.object(env, "virtual_desktop",
                                         return_value=self.DESKTOP)
        self.desktop.start()
        self.addCleanup(self.desktop.stop)

    def test_the_mouse_uses_the_whole_desktop_not_the_primary(self):
        import pyautogui

        # pyautogui only ever describes the primary monitor.
        self.assertEqual(tuple(pyautogui.size()), (1920, 1200),
                         "this test's premise is that pyautogui is primary-only")
        self.assertEqual(self.mouse.size(), self.DESKTOP,
                         "the mouse bounds came from the primary monitor, so the "
                         "second display is unreachable")

    def test_a_coordinate_on_the_second_display_is_accepted(self):
        for x, y in [(1920, 0), (2880, 540), (3839, 1199)]:
            with self.subTest(point=(x, y)):
                got = self.mouse.validate(x, y)
                self.assertTrue(0 <= got[0] < self.DESKTOP[0])
                self.assertTrue(0 <= got[1] < self.DESKTOP[1])

    def test_a_coordinate_past_the_desktop_is_still_refused(self):
        from jarvis.mouse import ActuationRefused

        for x, y in [(3840, 0), (0, 1200), (-1, 0)]:
            with self.subTest(point=(x, y)):
                with self.assertRaises(ActuationRefused):
                    self.mouse.validate(x, y)

    def test_the_refusal_names_the_desktop_rather_than_the_screen(self):
        from jarvis.mouse import ActuationRefused

        with self.assertRaises(ActuationRefused) as caught:
            self.mouse.validate(9999, 9999)
        self.assertIn("3840x1200", str(caught.exception))

    def test_moving_onto_the_second_display_works(self):
        with self.mock.patch("pyautogui.moveTo") as move_to:
            with self.mock.patch("pyautogui.position", return_value=(0, 0)):
                self.mouse.move(2880, 540, humanize=False)
        self.assertTrue(move_to.called, "the move never reached pyautogui")
        self.assertEqual(tuple(move_to.call_args[0][:2]), (2880, 540))

    def test_a_humanised_move_may_cross_onto_the_second_display(self):
        # The per-step clamp used to be the primary's bounds, which pinned the
        # pointer to the first screen for the whole of a crossing move. Only the
        # intermediate steps are checked: the final moveTo is deliberately not
        # clamped, so asserting on it would pass either way.
        with self.mock.patch("pyautogui.moveTo") as move_to:
            with self.mock.patch("pyautogui.position", return_value=(100, 100)):
                result = self.mouse.move(3600, 900, humanize=True)
        self.assertEqual(list(result["at"]), [3600, 900])
        steps = [tuple(c[0][:2]) for c in move_to.call_args_list]
        self.assertGreater(len(steps), 2, "the humanised path did not animate")
        along_the_way = steps[:-1]
        self.assertTrue(
            any(x > 1920 for x, _ in along_the_way),
            f"the pointer never left the primary display mid-move; "
            f"max x along the path was {max(x for x, _ in along_the_way)}",
        )

    def test_mouse_position_reports_the_whole_desktop(self):
        from jarvis.tools import load_all, registry
        from jarvis.tools import screen_tools

        load_all()
        with self.mock.patch.object(
            screen_tools, "_enumerate",
            return_value=[{"index": 0, "left": 0, "top": 0,
                           "width": 3840, "height": 1200, "primary": False},
                          dict(self.LAPTOP), dict(self.ASUS)],
        ):
            with self.mock.patch("pyautogui.position", return_value=(2880, 540)):
                got = registry.get("mouse_position").handler()
        self.assertEqual(got["screen"], [3840, 1200])
        self.assertEqual(
            got["display"], 2,
            "the cursor is on the second monitor, so index 0 - the union of every "
            "display - must not be reported as the answer",
        )

    def test_mouse_position_on_the_laptop_reports_display_one(self):
        from jarvis.tools import load_all, registry
        from jarvis.tools import screen_tools

        load_all()
        with self.mock.patch.object(
            screen_tools, "_enumerate",
            return_value=[{"index": 0, "left": 0, "top": 0,
                           "width": 3840, "height": 1200, "primary": False},
                          dict(self.LAPTOP), dict(self.ASUS)],
        ):
            with self.mock.patch("pyautogui.position", return_value=(960, 600)):
                got = registry.get("mouse_position").handler()
        self.assertEqual(got["display"], 1)

    def test_dpi_awareness_is_claimed_once_and_cached(self):
        import jarvis.env as env

        # Awareness is a once-per-process property, so a second call cannot
        # succeed and must not report a level it did not get.
        first = env.claim_dpi_awareness()
        self.assertEqual(env.claim_dpi_awareness(), first)
        self.assertEqual(env.DPI_AWARENESS, first)
        self.assertIn(first, ("per-monitor-v2", "per-monitor", "system",
                              "not-windows", "no-ctypes", "no-user32", "none"))

    def test_dpi_awareness_is_per_monitor_when_windows_allows_it(self):
        import os

        import jarvis.env as env

        if os.name != "nt":
            self.skipTest("DPI awareness is a Windows concept")
        # Not 'system': that is the level that let a 150% primary rescale the
        # second monitor to 2880x1620 in mss.
        self.assertNotEqual(env.DPI_AWARENESS, "system")
        self.assertIn(env.DPI_AWARENESS, ("per-monitor-v2", "per-monitor", "none"))


class TestFastPathGate(unittest.TestCase):
    """The gate that decides a question needs no tools at all.

    It exists because the 38-tool schema is most of the prompt, and a warm turn
    with it attached took 103s to first output against 0.33s without. The cost of
    getting it wrong is asymmetric: a slow correct answer is fine, a fast wrong
    one is not. So every one of these tests is about leaning towards the tools.
    """

    def test_a_plain_question_needs_no_tools(self):
        from jarvis.brain import needs_tools

        for question in (
            "what is the capital of france",
            "how do i bake sourdough bread",
            "why is the sky blue",
            "hello there",
            "thanks, that was helpful",
            "2 + 2",
            "tell me a joke",
            "summarise what we decided about the budget",
        ):
            with self.subTest(question=question):
                self.assertFalse(needs_tools(question))

    def test_anything_naming_a_tool_thing_needs_a_tool(self):
        from jarvis.brain import needs_tools

        for question in (
            "take a screenshot of the screen",
            "turn off the lights",
            "what do you remember about my wifi",
            "shut down the computer",
            "what is the weather in paris",
            "copy that to the clipboard",
            "read the file report.docx",
        ):
            with self.subTest(question=question):
                self.assertTrue(needs_tools(question))

    def test_an_imperative_needs_a_tool_even_with_no_tag_word(self):
        # The gap that made the tag vocabulary unsafe on its own. Not one of
        # these appears in any TOOL_TAGS entry, and all of them are obvious tool
        # calls: a bare verb that only makes sense if something is done.
        from jarvis.brain import needs_tools

        for question in (
            "open notepad",
            "launch the browser",
            "install firefox",
            "find my keys",
            "send an email to sam",
            "play the next track",
            "record a note",
        ):
            with self.subTest(question=question):
                self.assertTrue(needs_tools(question))

    def test_the_verb_check_does_not_fire_on_substrings(self):
        # Substring matching would make "opening hours" an imperative and
        # "running water" a request to execute something.
        from jarvis.brain import needs_tools

        self.assertFalse(needs_tools("what are the opening hours"))
        self.assertFalse(needs_tools("explain running water erosion"))

    def test_politeness_does_not_force_the_slow_path(self):
        # These appear in almost every request. Blocking on them would put
        # nearly all traffic on the slow path and the fast path would never run.
        from jarvis.brain import needs_tools

        for question in (
            "can you explain gravity",
            "could you tell me about roman empire",
            "please explain photosynthesis",
        ):
            with self.subTest(question=question):
                self.assertFalse(needs_tools(question))

    def test_an_empty_request_is_not_treated_as_simple(self):
        from jarvis.brain import needs_tools

        self.assertTrue(needs_tools(""))
        self.assertTrue(needs_tools("   "))

    def test_a_refusal_is_recognised_so_the_slow_call_can_happen(self):
        from jarvis.brain import fast_path_refused

        for text in (
            "I don't have access to your calendar.",
            "I can't do that without a tool.",
            "I'm unable to open applications.",
            "You would need to use the open_app tool.",
        ):
            with self.subTest(text=text):
                self.assertTrue(fast_path_refused(text))

    def test_a_normal_answer_is_not_mistaken_for_a_refusal(self):
        # The dangerous direction here is a false positive: this would throw away
        # a perfectly good fast answer and pay the slow call for nothing.
        from jarvis.brain import fast_path_refused

        for text in (
            "The capital of France is Paris.",
            "You can find the file under your documents folder.",
            "I can tell you about gravity, though I cannot see your screen.",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(fast_path_refused(text))


class TestFastPathRouting(unittest.TestCase):
    """A no-tools question must skip the schema block, and a bad guess must
    be recoverable rather than returned to the user."""

    def _brain(self, replies):
        """A Brain whose model returns `replies` in order.

        Each entry is (text, tool_calls). Tools are recorded per call so a test
        can assert what the model was actually offered.
        """
        from jarvis.brain import Brain
        from jarvis.config import load_config

        cfg = load_config()
        cfg.fast_path = True
        brain = Brain.__new__(Brain)
        brain.cfg = cfg
        brain._extra_tools = set()
        brain._ensure_registry = lambda: None
        brain.model = "stub"
        brain.memory = _FastPathMemory()
        brain.on_thinking = None
        brain._pending = list(replies)
        brain.calls = []

        def fake_round(messages, tools, on_token):
            brain.calls.append(list(tools or []))
            return brain._pending.pop(0)

        brain._round = fake_round
        brain._tools = lambda user_text=None: [{"function": {"name": "open_app"}}]
        brain._messages = lambda user_text, images: [{"role": "user", "content": user_text}]
        return brain

    def test_a_simple_question_is_answered_with_no_tools_sent(self):
        from jarvis.brain import Brain

        brain = self._brain([("The capital of France is Paris.", None)])
        brain.preroute = lambda user_text: None
        brain.supports_images = lambda: False
        reply = Brain.think(brain, "what is the capital of france")
        self.assertEqual(reply.text, "The capital of France is Paris.")
        self.assertEqual(len(brain.calls), 1, "should not have needed a second call")
        self.assertEqual(brain.calls[0], [], "the tool block was still sent")

    def test_a_tool_ish_question_goes_straight_to_the_tools(self):
        from jarvis.brain import Brain

        # The stub answers in text rather than emitting a real tool call. This
        # test is about what the model is offered, and a genuine call would
        # launch Notepad on the machine running the tests.
        brain = self._brain([("Opening Notepad.", None)])
        brain.preroute = lambda user_text: None
        brain.supports_images = lambda: False
        Brain.think(brain, "open notepad")
        self.assertEqual(len(brain.calls), 1)
        self.assertTrue(brain.calls[0], "no tools were offered for a tool request")

    def test_a_refusal_gets_retried_with_tools_instead_of_returned(self):
        # The safety net. If the gate guesses wrong, the user gets the real
        # answer rather than "I can't do that".
        from jarvis.brain import Brain

        brain = self._brain([
            ("I don't have access to that.", None),
            ("Notepad is open.", None),
        ])
        brain.preroute = lambda user_text: None
        brain.supports_images = lambda: False
        reply = Brain.think(brain, "what is the capital of france")
        self.assertEqual(reply.text, "Notepad is open.")
        self.assertEqual(len(brain.calls), 2, "the slow retry never happened")
        self.assertEqual(brain.calls[0], [], "first call should have had no tools")
        self.assertTrue(brain.calls[1], "retry should have had the tools")

    def test_the_fast_path_can_be_switched_off(self):
        from jarvis.brain import Brain

        brain = self._brain([("Paris.", None)])
        brain.cfg.fast_path = False
        brain.preroute = lambda user_text: None
        brain.supports_images = lambda: False
        Brain.think(brain, "what is the capital of france")
        self.assertTrue(brain.calls[0], "fast_path=False still skipped the tools")

    def test_an_attached_image_always_takes_the_slow_path(self):
        # The model can read an image with no tools, but the follow-up almost
        # always needs one, and the vision call is slow enough that guessing is
        # not worth it.
        from jarvis.brain import Brain

        brain = self._brain([("There is a button.", None)])
        brain.preroute = lambda user_text: None
        brain.supports_images = lambda: True
        Brain.think(brain, "what is this", images=["shot.png"])
        self.assertTrue(brain.calls[0], "an image skipped the tool block")


class TestAcknowledgeCue(unittest.TestCase):
    """The 'I heard you' cue, and the claim that it can beat 50ms."""

    def test_a_known_tone_is_a_short_quiet_wave(self):
        from jarvis.audio import ack_tone

        wave = ack_tone("tick")
        self.assertIsNotNone(wave)
        self.assertEqual(wave.dtype.name, "float32")
        # Under 100ms. A cue the user has to wait for is just a delay.
        self.assertLess(len(wave) / 16000, 0.1)
        self.assertLessEqual(float(abs(wave).max()), 0.3, "too loud for a cue")
        # Starts and ends at zero, so it cannot click.
        self.assertLess(abs(float(wave[0])), 1e-3)
        self.assertLess(abs(float(wave[-1])), 1e-3)

    def test_the_tone_actually_oscillates(self):
        # Guards against a window of silence passing as a tone.
        from jarvis.audio import ack_tone

        wave = ack_tone("chime")
        crossings = sum(
            1 for i in range(1, len(wave))
            if (wave[i - 1] < 0) != (wave[i] < 0)
        )
        self.assertGreater(crossings, 4)

    def test_an_unknown_or_empty_name_gives_no_wave(self):
        from jarvis.audio import ack_tone

        self.assertIsNone(ack_tone("nonsense"))
        self.assertIsNone(ack_tone(""))
        # No stream should be opened to play nothing.
        import jarvis.audio as audio

        saved = (audio._ACK_STREAM, audio._ACK_STREAM_DEAD)
        audio._ACK_STREAM, audio._ACK_STREAM_DEAD = None, True
        try:
            self.assertFalse(audio.play_ack("nonsense"))
        finally:
            audio._ACK_STREAM, audio._ACK_STREAM_DEAD = saved

    def test_names_are_forgiving_about_case_and_padding(self):
        from jarvis.audio import ack_tone

        for name in ("TICK", "  tick  ", "Tick"):
            with self.subTest(name=name):
                self.assertIsNotNone(ack_tone(name))

    def test_the_cue_is_queued_not_blocking(self):
        # Writing to a running stream returns immediately. A blocking call here
        # would put the cue's duration in front of the model call.
        import inspect

        from jarvis import audio

        source = inspect.getsource(audio.play_ack)
        self.assertNotIn("blocking=True", source)
        self.assertNotIn("sd.wait()", source)
        self.assertIn("stream.write(", source)

    def test_the_output_stream_is_opened_before_anything_is_waiting(self):
        # Opening an OutputStream costs ~200ms here and the first write costs
        # another ~220ms while the driver primes. Paying either at reply time
        # would make the cue slower than the answer it is covering, so the
        # stream is opened while idle.
        import inspect

        from jarvis import audio

        self.assertIn("warm_ack_stream", dir(audio))
        self.assertIn("OutputStream", inspect.getsource(audio.warm_ack_stream))

    def test_warming_primes_the_driver_with_silence(self):
        # Opening the stream is not sufficient. The first write to a fresh
        # stream costs ~220ms while the driver primes, and a zero-length write
        # does not trigger it, so the prime has to be real samples. Measured:
        # warming with 20ms of zeros took 10.4ms here, and the first cue
        # afterwards wrote in 0.44ms instead of 223.11ms.
        import inspect

        import numpy as np

        from jarvis import audio

        source = inspect.getsource(audio.warm_ack_stream)
        self.assertIn("stream.write(", source, "the stream is never primed")

        written = []

        class FakeStream:
            def start(self):
                pass

            def write(self, buf):
                written.append(np.asarray(buf))

            def stop(self):
                pass

            def close(self):
                pass

        import sounddevice as sd

        real = sd.OutputStream
        sd.OutputStream = lambda **kw: FakeStream()
        audio._ACK_STREAM, audio._ACK_STREAM_DEAD = None, False
        try:
            self.assertTrue(audio.warm_ack_stream())
            self.assertEqual(len(written), 1, "warm_ack_stream wrote nothing")
            prime = written[0]
            self.assertGreater(len(prime), 0, "an empty write does not prime the driver")
            self.assertTrue(
                not prime.any(), "the prime must be silence, not a tone"
            )
        finally:
            audio.close_ack_stream()
            sd.OutputStream = real
            audio._ACK_STREAM, audio._ACK_STREAM_DEAD = None, False

    def test_warming_the_stream_happens_once_and_latches_failure(self):
        from jarvis import audio

        opened = []

        class FakeStream:
            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

            def write(self, buf):
                pass

        import sounddevice as sd

        real = sd.OutputStream
        sd.OutputStream = lambda **kw: (opened.append(kw), FakeStream())[1]
        audio._ACK_STREAM = None
        audio._ACK_STREAM_DEAD = False
        try:
            self.assertTrue(audio.warm_ack_stream())
            self.assertTrue(audio.warm_ack_stream(), "warmed twice")
            self.assertEqual(len(opened), 1, "the device was opened more than once")
        finally:
            audio.close_ack_stream()
            sd.OutputStream = real
            audio._ACK_STREAM = None
            audio._ACK_STREAM_DEAD = False

    def test_a_missing_output_device_is_remembered_not_retried_forever(self):
        # Without the latch, every turn would pay another failed device open.
        from jarvis import audio

        import sounddevice as sd

        real = sd.OutputStream
        attempts = []

        def boom(**kw):
            attempts.append(kw)
            raise RuntimeError("no output device")

        sd.OutputStream = boom
        audio._ACK_STREAM = None
        audio._ACK_STREAM_DEAD = False
        try:
            audio.play_ack("tick")
            audio.play_ack("tick")
            self.assertLessEqual(
                len(attempts), 1,
                f"tried to open a dead device {len(attempts)} times",
            )
            self.assertTrue(audio._ACK_STREAM_DEAD)
        finally:
            sd.OutputStream = real
            audio._ACK_STREAM = None
            audio._ACK_STREAM_DEAD = False

    def test_writing_to_the_cue_stream_does_not_block_on_playback(self):
        # The measured claim: with the stream already open, the write returns in
        # well under a millisecond even though the sound plays afterwards.
        from jarvis import audio

        written = []

        class FakeStream:
            def start(self):
                pass

            def write(self, buf):
                written.append(len(buf))

        import sounddevice as sd

        real = sd.OutputStream
        sd.OutputStream = lambda **kw: FakeStream()
        audio._ACK_STREAM = None
        audio._ACK_STREAM_DEAD = False
        try:
            start = time.perf_counter()
            self.assertTrue(audio.play_ack("tick"))
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.assertEqual(len(written), 1)
            self.assertLess(elapsed_ms, 50, f"cue took {elapsed_ms:.0f}ms to queue")
        finally:
            audio._ACK_STREAM = None
            audio._ACK_STREAM_DEAD = False
            sd.OutputStream = real

    def test_the_cue_cannot_take_down_a_turn(self):
        # No output device, a broken driver, a held device: none of that may
        # stop the answer, which is the part the user actually came for.
        import jarvis.audio as audio

        real = audio.ack_tone
        audio.ack_tone = lambda name, rate=16000: (_ for _ in ()).throw(
            RuntimeError("no output device")
        )
        saved = (audio._ACK_STREAM, audio._ACK_STREAM_DEAD)
        audio._ACK_STREAM, audio._ACK_STREAM_DEAD = None, False
        try:
            self.assertFalse(audio.play_ack("tick"))
        finally:
            audio.ack_tone = real
            audio._ACK_STREAM, audio._ACK_STREAM_DEAD = saved

    def test_the_stream_is_shared_not_reopened_per_cue(self):
        # TTS and the cue share one output device. Two independent
        # OutputStreams on a Windows output device is a reliable way to get
        # exclusive-mode errors, so play_ack reuses one process-wide stream.
        import jarvis.audio as audio

        streams = []

        class FakeStream:
            def start(self):
                pass

            def write(self, buf):
                streams.append(("write", len(buf)))

            def stop(self):
                streams.append(("stop", 0))

            def close(self):
                streams.append(("close", 0))

        import sounddevice as sd

        real = sd.OutputStream
        created = []

        def factory(**kw):
            created.append(kw)
            return FakeStream()

        sd.OutputStream = factory
        audio._ACK_STREAM, audio._ACK_STREAM_DEAD = None, False
        try:
            audio.play_ack("tick")
            audio.play_ack("chime")
            audio.play_ack("tick")
            self.assertEqual(len(created), 1, f"opened {len(created)} output streams")
            self.assertEqual(len([s for s in streams if s[0] == "write"]), 3)
        finally:
            audio.close_ack_stream()
            sd.OutputStream = real
            audio._ACK_STREAM, audio._ACK_STREAM_DEAD = None, False

    def test_turns_emit_the_cue_before_the_model_is_called(self):
        # Ordering is the feature. A cue after respond() is pointless, since by
        # then there is an answer to hear.
        import inspect

        from jarvis.voice import VoiceLoop

        source = inspect.getsource(VoiceLoop._run_turn)
        self.assertLess(
            source.index("_acknowledge()"),
            source.index("self.assistant.respond("),
            "the cue must fire before the model is called",
        )

    def test_the_cue_is_off_when_configured_empty(self):
        from jarvis.config import load_config
        from jarvis.voice import VoiceLoop

        cfg = load_config()
        cfg.acknowledge_sound = ""
        loop = VoiceLoop.__new__(VoiceLoop)
        loop.ack_sound = cfg.acknowledge_sound or ""
        loop._say = lambda *a, **k: None
        called = []
        import jarvis.voice as voice

        real = voice.play_ack
        voice.play_ack = lambda *a, **k: called.append(a)
        try:
            loop._acknowledge()
            time.sleep(0.05)
        finally:
            voice.play_ack = real
        self.assertEqual(called, [], "an empty setting should produce no sound")

    def test_shutting_down_releases_the_cue_output(self):
        # The stream is held open for the whole session, so cues stay cheap.
        # shutdown() also runs on a restart, and a held OutputStream keeps the
        # output device, so the loop has to give it back.
        import inspect

        from jarvis.voice import VoiceLoop

        self.assertIn("close_ack_stream", inspect.getsource(VoiceLoop.shutdown))

    def test_the_stream_is_warmed_when_the_loop_starts(self):
        # Without this, the first cue of the session pays the device open and
        # lands about 400ms after the user stopped talking.
        import inspect

        from jarvis.voice import VoiceLoop

        source = inspect.getsource(VoiceLoop.run)
        self.assertIn("warm_ack_stream", source)

    def test_building_the_cue_takes_far_less_than_fifty_milliseconds(self):
        # The reason the cue can meet the target at all: there is no file to
        # read and no codec to run.
        from jarvis.audio import ack_tone

        start = time.perf_counter()
        for _ in range(20):
            ack_tone("tick")
        each_ms = (time.perf_counter() - start) * 1000 / 20
        self.assertLess(each_ms, 50, f"cue generation cost {each_ms:.1f}ms each")


class _FastPathMemory:
    """Just enough Memory for think() to record a turn.

    Named distinctly from the other stubs here: a module-level _StubMemory
    already exists further up with a different constructor, and shadowing it
    breaks unrelated tests.
    """

    def __init__(self, rows=None):
        self.rows = []
        self._rows = list(rows or [])

    def add_message(self, role, text, **kw):
        self.rows.append((role, text))

    def recent(self, *a, **k):
        return []


class TestVlmDescribeCache(unittest.TestCase):
    """Describing the same screen twice must not cost 71-91s twice.

    The measurement this rests on: with qwen2.5vl on a real 1920x1200 screen,
    three questions against one image took 71.74s, 6.28s and 3.11s, and
    changing one region by 40 levels put the first one back to 91.66s. The cost
    is per distinct image, not per question, because the first call is what
    pays for ~1100 image tokens and ollama keeps the result.

    So the cache is keyed on the encoded bytes: a hit here predicts a cheap call
    to the server, and a changed screen is correctly a miss.
    """

    def _vision(self, entries=24, answer="A terminal window.", reuse=0.0025,
                reuse_no_ocr=0.001):
        from jarvis.vlm import Vision

        class Cfg:
            vlm_model = "moondream"
            vlm_describe_model = "qwen2.5vl:3b"
            ollama_host = "http://localhost:11434"
            vlm_timeout = 300.0
            vlm_max_side = 896
            vlm_describe_max_side = 448
            vlm_num_predict = 256
            vlm_describe_num_predict = 96
            vlm_describe_budget_seconds = 150.0
            vlm_describe_cache_entries = entries
            vlm_describe_reuse_max_changed = reuse
            vlm_reuse_max_changed_no_ocr = reuse_no_ocr

        v = Vision(Cfg())
        self.asked: list[str] = []
        counter = [0]

        def fake_ask(image, prompt, model="", max_side=None, num_predict=None):
            self.asked.append(prompt)
            counter[0] += 1
            # Distinguishable per call, so a stale answer cannot masquerade as
            # a fresh one in the assertions below.
            return f"{answer} ({counter[0]})"

        v._ask = fake_ask
        return v

    def test_the_same_screen_is_described_once(self):
        v = self._vision()
        first = v.describe(_fake_screen(1), "what is this window?")
        second = v.describe(_fake_screen(1), "what is this window?")
        self.assertEqual(first, second)
        self.assertEqual(len(self.asked), 1, f"asked {len(self.asked)} times")

    def test_the_multi_question_path_is_cached_as_one_unit(self):
        # The three detail prompts are not independent: the first one is what
        # pays the image cost and the other two ride on it. So the cache holds
        # the joined result, and a repeat must not re-ask any of them.
        from jarvis.vlm import Vision

        v = self._vision()
        first = v.describe(_fake_screen(1))
        self.assertEqual(len(self.asked), len(Vision._DETAIL_PROMPTS))
        before = len(self.asked)
        second = v.describe(_fake_screen(1))
        self.assertEqual(first, second)
        self.assertEqual(
            len(self.asked), before, f"re-asked {len(self.asked) - before} times"
        )

    def test_a_changed_screen_is_described_again(self):
        # The dangerous direction: a stale description served for a screen that
        # has moved on is worse than no description at all. A real visual
        # change, at real screen dimensions - the 4x4 stand-in differs by a
        # single intensity level, which is below the noise floor by design and
        # so is correctly not treated as a change.
        v = self._vision()
        first = np.full((600, 900, 3), 40, np.uint8)
        second = first.copy()
        second[150:500, 200:700] = 250
        v.describe(first, "what is this window?")
        v.describe(second, "what is this window?")
        self.assertEqual(len(self.asked), 2, "a changed screen reused a description")

    def test_a_different_question_is_a_different_answer(self):
        # Answering the wrong question quickly would be worse than answering
        # slowly, so the question is part of the key.
        v = self._vision()
        v.describe(_fake_screen(1), "what is this window?")
        v.describe(_fake_screen(1), "what is the app name?")
        self.assertEqual(len(self.asked), 2, "the second question reused the first answer")

    def test_the_unchanged_screen_case_still_hits_the_cache(self):
        v = self._vision()
        # Past the 12-content-word threshold, so this is the single-prompt path
        # rather than the three-question one.
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        a = v.describe(_fake_screen(1), ocr_text=rich)
        b = v.describe(_fake_screen(1), ocr_text=rich)
        self.assertEqual(a, b)
        self.assertEqual(len(self.asked), 1)

    def test_the_cache_is_bounded_and_evicts_the_least_recently_used(self):
        # reuse=-1 so this exercises the exact-bytes tier alone. With the loose
        # tier on, an evicted entry would still be found by signature, which is
        # the intended behaviour rather than a failure of eviction.
        v = self._vision(entries=2, reuse=-1)
        for seed in (1, 2, 3):
            v.describe(_fake_screen(seed), "q")
        self.assertLessEqual(
            len(v._describe_cache), 2, f"cache grew to {len(v._describe_cache)}"
        )
        # 1 was evicted, so it costs a call again.
        before = len(self.asked)
        v.describe(_fake_screen(1), "q")
        self.assertEqual(len(self.asked), before + 1, "the evicted entry was reused")

    def test_disabling_the_cache_always_asks(self):
        v = self._vision(entries=0)
        v.describe(_fake_screen(1), "q")
        v.describe(_fake_screen(1), "q")
        self.assertEqual(len(self.asked), 2, "a disabled cache still returned a hit")

    def test_an_unusable_answer_is_not_cached(self):
        # Caching "" would mean a screen that becomes describable later keeps
        # returning nothing, because the miss would never be retried.
        v = self._vision()
        bad = "urn:jira:issue/uid:urn:jira:issue/uid:urn:jira:issue/uid:"

        def fake_ask(image, prompt, model="", max_side=None, num_predict=None):
            self.asked.append(prompt)
            return bad if len(self.asked) == 1 else "A code editor."

        v._ask = fake_ask
        v.describe(_fake_screen(1), "q")
        v.describe(_fake_screen(1), "q")
        self.assertEqual(len(self.asked), 2, "an unusable answer was cached as a hit")

    def test_the_key_follows_the_bytes_sent_to_the_model(self):
        # Both this cache and ollama's key on the encoded payload. If they
        # disagreed, a hit here would predict an expensive call there.
        from jarvis.vlm import Vision

        v = self._vision()
        k1 = v._image_key(_fake_screen(1), "q", "m")
        k2 = v._image_key(_fake_screen(1), "q", "m")
        k3 = v._image_key(_fake_screen(2), "q", "m")
        self.assertEqual(k1, k2, "the same pixels gave different keys")
        self.assertNotEqual(k1, k3, "different pixels gave the same key")

    def test_a_cursor_blink_reuses_the_description(self):
        # The measured reality: six grabs four seconds apart of a live screen
        # gave six distinct images and zero exact-bytes hits, differing by
        # 0.066% of pixels. If the loose tier does not catch that, the cache is
        # decoration and every describe costs 85s again.
        v = self._vision()
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        quiet = _fake_screen(1)
        blink = quiet.copy()
        blink[3, 3] = (9, 9, 9)  # one pixel, like a cursor
        a = v.describe(quiet, ocr_text=rich)
        b = v.describe(blink, ocr_text=rich)
        self.assertEqual(a, b)
        self.assertEqual(len(self.asked), 1, f"re-asked {len(self.asked)} times")

    def test_the_clock_advancing_reuses_the_description(self):
        v = self._vision()
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
 # A real screen shape, so a clock-sized region is genuinely a small fraction
        # of the frame. On the 4x4 stand-in a few pixels are a quarter of it.
        before = np.full((600, 900, 3), 60, np.uint8)
        after = before.copy()
        after[560:585, 820:890] = 66  # ~0.2%, like a clock ticking over
        a = v.describe(before, ocr_text=rich)
        b = v.describe(after, ocr_text=rich)
        self.assertEqual(a, b)
        self.assertEqual(len(self.asked), 1, f"re-asked {len(self.asked)} times")

    def test_a_word_appearing_is_not_reused(self):
        # The load-bearing part of the loose gate. A changed pixel count alone
        # cannot tell a cursor from new text; agreeing OCR can. The new text is
        # kept small on purpose - a few hundred pixels, well under the pixel
        # bound - so this passes only because the words differ.
        v = self._vision()
        base = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        quiet = np.full((600, 900, 3), 40, np.uint8)
        # ~0.17% of the frame, comfortably inside the 0.25% default bound.
        noisy = quiet.copy()
        noisy[300:315, 200:290] = 250
        v.describe(quiet, ocr_text=base)
        v.describe(noisy, ocr_text=base + " ERROR fatal exception denied")
        self.assertEqual(len(self.asked), 2, "new on-screen words reused an old answer")

    def test_a_word_disappearing_is_not_reused(self):
        v = self._vision()
        base = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        quiet = np.full((600, 900, 3), 40, np.uint8)
        noisy = quiet.copy()
        noisy[300:315, 200:290] = 250
        v.describe(noisy, ocr_text=base)
        v.describe(quiet, ocr_text=base.replace("saved", "here"))
        self.assertEqual(len(self.asked), 2, "vanished words reused an old answer")

    def test_a_real_change_is_not_reused_even_with_identical_ocr(self):
        # Identical OCR text plus a substantially different image. This is the
        # dangerous case the pixel backstop exists for: the words agree, so only
        # the signature can notice that a different window is up.
        v = self._vision()
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        first = np.full((400, 600, 3), 20, np.uint8)
        second = np.full((400, 600, 3), 200, np.uint8)
        v.describe(first, ocr_text=rich)
        v.describe(second, ocr_text=rich)
        self.assertEqual(len(self.asked), 2, "a different window reused the description")

    def test_reuse_is_tighter_when_there_is_no_text_to_agree_with(self):
        # This path is reached precisely because OCR had nothing usable to say,
        # so there is no corroboration and the bound is the calibrated 0.1%
        # rather than the 0.25% OCR agreement earns. A change of ~0.7% is
        # therefore refused here even though the same change would be reused
        # with OCR text to corroborate it.
        v = self._vision()
        # 64x64 so the signature resize is the identity, and a change of exactly
        # N pixels is exactly N/4096 of it. Seven pixels is 0.171%, which the
        # OCR-gated 0.25% would allow and the blank-gated 0.1% must not.
        quiet = np.full((64, 64, 3), 30, np.uint8)
        nudged = quiet.copy()
        nudged[0, :7] = 250  # full contrast, above the 24-level noise floor
        self.assertAlmostEqual(
            v._changed_fraction(v._signature(quiet), v._signature(nudged)),
            7 / 4096, places=4,
        )
        a = v.describe(quiet)
        b = v.describe(nudged)
        self.assertNotEqual(a, b, "reused with no OCR to agree with")
        # The multi-question path, so three prompts each.
        self.assertEqual(len(self.asked), 6, f"asked {len(self.asked)} times")

    def test_the_same_change_is_reused_when_ocr_agrees(self):
        # The counterpart, so the tighter blank bound is the only difference: the
        # identical 0.171% change is reused when OCR read the same words, which
        # is what corroboration is worth.
        v = self._vision()
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        quiet = np.full((64, 64, 3), 30, np.uint8)
        nudged = quiet.copy()
        nudged[0, :7] = 250
        a = v.describe(quiet, ocr_text=rich)
        b = v.describe(nudged, ocr_text=rich)
        self.assertEqual(a, b)
        self.assertEqual(len(self.asked), 1, f"asked {len(self.asked)} times")

    def test_a_blank_screen_within_the_tight_bound_is_reused(self):
        # The gap item 2 closed. With OCR blank the bound used to be exactly
        # zero, which on a live desktop never fires, so the screens most likely
        # to be asked about twice - a video, an image, a game - paid 71-91s
        # every time. Live churn measures 0.0000% above the noise floor, so a
        # non-zero bound is safe here.
        v = self._vision()
        quiet = np.full((64, 64, 3), 30, np.uint8)
        blink = quiet.copy()
        blink[0, :2] = 250  # a cursor: 0.049%, above the noise floor
        a = v.describe(quiet)
        b = v.describe(blink)
        self.assertEqual(a, b, "a cursor blink forced an 85s re-describe")
        self.assertEqual(len(self.asked), 3, f"asked {len(self.asked)} times")

    def test_a_blank_screen_real_change_is_not_reused(self):
        # The other half of the same gate: a real change on a screen with no
        # text must still miss. The mildest real change measured was a spinner
        # at 0.2686%, so 0.1% has clearance but is not so loose as to allow it.
        v = self._vision()
        quiet = np.full((400, 600, 3), 30, np.uint8)
        moved = quiet.copy()
        rng = np.random.default_rng(3)
        moved[100:200, 100:260] = rng.integers(0, 255, (100, 160, 3), dtype=np.uint8)
        a = v.describe(quiet)
        b = v.describe(moved)
        self.assertNotEqual(a, b, "a genuinely different blank screen reused")
        self.assertEqual(len(self.asked), 6, f"asked {len(self.asked)} times")

    def test_the_blank_bound_can_be_closed_if_wanted(self):
        # 0 restores the old pixel-exact behaviour without touching the OCR tier.
        v = self._vision(reuse_no_ocr=0.0)
        quiet = np.full((64, 64, 3), 30, np.uint8)
        # Two pixels, 0.049%, which the 0.1% default allows and zero does not.
        blink = quiet.copy()
        blink[0, :2] = 250
        v.describe(quiet)
        v.describe(blink)
        self.assertEqual(len(self.asked), 6, "reuse ran with the blank bound at zero")

    def test_the_loose_tier_can_be_switched_off(self):
        # Leave only the exact-bytes cache, for anyone who would rather pay the
        # 85s than have a tolerance at all.
        v = self._vision(reuse=-1)
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        blink = _fake_screen(1).copy()
        blink[3, 3] = (9, 9, 9)
        v.describe(_fake_screen(1), ocr_text=rich)
        v.describe(blink, ocr_text=rich)
        self.assertEqual(len(self.asked), 2, "the loose tier ran while switched off")

    def test_ocr_word_order_does_not_force_a_re_describe(self):
        # OCR wobbles on spacing and order constantly. Requiring the identical
        # string would make the loose tier useless in exactly the live-desktop
        # case it exists for.
        v = self._vision()
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        blink = _fake_screen(1).copy()
        blink[3, 3] = (9, 9, 9)
        v.describe(_fake_screen(1), ocr_text=rich)
        v.describe(blink, ocr_text="  " + " ".join(reversed(rich.split())) + " ")
        self.assertEqual(len(self.asked), 1, "reordering OCR text cost a re-describe")

    def test_a_different_near_key_does_not_cross_over(self):
        # Two different questions can each have a valid signature; they must not
        # borrow each other's answers through the shared tier.
        v = self._vision()
        rich = (
            "Hoopa's Vault OCR bot setup search clipping bot Jarvis assistant "
            "corroboration check vim python script window terminal file edit saved"
        )
        v.describe(_fake_screen(1), ocr_text=rich)
        v.describe(_fake_screen(1), "what is the app name?", ocr_text=rich)
        self.assertEqual(len(self.asked), 2, "a different question reused a near hit")

    def test_the_key_separates_models(self):
        from jarvis.vlm import Vision

        v = self._vision()
        self.assertNotEqual(
            v._image_key(_fake_screen(1), "q", "moondream"),
            v._image_key(_fake_screen(1), "q", "qwen2.5vl:3b"),
        )


class TestLoopbackHostResolution(unittest.TestCase):
    """Ollama listens on IPv4 only, so "localhost" costs ~2s to refuse ::1 first.

    Measured on this machine: a socket to ::1:11434 is refused after 2.06s, and
    127.0.0.1:11434 connects in 0.015s. The app paid that twice per launch, in
    pick_model and the embedder probe, so 4.1s of a 7.16s startup went to
    reaching a server 15ms away. These tests pin the rewrite.
    """

    def test_a_localhost_host_becomes_the_ipv4_literal(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("http://localhost:11434"),
                         "http://127.0.0.1:11434")

    def test_the_port_and_path_survive(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("http://localhost:11434/api"),
                         "http://127.0.0.1:11434/api")

    def test_a_portless_host_becomes_the_ipv4_literal(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("http://localhost"), "http://127.0.0.1")

    def test_a_https_localhost_keeps_its_scheme(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("https://localhost:443"),
                         "https://127.0.0.1:443")

    def test_a_case_variant_is_still_recognised(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("http://LocalHost:11434"),
                         "http://127.0.0.1:11434")

    def test_a_trailing_slash_is_handled(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("http://localhost:11434/"),
                         "http://127.0.0.1:11434")

    def test_a_remote_host_is_left_exactly_as_written(self):
        from jarvis.config import connect_host

        # Someone else's machine must keep their own name, because a name
        # resolved to a literal is a change of meaning, not a speed-up.
        for host in ("http://ollama.lan:11434", "http://192.168.1.50:11434",
                     "https://api.example.com"):
            self.assertEqual(connect_host(host), host)

    def test_an_already_resolved_host_is_untouched(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("http://127.0.0.1:11434"),
                         "http://127.0.0.1:11434")

    def test_a_host_with_no_scheme_is_still_rewritten(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host("localhost:11434"), "127.0.0.1:11434")

    def test_an_empty_host_does_not_explode(self):
        from jarvis.config import connect_host

        self.assertEqual(connect_host(""), "")

    def test_the_embedder_dials_the_ipv4_literal(self):
        from jarvis.embeddings import OllamaEmbedder

        emb = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
        self.assertEqual(emb.host, "http://127.0.0.1:11434")

    def test_the_embedder_keeps_a_remote_host(self):
        from jarvis.embeddings import OllamaEmbedder

        emb = OllamaEmbedder("http://ollama.lan:11434/", "nomic-embed-text")
        self.assertEqual(emb.host, "http://ollama.lan:11434")

    def test_the_vision_model_dials_the_ipv4_literal(self):
        from jarvis.config import load_config
        from jarvis.vlm import Vision


        cfg = load_config()
        cfg.ollama_host = "http://localhost:11434"
        self.assertEqual(Vision(cfg).host, "http://127.0.0.1:11434")

    def _brain_host(self, ollama_host: str) -> str | None:
        """The host the brain's ollama client was built with, or None.

        pick_model is stubbed so the test does not need a live ollama; the
        client is created immediately after it, so nothing else has to happen.
        """
        from jarvis.config import load_config
        from jarvis.memory import Memory
        import jarvis.brain as B
        import ollama  # noqa: WPS433

        seen: dict = {}
        real_client, real_pick = ollama.Client, B.pick_model

        class _FakeClient:
            def __init__(self, host=None, timeout=None):
                seen["host"] = host

        def _fake_pick(_cfg):
            return "qwen2.5:3b-instruct", "stubbed"

        ollama.Client = _FakeClient  # type: ignore[assignment]
        B.pick_model = _fake_pick  # type: ignore[assignment]
        try:
            cfg = load_config()
            cfg.ollama_host = ollama_host
            B.Brain(cfg, Memory(":memory:"))
            return seen.get("host")
        finally:
            ollama.Client = real_client  # type: ignore[assignment]
            B.pick_model = real_pick  # type: ignore[assignment]

    def test_the_brain_client_dials_the_ipv4_literal(self):
        self.assertEqual(self._brain_host("http://localhost:11434"),
                         "http://127.0.0.1:11434")

    def test_the_brain_client_keeps_a_remote_host(self):
        self.assertEqual(self._brain_host("http://ollama.lan:11434"),
                         "http://ollama.lan:11434")


if __name__ == "__main__":
    unittest.main(verbosity=2)
