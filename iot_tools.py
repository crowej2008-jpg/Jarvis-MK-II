"""Home automation: Home Assistant's REST API, plus generic MQTT.

Both integrations are optional. If the matching config is blank the tools
report a clear setup error instead of failing obscurely, so the model can tell
the user exactly what to fill in.
"""

from __future__ import annotations

import json
import time
from typing import Any

from . import ToolError, registry

_config = None


def bind_config(cfg) -> None:
    global _config
    _config = cfg


def _cfg():
    if _config is None:
        raise ToolError("configuration was not bound to the IoT tools")
    return _config


# ---------------------------------------------------------------------------
# Home Assistant
# ---------------------------------------------------------------------------
def _ha_url() -> str:
    url = (_cfg().home_assistant_url or "").strip()
    if not url:
        raise ToolError(
            "Home Assistant is not configured. Set home_assistant_url and "
            "home_assistant_token in %USERPROFILE%\\.jarvis\\config.json."
        )
    return url.rstrip("/")


def _ha_headers() -> dict[str, str]:
    token = (_cfg().home_assistant_token or "").strip()
    if not token:
        raise ToolError(
            "Home Assistant is not configured. Set home_assistant_token in "
            "%USERPROFILE%\\.jarvis\\config.json. A long-lived access token "
            "can be made under your HA profile page."
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _ha(method: str, path: str, payload: Any = None, timeout: float = 12.0) -> Any:
    import requests

    url = f"{_ha_url()}/api/{path.lstrip('/')}"
    try:
        resp = requests.request(
            method, url, headers=_ha_headers(), json=payload, timeout=timeout
        )
    except requests.RequestException as exc:
        raise ToolError(f"could not reach Home Assistant at {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise ToolError(
            f"Home Assistant returned {resp.status_code}: {resp.text[:300]}"
        )
    if not resp.content:
        return {"ok": True}
    try:
        return resp.json()
    except ValueError:
        return {"ok": True, "text": resp.text[:500]}


def _entity_id(entity: str) -> str:
    """Resolve what the model said into a real entity id.

    A proper id like 'light.kitchen' is passed straight through. Anything else
    is matched against the entities Home Assistant actually has, by object id
    and by friendly name, because the alternative this used to take -
    lowercasing and replacing the spaces - turned "the kitchen lights" into
    "the_kitchen_lights", which is not an entity id at all. The old docstring
    promised a lookup that did not exist; now it does, and an unresolvable
    phrase is reported instead of sent.
    """
    cleaned = str(entity).strip()
    if not cleaned:
        raise ToolError("no entity given.")
    if "." in cleaned:
        return cleaned

    needle = cleaned.lower()
    bare = needle.removeprefix("the ").strip()
    try:
        states = _ha("GET", "/states")
    except ToolError:
        states = []

    exact, loose = [], []
    for state in states:
        eid = str(state.get("entity_id", ""))
        if not eid:
            continue
        object_id = eid.split(".", 1)[1] if "." in eid else eid
        label = str(state.get("attributes", {}).get("friendly_name", "") or "")
        names = {object_id.lower(), eid.lower()}
        if label:
            names.add(label.lower())
        for candidate in (bare, needle):
            if not candidate:
                continue
            if candidate in names:
                exact.append(eid)
                break
        else:
            if any(bare in name or needle in name for name in names if name):
                loose.append(eid)

    for candidates, kind in ((exact, "matches"), (loose, "resembles")):
        if not candidates:
            continue
        found = sorted(set(candidates))
        if len(found) == 1:
            return found[0]
        # "the lights" matches every light in the house. Guessing one would
        # quietly change the wrong room, so the ambiguity is reported instead.
        raise ToolError(
            f"{entity!r} {kind} {len(found)} entities: {', '.join(found[:20])}. "
            "Be specific, e.g. 'light.kitchen'."
        )

    known = sorted(
        str(s.get("entity_id", ""))
        for s in states
        if str(s.get("entity_id", ""))
    )[:20]
    hint = f" Known entities: {', '.join(known)}." if known else ""
    raise ToolError(
        f"no Home Assistant entity matches {entity!r}. Use ha_list_entities to see "
        f"what exists, or give the entity id directly, e.g. 'light.kitchen'.{hint}"
    )


@registry.add(
    "ha_list_entities",
    "List Home Assistant entities, optionally filtered to one domain "
    "(light, switch, sensor, climate, media_player, cover, ...).",
    {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Filter to a domain, e.g. 'light'. Empty for all.",
            },
            "search": {"type": "string", "description": "Filter by name or entity id."},
        },
    },
    tags=("home",),
)
def ha_list_entities(domain: str = "", search: str = "") -> dict[str, Any]:
    states = _ha("GET", "/states")
    rows = []
    needle = search.strip().lower()
    for state in states:
        eid = state.get("entity_id", "")
        if domain and not eid.startswith(f"{domain.lower()}."):
            continue
        attrs = state.get("attributes", {})
        label = attrs.get("friendly_name", eid)
        if needle and needle not in f"{eid} {label}".lower():
            continue
        row = {
            "entity_id": eid,
            "name": label,
            "state": state.get("state"),
        }
        for key in ("unit_of_measurement", "brightness", "current_temperature"):
            if key in attrs:
                row[key] = attrs[key]
        rows.append(row)
    rows.sort(key=lambda r: r["entity_id"])
    return {"count": len(rows), "entities": rows[:200]}


@registry.add(
    "ha_get_state",
    "Read the current state of one Home Assistant entity.",
    {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "An entity id like 'light.kitchen', or a name "
                "like 'the porch lights'.",
            }
        },
        "required": ["entity"],
    },
    tags=("home",),
)
def ha_get_state(entity: str) -> dict[str, Any]:
    eid = _entity_id(entity)
    state = _ha("GET", f"/states/{eid}")
    attrs = state.get("attributes", {})
    return {
        "entity_id": state.get("entity_id", eid),
        "name": attrs.get("friendly_name", eid),
        "state": state.get("state"),
        "attributes": {
            k: v for k, v in attrs.items() if k != "friendly_name"
        },
        "last_changed": state.get("last_changed"),
    }


@registry.add(
    "ha_call_service",
    "Call a Home Assistant service, e.g. turn on a light, set a thermostat, "
    "or run a script or scene. Use this to actually change the house.",
    {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Service domain, e.g. 'light', 'switch', 'climate', 'script'.",
            },
            "service": {
                "type": "string",
                "description": "Service name, e.g. 'turn_on', 'turn_off', 'set_temperature'.",
            },
            "entity": {
                "type": "string",
                "description": "Target entity. An id like 'light.kitchen' or a "
                "name like 'the porch lights', which is resolved against the "
                "entities Home Assistant reports. May be empty for "
                "domain-wide calls.",
            },
            "data": {
                "type": "object",
                "description": "Extra service data, e.g. {\"brightness_pct\": 40}.",
            },
        },
        "required": ["domain", "service"],
    },
    dangerous=True,
    tags=("home",),
)
def ha_call_service(
    domain: str, service: str, entity: str = "", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    from . import system_tools

    domain, service = str(domain).strip(), str(service).strip()
    payload = dict(data or {})
    if isinstance(entity, dict):
        # Writing the Home Assistant service body into `entity` instead of
        # `data` is the natural mistake, and it used to surface as an
        # AttributeError from deep inside string handling. The intent is
        # unambiguous, so honour it rather than failing obscurely.
        stray = dict(entity)
        target = stray.pop("entity_id", None)
        if target is None:
            raise ToolError(
                "entity was given as an object with no 'entity_id' in it. Pass "
                "entity_id as a string, or put the rest in data."
            )
        for key, value in stray.items():
            payload.setdefault(key, value)
        entity = str(target)
    elif not isinstance(entity, str):
        raise ToolError(
            f"entity must be an entity id string, got {type(entity).__name__}."
        )

    label = f"{domain}.{service}"
    if entity:
        label += f" on {_entity_id(entity)}"
    system_tools._approve("ha_call_service", f'You asked me to run the "{label}" service.')

    path = f"/services/{domain}/{service}"
    if entity:
        payload["entity_id"] = _entity_id(entity)
    result = _ha("POST", path, payload)
    return {"called": label, "payload": payload, "result": result}


@registry.add(
    "ha_run_script",
    "Trigger a Home Assistant script, scene, or automation by name.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Script, scene, or automation name."}
        },
        "required": ["name"],
    },
    dangerous=True,
    tags=("home",),
)
def ha_run_script(name: str) -> dict[str, Any]:
    states = _ha("GET", "/states")
    needle = str(name).strip().lower()
    if not needle:
        raise ToolError("no script name given.")

    # Exact matches first, then substring. Every identifier a script can be
    # called by is considered: the full entity id, the object id after the dot,
    # and the friendly name. Reading the friendly name with an empty default,
    # as this did, made every script without one unfindable by its own name.
    exact: list[str] = []
    loose: list[str] = []
    for state in states:
        eid = str(state.get("entity_id", ""))
        if not eid.startswith(("script.", "scene.")):
            continue
        object_id = eid.split(".", 1)[1]
        label = str(state.get("attributes", {}).get("friendly_name", "") or "")
        names = {eid.lower(), object_id.lower()}
        if label:
            names.add(label.lower())
        if needle in names:
            exact.append(eid)
        elif any(needle in candidate for candidate in names):
            loose.append(eid)

    for eid in exact + loose:
        return ha_call_service(eid.split(".", 1)[0], "turn_on", eid)
    known = sorted(
        str(s.get("entity_id", ""))
        for s in states
        if str(s.get("entity_id", "")).startswith(("script.", "scene."))
    )
    raise ToolError(
        f"no script or scene matching {name!r}."
        + (f" Available: {', '.join(known[:20])}." if known else
           " Call ha_list_entities with domain='script' to see what exists.")
    )


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
def _mqtt_client(timeout: float = 10.0):
    """Build a client and check the configuration. Does not connect.

    Connecting is left to _mqtt_open so that the caller can install its
    on_connect first: paho's connect() reads the broker's CONNACK
    synchronously, so a handler attached afterwards never fires.
    """
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise ToolError(
            f"paho-mqtt is not installed: {exc}. Run: pip install paho-mqtt"
        ) from exc

    cfg = _cfg()
    if not (cfg.mqtt_host or "").strip():
        raise ToolError(
            "MQTT is not configured. Set mqtt_host in "
            "%USERPROFILE%\\.jarvis\\config.json."
        )

    try:
        # CallbackAPIVersion.VERSION1 keeps this working on paho-mqtt 1.x and 2.x.
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            client = mqtt.Client()
        if cfg.mqtt_username:
            client.username_pw_set(cfg.mqtt_username, cfg.mqtt_password or None)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not set up the MQTT client: {exc}") from exc
    return client, mqtt


def _mqtt_open(client, mqtt, on_connect=None, timeout: float = 10.0) -> int:
    """Connect, start the network loop, and wait for the broker's CONNACK.

    Returns the CONNACK return code, 0 for success. Starting the loop is not
    optional: paho only reads the socket from that thread, so without it a QoS 1
    or 2 publish never completes its handshake and reports a delivery the broker
    never acknowledged. Bad credentials are likewise invisible until CONNACK
    arrives, and paho signals them by return code rather than by raising, so a
    publish against a rejecting broker used to report plain success.
    """
    cfg = _cfg()
    seen: list[int] = []

    def handle(_c, _userdata, _flags, rc, *args):
        seen.append(int(rc))
        if int(rc) == 0 and on_connect is not None:
            on_connect(_c, _userdata, _flags, rc, *args)

    client.on_connect = handle
    try:
        rc = client.connect(cfg.mqtt_host, int(cfg.mqtt_port), keepalive=30)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(
            f"could not connect to MQTT broker at {cfg.mqtt_host}:{cfg.mqtt_port}: {exc}"
        ) from exc
    if int(rc) != 0:
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        raise ToolError(
            f"could not connect to MQTT broker at {cfg.mqtt_host}:{cfg.mqtt_port}: {rc}"
        )

    client.loop_start()
    deadline = time.time() + timeout
    while not seen and time.time() < deadline:
        time.sleep(0.01)
    return seen[0] if seen else -1


_MQTT_CONNACK = {
    1: "the broker refused the connection as unacceptable",
    2: "the broker rejected the client identifier",
    3: "the broker is unavailable",
    4: "the broker rejected the username or password",
    5: "the broker did not authorise this client",
}


def _mqtt_auth_error(client, mqtt, rc: int) -> ToolError:
    client.disconnect()
    try:
        client.loop_stop()
    except Exception:  # noqa: BLE001
        pass
    why = _MQTT_CONNACK.get(rc, f"the broker returned connection code {rc}")
    return ToolError(
        f"the MQTT broker at {_cfg().mqtt_host}:{_cfg().mqtt_port} rejected the "
        f"connection: {why} (code {rc}). Check mqtt_username and mqtt_password."
    )


@registry.add(
    "mqtt_publish",
    "Publish a message to an MQTT topic, e.g. to trigger a device or scene.",
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Full topic to publish to."},
            "payload": {
                "type": "string",
                "description": "Message body. JSON strings are sent as-is.",
            },
            "retain": {"type": "boolean"},
            "qos": {"type": "integer", "description": "0, 1, or 2."},
        },
        "required": ["topic", "payload"],
    },
    dangerous=True,
    tags=("mqtt",),
)
def mqtt_publish(
    topic: str, payload: str, retain: bool = False, qos: int = 0
) -> dict[str, Any]:
    from . import system_tools

    system_tools._approve("mqtt_publish", f'You asked me to publish to "{topic}".')
    client, mqtt = _mqtt_client()
    try:
        rc = _mqtt_open(client, mqtt)
        if rc != 0:
            raise _mqtt_auth_error(client, mqtt, rc)
        info = client.publish(topic, payload, qos=int(qos), retain=bool(retain))
        info.wait_for_publish(timeout=8)
        delivered = info.is_published()
    finally:
        # Disconnect before stopping the loop. Stopping first makes paho's
        # network thread block until its select times out, which cost a full
        # second on every single publish and listen.
        client.disconnect()
        try:
            client.loop_stop()
        except Exception:  # noqa: BLE001
            pass

    if not delivered:
        # QoS 0 is fire-and-forget, so this is only reachable at QoS 1 or 2,
        # where the broker never acknowledged. Saying so beats reporting
        # delivered=false, which reads to the model as a success it may relay.
        raise ToolError(
            f'the broker at {_cfg().mqtt_host} did not acknowledge the publish to '
            f'"{topic}" at QoS {qos}. The message may or may not have been acted on.'
        )
    return {
        "topic": topic,
        "bytes": len(payload.encode("utf-8")),
        "delivered": True,
        "retain": bool(retain),
    }


@registry.add(
    "mqtt_listen",
    "Listen on an MQTT topic for a short while and return the first message "
    "seen. Good for checking a sensor's current value.",
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Topic or wildcard, e.g. 'home/+/state'."},
            "timeout": {
                "type": "number",
                "description": "Seconds to listen. Default 5.",
            },
        },
        "required": ["topic"],
    },
    tags=("mqtt",),
)
def mqtt_listen(topic: str, timeout: float = 5.0) -> dict[str, Any]:
    client, mqtt = _mqtt_client()
    received: list[dict[str, Any]] = []
    refused: list[int] = []
    deadline = time.time() + max(1.0, min(float(timeout), 60.0))

    def subscribe(_c, _u, _f, _rc, *_a):
        _c.subscribe(topic)

    def on_message(_c, _u, msg):
        try:
            body = msg.payload.decode("utf-8", errors="replace")
            parsed: Any = json.loads(body)
        except Exception:  # noqa: BLE001
            parsed = None
        received.append({"topic": msg.topic, "payload": parsed if parsed is not None else body})

    client.on_message = on_message
    try:
        rc = _mqtt_open(client, mqtt, on_connect=subscribe)
        if rc != 0:
            refused.append(rc)
        while time.time() < deadline and not received and not refused:
            time.sleep(0.02)
    finally:
        client.disconnect()
        try:
            client.loop_stop()
        except Exception:  # noqa: BLE001
            pass

    if refused:
        raise _mqtt_auth_error(client, mqtt, refused[0])

    if not received:
        return {"topic": topic, "messages": [], "note": "Nothing published in that window."}
    return {"topic": topic, "count": len(received), "messages": received[:20]}
