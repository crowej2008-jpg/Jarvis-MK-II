"""Real local servers used by the tests, so the IoT tools are tested honestly.

Mocking paho-mqtt or requests would only prove the mock works. The Home
Assistant tools talk HTTP, so these tests run a real HTTP server on loopback.
The MQTT tools speak a binary protocol, so these tests run a real MQTT 3.1.1
broker on loopback and assert against what actually arrived on the wire.

Nothing here reaches the network beyond 127.0.0.1 and no real device, broker or
Home Assistant instance is contacted.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# MQTT control packet types.
CONNECT, CONNACK, PUBLISH, PUBACK = 1, 2, 3, 4
PUBREC, PUBREL, PUBCOMP = 5, 6, 7
SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK = 8, 9, 10, 11
PINGREQ, PINGRESP, DISCONNECT = 12, 13, 14


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
def _encode_remaining(length: int) -> bytes:
    out = bytearray()
    while True:
        byte = length % 128
        length //= 128
        if length:
            byte |= 0x80
        out.append(byte)
        if not length:
            return bytes(out)


def _encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("!H", len(raw)) + raw


def topic_matches(filt: str, topic: str) -> bool:
    """MQTT topic filter matching, including the + and # wildcards."""
    if filt == topic:
        return True
    parts, actual = filt.split("/"), topic.split("/")
    for i, part in enumerate(parts):
        if part == "#":
            return i <= len(actual)
        if i >= len(actual):
            return False
        if part == "+":
            continue
        if part != actual[i]:
            return False
    return len(parts) == len(actual)


class MqttBroker:
    """A small but real MQTT 3.1.1 broker, on loopback, for tests only."""

    def __init__(self, username: str = "", password: str = ""):
        self.username = username
        self.password = password
        self.retained: dict[str, bytes] = {}
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.auth_failures = 0
        self._subscribers: list[tuple[str, object]] = []
        self._lock = threading.Lock()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port: int = self._sock.getsockname()[1]
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # -- lifecycle --------------------------------------------------------
    def stop(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    # -- server side publishing -------------------------------------------
    def wait_for_subscription(self, filt: str, timeout: float = 5.0) -> bool:
        """Block until a client has actually subscribed.

        Publishing before the SUBSCRIBE lands silently drops the message, which
        would make these tests race rather than test anything.
        """
        import time as timemod

        deadline = timemod.time() + timeout
        while timemod.time() < deadline:
            with self._lock:
                if any(f == filt for f, _ in self._subscribers):
                    return True
            timemod.sleep(0.02)
        return False

    def publish(self, topic: str, payload, qos: int = 0, retain: bool = False) -> int:
        """Push a message as if a device had sent it. Returns subscriber count."""
        import queue as queuemod

        body = payload.encode("utf-8") if isinstance(payload, str) else payload
        del qos
        with self._lock:
            targets = [q for f, q in self._subscribers if topic_matches(f, topic)]
        for q in targets:
            q.put((topic, body, False))
        return len(targets)

    # -- internals ---------------------------------------------------------
    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    @staticmethod
    def _read_exact(conn, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf

    def _read_packet(self, conn):
        header = self._read_exact(conn, 1)[0]
        multiplier, remaining = 1, 0
        while True:
            byte = self._read_exact(conn, 1)[0]
            remaining += (byte & 0x7F) * multiplier
            if not byte & 0x80:
                break
            multiplier *= 128
            if multiplier > 128 ** 3:
                raise ValueError("malformed remaining length")
        return header, (self._read_exact(conn, remaining) if remaining else b"")

    def _session(self, conn) -> None:
        import queue as queuemod

        inbox: queuemod.Queue = queuemod.Queue()
        filters: list[str] = []
        alive = threading.Event()
        alive.set()
        # Every write to the socket goes through here. Two threads writing to
        # one socket can interleave, and a retained PUBLISH overtaking its own
        # SUBACK makes a real client discard the message.
        send_lock = threading.Lock()

        def send(data: bytes) -> bool:
            with send_lock:
                try:
                    conn.sendall(data)
                    return True
                except OSError:
                    return False

        def writer():
            while alive.is_set():
                try:
                    item = inbox.get(timeout=0.2)
                except Exception:  # noqa: BLE001
                    continue
                if item is None:
                    continue
                topic, body, is_retained = item
                packet = (
                    bytes([(PUBLISH << 4) | (1 if is_retained else 0)])
                    + _encode_remaining(2 + len(topic) + len(body))
                    + _encode_string(topic)
                    + body
                )
                if not send(packet):
                    return

        threading.Thread(target=writer, daemon=True).start()
        try:
            while not self._stop:
                try:
                    header, body = self._read_packet(conn)
                except (ConnectionError, OSError, ValueError):
                    break
                kind = header >> 4

                if kind == CONNECT:
                    if not self._check_auth(body):
                        self.auth_failures += 1
                        send(bytes([CONNACK << 4, 2, 0, 5]))
                        break
                    send(bytes([CONNACK << 4, 2, 0, 0]))

                elif kind == SUBSCRIBE:
                    pid = struct.unpack("!H", body[:2])[0]
                    pos, codes, subs = 2, bytearray(), []
                    while pos < len(body):
                        n = struct.unpack("!H", body[pos:pos + 2])[0]
                        subs.append(body[pos + 2:pos + 2 + n].decode("utf-8"))
                        pos += 2 + n + 1
                        codes.append(body[pos - 1] & 0x03)
                    with self._lock:
                        for filt in subs:
                            self._subscribers.append((filt, inbox))
                    filters = subs
                    ack = (bytes([SUBACK << 4]) + _encode_remaining(2 + len(codes))
                           + struct.pack("!H", pid) + bytes(codes or [0]))
                    # Retained messages are sent here, in order, straight after
                    # the SUBACK, rather than via the writer thread.
                    for filt in subs:
                        with self._lock:
                            held = [(t, p) for t, p in self.retained.items()
                                    if topic_matches(filt, t)]
                        for held_topic, held_payload in held:
                            packet = (
                                bytes([(PUBLISH << 4) | 0x01])
                                + _encode_remaining(2 + len(held_topic)
                                                    + len(held_payload))
                                + _encode_string(held_topic)
                                + held_payload
                            )
                            if not send(ack + packet):
                                break
                            ack = b""
                    if ack and not send(ack):
                        break

                elif kind == UNSUBSCRIBE:
                    pid = struct.unpack("!H", body[:2])[0]
                    with self._lock:
                        self._subscribers = [s for s in self._subscribers
                                             if s[0] not in filters]
                    if not send(bytes([UNSUBACK << 4, 2]) + struct.pack("!H", pid)):
                        break

                elif kind == PUBLISH:
                    qos = (header >> 1) & 0x03
                    retain = bool(header & 0x01)
                    n = struct.unpack("!H", body[:2])[0]
                    topic = body[2:2 + n].decode("utf-8")
                    pos = 2 + n
                    pid = None
                    if qos:
                        pid = struct.unpack("!H", body[pos:pos + 2])[0]
                        pos += 2
                    payload = body[pos:]
                    self.published.append((topic, payload, qos, retain))
                    if retain:
                        with self._lock:
                            if payload:
                                self.retained[topic] = payload
                            else:
                                self.retained.pop(topic, None)
                    with self._lock:
                        targets = [q for f, q in self._subscribers
                                   if topic_matches(f, topic)]
                    for q in targets:
                        q.put((topic, payload, False))
                    if qos == 1:
                        send(bytes([PUBACK << 4, 2]) + struct.pack("!H", pid))
                    elif qos == 2:
                        send(bytes([PUBREC << 4, 2]) + struct.pack("!H", pid))

                elif kind == PUBREL:
                    pid = struct.unpack("!H", body[:2])[0]
                    send(bytes([PUBCOMP << 4, 2]) + struct.pack("!H", pid))

                elif kind == PINGREQ:
                    send(bytes([PINGRESP << 4, 0]))

                elif kind == DISCONNECT:
                    break
        finally:
            alive.clear()
            with self._lock:
                self._subscribers = [s for s in self._subscribers
                                     if s[0] not in filters]
            try:
                conn.close()
            except OSError:
                pass

    def _check_auth(self, body: bytes) -> bool:
        if not self.username:
            return True
        pos = 0
        n = struct.unpack("!H", body[pos:pos + 2])[0]
        pos += 2 + n  # protocol name
        pos += 1  # protocol level
        flags = body[pos]
        pos += 3  # connect flags byte, then keep alive
        pid = struct.unpack("!H", body[pos:pos + 2])[0]
        pos += 2 + pid  # client id
        if flags & 0x04:
            for _ in range(2):
                n = struct.unpack("!H", body[pos:pos + 2])[0]
                pos += 2 + n
        if not flags & 0x80:
            return False
        n = struct.unpack("!H", body[pos:pos + 2])[0]
        user = body[pos + 2:pos + 2 + n].decode("utf-8")
        pos += 2 + n
        password = ""
        if flags & 0x40:
            n = struct.unpack("!H", body[pos:pos + 2])[0]
            password = body[pos + 2:pos + 2 + n].decode("utf-8")
        return user == self.username and password == (self.password or "")


# ---------------------------------------------------------------------------
# Home Assistant
# ---------------------------------------------------------------------------
class _FakeHomeAssistant(BaseHTTPRequestHandler):
    """Just enough of the Home Assistant API to exercise the tools."""

    token = "good-token"
    calls: list[tuple[str, str, object]] = []
    states: list[dict] = []

    def log_message(self, *args):
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            return self._send(401, {"message": "Unauthorized"})
        type(self).calls.append(("GET", self.path, None))
        if self.path == "/api/states":
            return self._send(200, self.states)
        if self.path.startswith("/api/states/"):
            entity_id = self.path.rsplit("/", 1)[-1]
            for state in self.states:
                if state.get("entity_id") == entity_id:
                    return self._send(200, state)
            return self._send(404, {"message": "Entity not found."})
        return self._send(404, {"message": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self._read_exact_body(length) or b"{}") if length else {}
        type(self).calls.append(("POST", self.path, body))
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            return self._send(401, {"message": "Unauthorized"})
        if "/error" in self.path:
            return self._send(500, {"message": "kaboom"})
        if self.path.startswith("/api/services/"):
            entity_id = body.get("entity_id", "")
            return self._send(200, [{"entity_id": entity_id,
                                     "state": self.state_after(entity_id),
                                     "attributes": {}}])
        return self._send(404, {"message": "not found"})

    def _read_exact_body(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.rfile.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    @staticmethod
    def state_after(entity_id: str) -> str:
        return "off" if entity_id.endswith(("goodnight", ".off")) else "on"


class FakeHomeAssistant:
    """A real loopback HTTP server speaking the Home Assistant API."""

    def __init__(self, states: list[dict] | None = None, token: str = "good-token"):
        handler = type("Handler", (_FakeHomeAssistant,), {
            "token": token,
            "calls": [],
            "states": states if states is not None else [
                {"entity_id": "light.porch", "state": "off",
                 "attributes": {"friendly_name": "Porch"}},
                {"entity_id": "sensor.outside_temp", "state": "7.5",
                 "attributes": {"unit_of_measurement": "°C"}},
                # No friendly_name: scripts like this used to be unfindable.
                {"entity_id": "script.goodnight", "state": "off", "attributes": {}},
                {"entity_id": "scene.movie", "state": "unknown",
                 "attributes": {"friendly_name": "Movie Time"}},
            ],
        })
        self.calls = handler.calls
        self.states = handler.states
        # Threaded, because requests keeps connections alive and a
        # single-threaded server then makes the next request wait for a
        # connection that is not going to close.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def paths(self) -> list[str]:
        return [path for _, path, _ in self.calls]
