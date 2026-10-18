"""Loopback-only HTTP API and static-file host for the local farm simulation.

Run with ``python3 -m farm_sim.server``. The service has no hosted inference
dependency and refuses to start unless the validated local policy checkpoint is
available. Physiological time advances only in discrete simulation ticks;
``speed`` changes wall-clock pacing, never the transition timestep.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .memory import retrieve
from .model import FEATURE_NAMES
from .perception import encode_features, perceive
from .policy import CategoricalPolicy, load_checkpoint
from .simulation import Simulation, make_world, serialize_world, synthetic_time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "policies" / "default-policy.npz"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
LOOPBACK_HOST = "127.0.0.1"
MAX_BODY_BYTES = 1_000_000
ANIMAL_PATH = re.compile(r"^/api/animals/([^/]+)$")


class FarmController:
    """Serialize API and clock access to one authoritative Simulation."""

    def __init__(self, policy: CategoricalPolicy, *, seed: int = 7):
        if policy is None:
            raise ValueError("a validated local policy is required; no heuristic fallback is available")
        self.policy = policy
        self.checkpoint_id = policy.policy_id
        self.simulation = Simulation(make_world(seed), policy)
        self.lock = threading.RLock()
        self.paused = True
        self.speed = 1.0
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._clock_thread: threading.Thread | None = None

    def start_clock(self) -> None:
        if self._clock_thread is None:
            self._clock_thread = threading.Thread(target=self._clock_loop, name="farm-clock", daemon=True)
            self._clock_thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
        if self._clock_thread is not None:
            self._clock_thread.join(timeout=2)

    def _clock_loop(self) -> None:
        while not self._closed.is_set():
            with self.lock:
                paused, delay = self.paused, 1.0 / self.speed
            if paused:
                self._wake.wait()
                self._wake.clear()
                continue
            # A control change wakes the wait so pause/speed changes take effect
            # promptly. A wake does not itself advance logical time.
            if self._wake.wait(delay):
                self._wake.clear()
                continue
            with self.lock:
                if not self.paused:
                    self.simulation.step()

    def state(self) -> dict[str, Any]:
        with self.lock:
            world = self.simulation.world
            result = serialize_world(world)
            result.update({"paused": self.paused, "speed": self.speed,
                           "policy_id": self.checkpoint_id,
                           "clock": {**synthetic_time(world.tick), "tick": world.tick,
                                     "paused": self.paused, "speed": self.speed,
                                     "logical_dt": 1, "action_duration_ticks": 1,
                                     "time_unit": "abstract_simulation_tick",
                                     "calendar": "synthetic_virtual_day"},
                           "event_retention": self.simulation.event_retention()})
            result["animals"] = [self._animal_view(a.animal_id) for a in world.animals]
            return result

    def _animal_view(self, animal_id: str) -> dict[str, Any] | None:
        world = self.simulation.world
        animal = next((a for a in world.animals if a.animal_id == animal_id), None)
        if animal is None:
            return None
        observation = perceive(world, animal)
        memory = retrieve(animal.memory, world.tick)
        current_features = encode_features(observation, memory, animal, world.width, world.height)
        has_decision = bool(animal.last_probabilities)
        probabilities = animal.last_probabilities or self.policy.probabilities(current_features)
        observed = {
            "tick": observation.tick, "animal_id": observation.animal_id,
            "position": {"x": observation.position.x, "y": observation.position.y},
            "visible_forage": [_patch(p) for p in observation.visible_forage],
            "visible_water": [_patch(p) for p in observation.visible_water],
            "nearest_forage_distance": observation.nearest_forage_distance,
            "nearest_water_distance": observation.nearest_water_distance,
            "local_herd_count": observation.local_herd_count,
            "local_herd_positions": [_position(p) for p in observation.local_herd_positions],
            "local_blocked_fraction": observation.local_blocked_fraction,
        }
        serialized = next(x for x in serialize_world(world)["animals"] if x["id"] == animal_id)
        serialized.update({
            "policy_id": self.checkpoint_id,
            "observation": observed,
            "feature_names": list(FEATURE_NAMES),
            # `features` remains an alias for the current, freshly observed inputs.
            # Decision inputs are retained separately so probabilities and action
            # can always be traced to the exact pre-transition vector they used.
            "features": list(current_features),
            "current_features": list(current_features),
            "decision_tick": world.tick - 1 if has_decision else None,
            "decision_features": list(animal.last_features) if has_decision else None,
            "decision_probabilities": list(animal.last_probabilities) if has_decision else None,
            "memory_features": {
                "recent_water_success": memory.recent_water_success,
                "recent_forage_success": memory.recent_forage_success,
                "water_access_blocked": memory.water_access_blocked,
                "forage_depleted": memory.forage_depleted,
                "event_recency": memory.event_recency,
                "retrieved_events": [_memory_event(e) for e in memory.retrieved_events],
            },
            "probabilities": list(probabilities),
            "last_events": [self._event_dict(event, event_id) for event_id, event
                            in zip(self.simulation.event_ids, self.simulation.events)
                            if event.animal_id == animal_id][-12:],
        })
        return serialized

    @staticmethod
    def _event_dict(event, event_id: int) -> dict[str, Any]:
        return {"id": event_id, "tick": event.tick, "animal_id": event.animal_id,
                "kind": event.kind, "detail": event.detail, "reward": event.reward}

    def events(self, after: int = -1) -> dict[str, Any]:
        with self.lock:
            events = [self._event_dict(event, event_id)
                      for event_id, event in zip(self.simulation.event_ids, self.simulation.events)
                      if event_id > after]
            return {"events": events, "retention": self.simulation.event_retention(after)}

    def animal(self, animal_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self._animal_view(animal_id)

    def reset(self, seed: int = 7) -> dict[str, Any]:
        with self.lock:
            world = self.simulation.reset(seed)
            return {"ok": True, "tick": world.tick, **synthetic_time(world.tick), "seed": world.seed,
                    "policy_id": self.checkpoint_id, "animal_count": len(world.animals)}

    def step(self) -> dict[str, Any]:
        with self.lock:
            result = self.simulation.step()
            return {"ok": True, "tick": result.world.tick,
                    **synthetic_time(result.world.tick), "events_added": len(result.events),
                    "state": self.state()}

    def control(self, action: str, speed: float | None = None) -> dict[str, Any]:
        with self.lock:
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action != "speed":
                raise ValueError("action must be pause, resume, or speed")
            if action == "speed" or speed is not None:
                if speed is None or isinstance(speed, bool) or not isinstance(speed, (int, float)):
                    raise ValueError("speed must be a number between 0.1 and 20")
                if not 0.1 <= float(speed) <= 20:
                    raise ValueError("speed must be between 0.1 and 20")
                self.speed = float(speed)
            self._wake.set()
            tick = self.simulation.world.tick
            return {"ok": True, "paused": self.paused, "speed": self.speed,
                    "tick": tick, **synthetic_time(tick), "logical_dt": 1,
                    "action_duration_ticks": 1}


def _position(p) -> dict[str, int]:
    return {"x": p.x, "y": p.y}


def _patch(p) -> dict[str, Any]:
    return {"x": p.position.x, "y": p.position.y, "available": p.available}


def _memory_event(e) -> dict[str, Any]:
    return {"tick": e.tick, "kind": e.kind, "success": e.success,
            "x": e.position.x, "y": e.position.y}


def create_server(*, host: str = LOOPBACK_HOST, port: int = 8000,
                  checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
                  static_dir: str | Path = STATIC_ROOT,
                  policy: CategoricalPolicy | None = None) -> ThreadingHTTPServer:
    """Build a server, loading and validating the default checkpoint by default."""
    if host != LOOPBACK_HOST:
        raise ValueError("farm server may bind only to 127.0.0.1")
    # ``policy`` is an explicit test seam; production startup always loads the
    # strict NPZ+JSON checkpoint and never substitutes a baseline.
    if policy is None:
        policy = load_checkpoint(checkpoint_path)
    controller = FarmController(policy)
    static_root = Path(static_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalFarm/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Avoid logging potentially sensitive query/body content by default.
            return

        def _send(self, status: int, value: Any, *, content_type: str = "application/json; charset=utf-8") -> None:
            if content_type.startswith("application/json"):
                body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
            elif isinstance(value, bytes):
                body = value
            else:
                body = str(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": message})

        def _body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("request body exceeds 1000000 bytes")
            if length == 0:
                return {}
            try:
                value = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/api/state":
                self._send(200, controller.state())
            elif path == "/api/events":
                try:
                    values = parse_qs(parsed.query).get("after", ["-1"])
                    after = int(values[-1])
                    if after < -1:
                        raise ValueError
                except ValueError:
                    self._error(400, "after must be an integer greater than or equal to -1")
                    return
                self._send(200, controller.events(after))
            else:
                match = ANIMAL_PATH.fullmatch(path)
                if match:
                    animal = controller.animal(unquote(match.group(1)))
                    if animal is None:
                        self._error(404, "animal not found")
                    else:
                        self._send(200, animal)
                    return
                self._serve_static(path)

        def _serve_static(self, path: str) -> None:
            if not static_root.is_dir() or not (static_root / "index.html").is_file():
                self._error(503, "browser files are not installed yet (expected farm_sim/static)")
                return
            relative = "index.html" if path == "/" else path.lstrip("/")
            target = (static_root / relative).resolve()
            if target != static_root and static_root not in target.parents:
                self._error(404, "file not found")
                return
            if not target.is_file():
                self._error(404, "file not found")
                return
            try:
                data = target.read_bytes()
            except OSError:
                self._error(500, "static file could not be read")
                return
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                mime += "; charset=utf-8"
            self._send(200, data, content_type=mime)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            try:
                body = self._body()
                if path == "/api/reset":
                    seed = body.get("seed", 7)
                    if isinstance(seed, bool) or not isinstance(seed, int):
                        raise ValueError("seed must be an integer")
                    self._send(200, controller.reset(seed))
                elif path == "/api/step":
                    if body:
                        raise ValueError("step takes an empty JSON object")
                    self._send(200, controller.step())
                elif path == "/api/control":
                    action = body.get("action")
                    if not isinstance(action, str):
                        raise ValueError("action must be pause, resume, or speed")
                    self._send(200, controller.control(action, body.get("speed")))
                else:
                    self._error(404, "endpoint not found")
            except ValueError as exc:
                self._error(400, str(exc))
            except (TypeError, OverflowError) as exc:
                self._error(400, str(exc))

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.farm_controller = controller  # type: ignore[attr-defined]
    controller.start_clock()
    return server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    try:
        server = create_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Unable to start local farm server: {exc}") from exc
    print(f"Farm simulation available at http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.farm_controller.close()  # type: ignore[attr-defined]
        server.server_close()


if __name__ == "__main__":
    main()
