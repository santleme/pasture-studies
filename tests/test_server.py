"""Loopback HTTP contract tests for the browser-facing local API."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from farm_sim.server import DEFAULT_CHECKPOINT, create_server
from farm_sim.model import SimEvent
from farm_sim.simulation import EVENT_HISTORY_LIMIT


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(port=0, checkpoint_path=DEFAULT_CHECKPOINT)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.farm_controller.close()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.base + path, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data is not None else {})
        with urlopen(request, timeout=3) as response:
            raw = response.read()
            content_type = response.headers.get_content_type()
            return response.status, json.loads(raw) if content_type == "application/json" else raw

    def test_state_has_five_agents_policy_and_inspectable_inputs(self):
        self.request("POST", "/api/reset", {"seed": 7})
        _, state = self.request("GET", "/api/state")
        self.assertEqual(len(state["animals"]), 5)
        self.assertTrue(state["paused"])
        self.assertEqual(state["clock"]["logical_dt"], 1)
        self.assertEqual(state["clock"]["absolute_tick"], state["tick"])
        self.assertEqual(state["synthetic_day_index"], state["tick"] // 24)
        self.assertEqual(state["tick_within_day"], state["tick"] % 24)
        self.assertEqual(state["clock"]["ticks_per_virtual_day"], 24)
        self.assertEqual(state["clock"]["time_unit"], "abstract_simulation_tick")
        self.assertTrue(state["policy_id"])
        animal_id = state["animals"][0]["id"]
        _, animal = self.request("GET", f"/api/animals/{animal_id}")
        self.assertEqual(animal["id"], animal_id)
        self.assertEqual(len(animal["features"]), 20)
        self.assertEqual(len(animal["probabilities"]), 6)
        self.assertIn("observation", animal)
        self.assertIn("retrieved_events", animal["memory_features"])
        self.assertIn("last_events", animal)
        self.assertEqual(animal["policy_id"], state["policy_id"])

    def test_reset_step_events_pause_and_speed_contract(self):
        _, reset = self.request("POST", "/api/reset", {"seed": 42})
        self.assertEqual(reset["seed"], 42)
        _, before = self.request("GET", "/api/state")
        self.assertEqual(before["tick"], 0)
        _, stepped = self.request("POST", "/api/step", {})
        self.assertEqual(stepped["tick"], 1)
        self.assertEqual(stepped["state"]["tick"], 1)
        self.assertEqual(stepped["state"]["clock"]["absolute_tick"], 1)
        self.assertEqual(stepped["state"]["clock"]["tick_within_day"], 1)
        _, events = self.request("GET", "/api/events?after=-1")
        self.assertGreaterEqual(len(events["events"]), 5)
        self.assertIn("retention", events)
        first_id = events["events"][0]["id"]
        _, tail = self.request("GET", f"/api/events?after={first_id}")
        self.assertTrue(all(event["id"] > first_id for event in tail["events"]))

        _, paused = self.request("POST", "/api/control", {"action": "pause"})
        tick = paused["tick"]
        time.sleep(0.12)
        _, still = self.request("GET", "/api/state")
        self.assertEqual(still["tick"], tick)
        _, speed = self.request("POST", "/api/control", {"action": "speed", "speed": 5})
        self.assertTrue(speed["paused"])
        self.assertEqual(speed["speed"], 5.0)
        _, resumed = self.request("POST", "/api/control", {"action": "resume"})
        self.assertFalse(resumed["paused"])
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            _, current = self.request("GET", "/api/state")
            if current["tick"] > tick:
                break
            time.sleep(0.02)
        self.assertGreater(current["tick"], tick)
        self.request("POST", "/api/control", {"action": "pause"})

    def test_clock_speed_changes_pacing_metadata_not_equal_tick_state(self):
        def run_at_speed(speed: float):
            self.request("POST", "/api/reset", {"seed": 314})
            _, paused = self.request("POST", "/api/control", {"action": "pause"})
            self.assertTrue(paused["paused"])
            _, configured = self.request("POST", "/api/control", {"action": "speed", "speed": speed})
            self.assertEqual(configured["speed"], speed)
            for _ in range(48):
                self.request("POST", "/api/step", {})
            _, state = self.request("GET", "/api/state")
            _, events = self.request("GET", "/api/events?after=-1")
            return state, events["events"]

        slower, slower_events = run_at_speed(0.5)
        faster, faster_events = run_at_speed(4.0)

        self.assertEqual(slower["speed"], 0.5)
        self.assertEqual(faster["speed"], 4.0)
        self.assertEqual(slower["clock"]["speed"], 0.5)
        self.assertEqual(faster["clock"]["speed"], 4.0)
        self.assertEqual(slower["tick"], faster["tick"])
        self.assertEqual(slower["clock"]["absolute_tick"], 48)
        self.assertEqual(faster["clock"]["absolute_tick"], 48)

        # Speed and paused status describe the controller, not logical world state.
        def without_pacing_metadata(state):
            logical = json.loads(json.dumps(state))
            logical.pop("speed")
            logical.pop("paused")
            logical["clock"].pop("speed")
            logical["clock"].pop("paused")
            return logical

        self.assertEqual(without_pacing_metadata(slower), without_pacing_metadata(faster))
        self.assertEqual(slower_events, faster_events)

    def test_event_api_reports_retention_and_reset_scoped_cursors(self):
        self.request("POST", "/api/reset", {"seed": 8})
        simulation = self.server.farm_controller.simulation
        simulation.append_events(tuple(SimEvent(i, "cow-1", "test", str(i))
                                       for i in range(EVENT_HISTORY_LIMIT + 2)))
        _, response = self.request("GET", "/api/events?after=0")
        self.assertEqual(len(response["events"]), EVENT_HISTORY_LIMIT)
        self.assertEqual(response["events"][0]["id"], 2)
        self.assertEqual(response["events"][-1]["id"], EVENT_HISTORY_LIMIT + 1)
        self.assertTrue(response["retention"]["has_gap_after_cursor"])
        self.assertEqual(response["retention"]["limit"], EVENT_HISTORY_LIMIT)

        _, reset = self.request("POST", "/api/reset", {"seed": 8})
        self.assertEqual(reset["tick"], 0)
        self.assertEqual(reset["synthetic_day_index"], 0)
        self.assertEqual(reset["tick_within_day"], 0)
        _, empty = self.request("GET", "/api/events?after=-1")
        self.assertEqual(empty["events"], [])
        self.assertEqual(empty["retention"]["oldest_id"], None)
        _, stepped = self.request("POST", "/api/step", {})
        _, first_batch = self.request("GET", "/api/events?after=-1")
        self.assertEqual(first_batch["events"][0]["id"], 0)

    def test_inspection_keeps_decision_inputs_aligned_with_probabilities(self):
        self.request("POST", "/api/reset", {"seed": 91})
        _, stepped = self.request("POST", "/api/step", {})
        animal_id = stepped["state"]["animals"][0]["id"]
        _, animal = self.request("GET", f"/api/animals/{animal_id}")
        self.assertEqual(animal["decision_tick"], 0)
        self.assertEqual(animal["decision_probabilities"], animal["probabilities"])
        self.assertEqual(len(animal["decision_features"]), 20)
        self.assertEqual(len(animal["current_features"]), 20)
        self.assertNotEqual(animal["decision_features"], animal["current_features"])

    def test_validation_errors_and_loopback_only_binding(self):
        with self.assertRaises(ValueError):
            create_server(host="0.0.0.0", port=0)
        with self.assertRaisesRegex(ValueError, "checkpoint missing"):
            create_server(port=0, checkpoint_path="/definitely/missing/policy.npz")
        for path, body in (("/api/control", {"action": "turbo"}),
                           ("/api/control", {"action": "speed", "speed": 100}),
                           ("/api/reset", {"seed": True})):
            request = Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                              headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=3)
            self.assertEqual(caught.exception.code, 400)
        with tempfile.TemporaryDirectory() as missing_static:
            alternate = create_server(port=0, checkpoint_path=DEFAULT_CHECKPOINT, static_dir=missing_static)
            try:
                thread = threading.Thread(target=alternate.serve_forever, daemon=True)
                thread.start()
                try:
                    urlopen(f"http://127.0.0.1:{alternate.server_port}/", timeout=3)
                    self.fail("missing browser UI should be reported as unavailable")
                except HTTPError as response:
                    self.assertEqual(response.code, 503)
                    self.assertIn(b"static", response.read())
                alternate.shutdown()
                thread.join(timeout=2)
            finally:
                alternate.farm_controller.close()
                alternate.server_close()

    def test_static_missing_asset_and_read_failure_are_distinct(self):
        with tempfile.TemporaryDirectory() as static_dir:
            Path(static_dir, "index.html").write_text("<main>Farm</main>")
            server = create_server(port=0, checkpoint_path=DEFAULT_CHECKPOINT, static_dir=static_dir)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with self.assertRaises(HTTPError) as missing:
                    urlopen(base + "/missing.js", timeout=3)
                self.assertEqual(missing.exception.code, 404)

                with patch("pathlib.Path.read_bytes", side_effect=OSError("private filesystem detail")):
                    with self.assertRaises(HTTPError) as unreadable:
                        urlopen(base + "/", timeout=3)
                self.assertEqual(unreadable.exception.code, 500)
                payload = unreadable.exception.read()
                self.assertIn(b"static file could not be read", payload)
                self.assertNotIn(b"private filesystem detail", payload)
            finally:
                server.shutdown()
                server.farm_controller.close()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
