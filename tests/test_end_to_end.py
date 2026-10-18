"""End-to-end HTTP smoke tests using the shipped trained local checkpoint."""
from __future__ import annotations

import json
import threading
import unittest
from urllib.request import Request, urlopen

from farm_sim.server import DEFAULT_CHECKPOINT, create_server


class FarmEndToEndTests(unittest.TestCase):
    """Exercise the same server/checkpoint path used by a local browser run."""

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
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        with urlopen(request, timeout=3) as response:
            raw = response.read()
            content_type = response.headers.get_content_type()
            return response.status, content_type, json.loads(raw) if content_type == "application/json" else raw

    def test_browser_entry_and_assets_are_served(self):
        status, content_type, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/html")
        self.assertIn(b"pasture map", page.lower())
        self.assertIn(b"app.js", page)
        self.assertIn(b"style.css", page)
        self.assertIn(b'id="decision-context"', page)
        self.assertIn(b'id="event-gap-notice"', page)
        self.assertIn(b'SYNTHETIC TIME', page)
        self.assertIn(b'24 abstract simulation ticks per virtual day', page)

        for asset, expected_type, marker in (
            ("/style.css", "text/css", b"canvas"),
            ("/app.js", "text/javascript", b"/api/state"),
        ):
            with self.subTest(asset=asset):
                status, content_type, payload = self.request("GET", asset)
                self.assertEqual(status, 200)
                self.assertEqual(content_type, expected_type)
                self.assertIn(marker, payload)
                if asset == "/app.js":
                    self.assertIn(b"decision_features", payload)
                    self.assertIn(b"decision_probabilities", payload)
                    self.assertIn(b"has_gap_after_cursor", payload)

    def test_learned_checkpoint_inspection_and_clock_event_workflow(self):
        _, _, state = self.request("GET", "/api/state")
        self.assertEqual(len(state["animals"]), 5)
        self.assertEqual(state["policy_id"], "reinforce-seed-11")
        self.assertTrue(state["paused"])
        self.assertEqual((state["clock"]["synthetic_day_index"],
                          state["clock"]["tick_within_day"]), (0, 0))

        animal_id = state["animals"][0]["id"]
        _, _, animal = self.request("GET", f"/api/animals/{animal_id}")
        self.assertEqual(animal["id"], animal_id)
        self.assertEqual(animal["policy_id"], state["policy_id"])
        self.assertEqual(len(animal["probabilities"]), 6)
        self.assertAlmostEqual(sum(animal["probabilities"]), 1.0)
        self.assertEqual(len(animal["features"]), len(animal["feature_names"]))
        self.assertIn("visible_forage", animal["observation"])
        self.assertIn("visible_water", animal["observation"])
        self.assertIn("retrieved_events", animal["memory_features"])

        _, _, resumed = self.request("POST", "/api/control", {"action": "resume"})
        self.assertFalse(resumed["paused"])
        _, _, paused = self.request("POST", "/api/control", {"action": "pause"})
        self.assertTrue(paused["paused"])

        _, _, stepped = self.request("POST", "/api/step", {})
        self.assertEqual(stepped["tick"], 1)
        self.assertEqual(stepped["state"]["policy_id"], "reinforce-seed-11")
        _, _, events = self.request("GET", "/api/events?after=-1")
        self.assertTrue(events["events"])
        self.assertEqual([event["id"] for event in events["events"]],
                         sorted(event["id"] for event in events["events"]))

        _, _, reset = self.request("POST", "/api/reset", {"seed": 73})
        self.assertEqual(reset["seed"], 73)
        self.assertEqual(reset["tick"], 0)
        _, _, cleared = self.request("GET", "/api/events?after=-1")
        self.assertEqual(cleared["events"], [])

    def test_same_seed_replays_identically_with_live_trained_policy(self):
        def replay(seed: int):
            _, _, reset = self.request("POST", "/api/reset", {"seed": seed})
            self.assertEqual(reset["policy_id"], "reinforce-seed-11")
            snapshots = []
            for expected_tick in range(1, 5):
                _, _, step = self.request("POST", "/api/step", {})
                self.assertEqual(step["tick"], expected_tick)
                snapshots.append(step["state"])
            _, _, events = self.request("GET", "/api/events?after=-1")
            return snapshots, events["events"]

        first = replay(29)
        second = replay(29)
        self.assertEqual(first, second)
        actions = [animal["action"] for animal in first[0][0]["animals"]]
        self.assertEqual(len(actions), 5)
        self.assertTrue(all(action for action in actions))


if __name__ == "__main__":
    unittest.main()
