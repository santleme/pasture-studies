"""Integrity checks for the non-authorizing analysis release candidate."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from validation.preregister_torch_mvp1 import canonical_sha

ROOT = Path(__file__).resolve().parents[1]


class TorchReleaseCandidateTests(unittest.TestCase):
    def test_candidate_hashes_current_sources_and_preserves_approval_denials(self):
        path = ROOT / "validation/protocols/pasture-study-analysis-v2-candidate.json"
        doc = json.loads(path.read_text())
        claimed = doc["candidate_sha256"]
        body = {k: v for k, v in doc.items() if k != "candidate_sha256"}
        self.assertEqual(claimed, canonical_sha(body))
        for relative, digest in doc["provenance"]["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)
        self.assertEqual(doc["provenance"]["source_set_sha256"], canonical_sha(doc["provenance"]["source_sha256"]))
        self.assertFalse(doc["approval"]["pilot_authorized"])
        self.assertFalse(doc["approval"]["training_outcomes_authorized"])
        self.assertFalse(doc["approval"]["locked_test_access_authorized"])
        self.assertFalse(doc["release_eligible"])
        self.assertFalse(doc["historical_protocol"]["thresholds_changed"])
        self.assertFalse(doc["historical_protocol"]["rewards_changed"])


if __name__ == "__main__":
    unittest.main()
