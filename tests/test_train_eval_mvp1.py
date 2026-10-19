import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from farm_sim.model import Action, FEATURE_NAMES, SimEvent, ToyPhysiology
from farm_sim.policy import CategoricalPolicy
from farm_sim.simulation import make_world
from validation.train_eval_mvp1 import (
    BOOTSTRAPS, ROOT, bootstrap_ci, collect_encoder_states, comparison_margins,
    one_step_rewards, persist_training_summary, resource_event_tally, rollout,
    softmax_logits, task_cases, verify_locked_cohort, verify_prelock_attestation,
    write_prelock_attestation,
)


def _frozen_manifest_fixture():
    families = ("primary_starter_14x10", "wide_grid_20x14", "resource_scarce_relocated")
    return {
        "experiment_conditions": {"confirmatory_learning_conditions": [{"id": "encoder_a"}, {"id": "encoder_b"}]},
        "seed_protocol": {
            "locked_test_world_seeds_by_family": {
                family: list(range(20_001 + index * 100, 20_031 + index * 100))
                for index, family in enumerate(families)
            },
            "encoder_training_world_seeds": list(range(10_001, 10_011)),
            "training_checkpoint_seed_ids": [11_001, 11_002],
        },
    }


class TrainingEvaluationHarnessTests(unittest.TestCase):
    def test_four_case_training_inputs_keep_memory_only_pairwise_changes(self):
        cases = task_cases()
        self.assertEqual(len(cases), 4)
        for i, j in ((0, 1), (2, 3)):
            changed = np.flatnonzero(cases[i][0] != cases[j][0])
            self.assertEqual(len(changed), 1)
            self.assertIn(FEATURE_NAMES[int(changed[0])], ("mem_water_success", "mem_forage_success"))

    def test_encoder_training_states_are_seeded_actual_encoded_states_and_rewards(self):
        first = collect_encoder_states([12001])
        second = collect_encoder_states([12001])
        self.assertEqual(len(first), 72 * 5)
        for (x1, r1), (x2, r2) in zip(first, second):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(r1, r2)
            self.assertEqual(x1.shape, (len(FEATURE_NAMES),))
            self.assertEqual(r1.shape, (len(Action),))
            self.assertTrue(np.isfinite(x1).all() and np.isfinite(r1).all())

    def test_action_rewards_use_existing_toy_step_and_are_action_aligned(self):
        world = make_world(123)
        animal = world.animals[0]
        result = one_step_rewards(world, animal, np.zeros(len(FEATURE_NAMES)))
        self.assertEqual(result.shape, (len(Action),))
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(result[tuple(Action).index(Action.WALK)], result[tuple(Action).index(Action.REST)])

    def test_crossed_bootstrap_is_seeded_and_paired_cell_mean(self):
        cells = {1: {10: 1.0, 11: 3.0}, 2: {10: 3.0, 11: 5.0}}
        a = bootstrap_ci(cells, 250, 13)
        b = bootstrap_ci(cells, 250, 13)
        self.assertEqual(a, b)
        self.assertEqual(a["mean"], 3.0)
        self.assertEqual(a["training_seed_count"], 2)
        self.assertEqual(a["world_seed_count"], 2)

    def test_preregistered_locked_world_sets_have_30_per_family(self):
        manifest = _frozen_manifest_fixture()
        worlds = manifest["seed_protocol"]["locked_test_world_seeds_by_family"]
        self.assertEqual(set(worlds), {"primary_starter_14x10", "wide_grid_20x14", "resource_scarce_relocated"})
        self.assertTrue(all(len(seed_ids) == 30 for seed_ids in worlds.values()))
        training_worlds = set(manifest["seed_protocol"]["encoder_training_world_seeds"])
        self.assertTrue(all(training_worlds.isdisjoint(ids) for ids in worlds.values()))

    def test_locked_phase_never_reserializes_frozen_training_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training-summary.json"
            frozen = b'{"frozen": true}\n'
            path.write_bytes(frozen)
            self.assertFalse(persist_training_summary("locked", path, {"new": 1}, {}))
            self.assertEqual(path.read_bytes(), frozen)
            self.assertFalse(persist_training_summary("evaluate", path, {}, {}))
            self.assertEqual(path.read_bytes(), frozen)

    def test_locked_cohort_guard_checks_selected_and_initial_hashes_and_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "cohort"
            output.mkdir()
            manifest = _frozen_manifest_fixture()
            manifest_path = Path(directory) / "experiment-manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            patcher = patch("validation.train_eval_mvp1.MANIFEST", manifest_path)
            patcher.start()
            self.addCleanup(patcher.stop)
            conditions = [x["id"] for x in manifest["experiment_conditions"]["confirmatory_learning_conditions"]]
            seeds = manifest["seed_protocol"]["training_checkpoint_seed_ids"]
            allowlist = {}; hashes = {}; training = {}
            for condition in conditions:
                allowlist[condition] = {}; training[condition] = {"training_seeds": []}
                for seed in seeds:
                    prefix = f"{condition}-seed-{seed}"
                    paths = {"checkpoint_path": prefix+".trained.npz", "sidecar_path": prefix+".trained.json",
                             "initial_checkpoint_path": prefix+".initial.npz", "initial_sidecar_path": prefix+".initial.json"}
                    scores = {"500": 1.0, "1000": .5, "1500": .25, "2000": 0.0}
                    selected_meta = {"condition":condition,"training_condition":condition,"training_seed":seed,
                                     "selected_episode":500,"candidate_validation_means":scores}
                    initial_meta = {"condition":condition,"training_condition":condition,"training_seed":seed,
                                    "selected_episode":500,"role":"same-seed initialization comparator"}
                    for key in ("checkpoint_path","initial_checkpoint_path"):
                        path = output/paths[key]; path.write_bytes((condition+str(seed)+key).encode())
                        hashes[str(path.relative_to(ROOT))] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                    for key, metadata in (("sidecar_path",selected_meta),("initial_sidecar_path",initial_meta)):
                        path=output/paths[key];path.write_text(json.dumps({"metadata":metadata},sort_keys=True)+"\n")
                        hashes[str(path.relative_to(ROOT))] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                    item={"training_seed":seed,"selected_epoch":500}
                    for key,digest_key in (("checkpoint_path","checkpoint_sha256"),("sidecar_path","sidecar_sha256"),
                                           ("initial_checkpoint_path","initial_checkpoint_sha256"),("initial_sidecar_path","initial_sidecar_sha256")):
                        rel=str((output/paths[key]).relative_to(ROOT));item[key]=rel;item[digest_key]=hashes[rel]
                    allowlist[condition][str(seed)]=item
                    training[condition]["training_seeds"].append({"seed":seed,"selected_episode":500,"validation_mean_return_by_candidate":scores})
            protocol={"candidate_training_seed_ids":seeds,"selected_checkpoint_allowlist":allowlist}
            summary={"checkpoint_sha256":hashes,"training":training}
            self.assertTrue(verify_locked_cohort(output,protocol,summary)["summary_checkpoint_map"])

            selected=ROOT/allowlist[conditions[0]][str(seeds[0])]["checkpoint_path"]
            selected.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"artifact hash mismatch"):
                verify_locked_cohort(output,protocol,summary)

            # Even self-consistent hashes cannot hide a selected-epoch metadata mismatch.
            selected.write_bytes((conditions[0]+str(seeds[0])+"checkpoint_path").encode())
            item=allowlist[conditions[0]][str(seeds[0])]
            sidecar=ROOT/item["sidecar_path"]
            metadata=json.loads(sidecar.read_text())["metadata"];metadata["selected_episode"]=1000
            sidecar.write_text(json.dumps({"metadata":metadata},sort_keys=True)+"\n")
            digest=__import__("hashlib").sha256(sidecar.read_bytes()).hexdigest()
            rel=str(sidecar.relative_to(ROOT));hashes[rel]=digest;item["sidecar_sha256"]=digest
            with self.assertRaisesRegex(ValueError,"selected epoch mismatch"):
                verify_locked_cohort(output,protocol,summary)

            # Initial checkpoint and initial sidecar are independently bound too.
            selected_meta={"condition":conditions[0],"training_condition":conditions[0],"training_seed":seeds[0],
                           "selected_episode":500,"candidate_validation_means":{"500":1.0,"1000":.5,"1500":.25,"2000":0.0}}
            sidecar.write_text(json.dumps({"metadata":selected_meta},sort_keys=True)+"\n")
            digest=__import__("hashlib").sha256(sidecar.read_bytes()).hexdigest()
            hashes[rel]=digest;item["sidecar_sha256"]=digest
            initial=ROOT/item["initial_checkpoint_path"]
            initial_bytes=initial.read_bytes();initial.write_bytes(b"initial tamper")
            with self.assertRaisesRegex(ValueError,"artifact hash mismatch"):
                verify_locked_cohort(output,protocol,summary)
            initial.write_bytes(initial_bytes)
            initial_sidecar=ROOT/item["initial_sidecar_path"]
            initial_metadata=json.loads(initial_sidecar.read_text())["metadata"]
            initial_metadata["selected_episode"]=1000
            initial_sidecar.write_text(json.dumps({"metadata":initial_metadata},sort_keys=True)+"\n")
            digest=__import__("hashlib").sha256(initial_sidecar.read_bytes()).hexdigest()
            rel=str(initial_sidecar.relative_to(ROOT));hashes[rel]=digest;item["initial_sidecar_sha256"]=digest
            with self.assertRaisesRegex(ValueError,"initial sidecar selected epoch mismatch"):
                verify_locked_cohort(output,protocol,summary)

    def test_prelock_attestation_binds_protocol_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output=Path(directory); protocol=output/"frozen-analysis-protocol.json"
            protocol.write_text(json.dumps({"training_eval_source_sha256":"src","test_source_sha256":"test",
                "manifest_sha256":"manifest","training_summary_sha256":"summary","precision_pilot_sha256":"pilot"}))
            attestation=write_prelock_attestation(output,protocol)
            self.assertTrue(all(verify_prelock_attestation(protocol,output/"pre-lock-attestation.json").values()))
            protocol.write_text(protocol.read_text()+" ")
            self.assertFalse(verify_prelock_attestation(protocol,output/"pre-lock-attestation.json")["protocol_digest"])

    def test_near_zero_baseline_uses_frozen_pilot_absolute_margins(self):
        self.assertEqual(comparison_margins(2.0,.03),(.1,.2,"baseline_absolute_mean"))
        self.assertEqual(comparison_margins(.05,.03),(.03,.03,"frozen_pilot_absolute_fallback"))

    def test_resource_metrics_come_from_success_and_depletion_events(self):
        events=(SimEvent(1,"cow-1","forage_success","forage_success"),
                SimEvent(2,"cow-1","forage_depleted","forage_depleted"),
                SimEvent(2,"cow-2","water_success","water_success"),
                SimEvent(3,"cow-2","water_blocked","water_blocked"))
        self.assertEqual(dict(resource_event_tally(events)),{"forage_success":1,"forage_depleted":1,"water_success":1,"water_blocked":1})
        episode=rollout(__import__("farm_sim.policy",fromlist=["HeuristicBaseline"]).HeuristicBaseline(),"primary_starter_14x10",902)
        self.assertEqual(len(episode["resource_state_trajectory_by_tick"]),episode["ticks"]+1)
        self.assertEqual(set(episode["resource_interactions"]["event_counts"]),{"forage_success","forage_depleted","water_success","water_blocked"})
        self.assertEqual(episode["resource_interactions"]["final_resource_state"],episode["resource_state_trajectory_by_tick"][-1])


if __name__ == "__main__":
    unittest.main()
