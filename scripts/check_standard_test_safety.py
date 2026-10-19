"""Reject local training calls in tests collected by the standard suites."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests"
OPT_IN_DIRECTORY = TEST_ROOT / "opt_in_training"
TRAINING_CALLS = {
    "learn", "fit", "train_algorithm", "train_and_evaluate", "train_generalized_dagger",
    "train_ppo_finetune", "train_ppo", "generate_bundle", "build_corpus", "replay_once",
    "collect_teacher_data", "run_readiness_ddp", "train_policy", "train_suite",
}


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _approved_kaggle_boundary_test(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Allow the one test whose only trainer calls assert the Kaggle guard fires first."""
    if node.name != "test_direct_dagger_and_ppo_training_helpers_are_kaggle_only":
        return False
    source = ast.unparse(node)
    return all(fragment in source for fragment in (
        "farm_sim.execution_policy.is_kaggle_runtime", "return_value=False",
        "validation.dagger_generalization.collect_domain_data",
        "train_generalized_dagger(", "train_ppo_finetune(",
        "Kaggle-only", "collect.assert_not_called()",
    ))


def inspect_standard_tests(test_root: Path = TEST_ROOT) -> list[str]:
    failures: list[str] = []
    opt_in_directory = test_root / "opt_in_training"
    for path in sorted(test_root.rglob("test_*.py")):
        if opt_in_directory in path.parents:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: cannot parse standard test: {exc}")
            continue
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = _call_name(call.func)
            enclosing_test = call
            while enclosing_test in parents and not (
                    isinstance(enclosing_test, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and enclosing_test.name.startswith("test_")):
                enclosing_test = parents[enclosing_test]
            approved = (isinstance(enclosing_test, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and _approved_kaggle_boundary_test(enclosing_test))
            try:
                relative = path.relative_to(ROOT)
            except ValueError:
                relative = path
            if name in TRAINING_CALLS:
                if not (approved and name in {"train_generalized_dagger", "train_ppo_finetune"}):
                    failures.append(
                        f"{relative}:{call.lineno}: test calls training/data-generation API {name}()"
                    )
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "step"
                    and isinstance(call.func.value, ast.Name)
                    and ("optimizer" in call.func.value.id.lower()
                         or call.func.value.id.lower().endswith("_opt"))):
                failures.append(f"{relative}:{call.lineno}: test performs optimizer.step()")
    return failures


def main() -> int:
    failures = inspect_standard_tests()
    if failures:
        print("Standard test safety gate FAILED before test execution:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("Standard test safety gate passed: no training/data-generation calls in default tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
