"""Guard unittest discovery before any standard test module is imported."""

from scripts.check_standard_test_safety import main as _standard_test_safety_gate

if _standard_test_safety_gate() != 0:
    raise RuntimeError("standard test safety gate failed; no unittest tests were imported")
