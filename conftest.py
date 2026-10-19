"""Run the standard-test training guard before pytest collects or executes tests."""


def pytest_sessionstart(session):
    del session
    from scripts.check_standard_test_safety import main

    if main() != 0:
        raise RuntimeError("standard test safety gate failed; no tests were executed")
