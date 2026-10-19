# PettingZoo Parallel pilot

## Scope and interpretation

This optional adapter exposes the five existing animals as PettingZoo Parallel
agents. Each `step(actions)` requires one discrete action from every live animal
and collects all decisions from that tick's shared pre-transition observations
before advancing the simulation. It then delegates to the existing
`Simulation.step_with_decisions` path. That path commits movement, resource
consumption, events, and physiology sequentially in stable animal-ID order.

Therefore this is a **joint-action collection interface with ordered commit**,
not simultaneous physics or order-independent resource arbitration. The first
stable-ID animal can consume a shared resource before a later animal's turn.
Consumers must treat priority as part of the environment dynamics. The adapter
does not modify production simulation/reward code and performs no training.

Observations are the existing 20-element per-animal feature vector; the action
space is the six-value `Action` enum. Individual rewards are the existing
per-animal reward. Episodes truncate after `max_cycles` (72 by default).

## Verification

Run the API smoke after installing the optional dependencies:

```sh
python -m pip install -r requirements-pettingzoo-pilot.txt
python -m validation.pettingzoo_parallel_pilot
python -m pytest -q tests/test_pettingzoo_parallel_pilot.py
```

The tests exercise fixed-seed reset reproducibility, PettingZoo's
`parallel_api_test`, complete action-batch validation, shared-snapshot action
collection, and the simulation's stable-ID consumable-resource arbitration.
No PPO training, Kaggle execution, or external service is part of this pilot.

## Results (2026-10-19)

- `PYTHONPATH=/tmp/pettingzoo-pilot-deps:$PYTHONPATH python3 -m pytest -q tests/test_pettingzoo_parallel_pilot.py` — **PASS**, 3 passed.
- `PYTHONPATH=/tmp/pettingzoo-pilot-deps python3 -m validation.pettingzoo_parallel_pilot` — **PASS**, PettingZoo Parallel API smoke passed with PettingZoo 1.25.0 and Gymnasium 1.2.3.
- `python3 -m compileall -q validation/pettingzoo_parallel_pilot.py tests/test_pettingzoo_parallel_pilot.py` — **PASS**.
- Linter/type checker/full repository suite — **not run**; no project linter/type-check configuration was found, and this task's verification was scoped to the adapter API/seed tests.
