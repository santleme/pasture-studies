"""Device microbenchmark for the exact small actor; not a simulator speedup claim."""
from __future__ import annotations
import argparse, json, platform, statistics, time
from pathlib import Path
import numpy as np
from farm_sim.execution_policy import require_kaggle_for_full_workload
from farm_sim.torch_training import ActorCritic, require_torch, resolve_device, set_deterministic


def sync(torch, device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark(device: str, *, batches: list[int], warmups: int, iterations: int, repeats: int):
    lightweight = len(batches) <= 2 and max(batches, default=0) <= 5 and warmups <= 1 and iterations <= 2 and repeats <= 1
    require_kaggle_for_full_workload("Torch profiling benchmark", smoke=lightweight)
    torch = require_torch()
    result = []
    for batch in batches:
        device_obj = torch.device(device)
        model = ActorCritic(seed=92017, dtype=torch.float32).to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(31007 + batch)
        x_cpu = torch.tensor(rng.normal(size=(batch, 20)).astype(np.float32))
        y_cpu = torch.tensor(rng.integers(0, 6, size=batch), dtype=torch.long)
        # Forward-only inference, with inputs resident on the target device.
        x_infer = x_cpu.to(device_obj)
        model.eval()
        def infer():
            with torch.no_grad(): model(x_infer)
        for _ in range(warmups): infer()
        sync(torch, device)
        inference_samples = []
        for _ in range(repeats):
            sync(torch, device); start = time.perf_counter()
            for _ in range(iterations): infer()
            sync(torch, device)
            inference_samples.append((time.perf_counter() - start) / iterations)
        result.append({"device": device, "batch_size": batch, "phase": "forward-resident",
                       "warmups": warmups, "iterations_per_repeat": iterations,
                       "seconds_per_update_repeats": inference_samples,
                       "median_seconds": statistics.median(inference_samples),
                       "p95_seconds": float(np.percentile(inference_samples, 95)),
                       "updates_per_second": 1 / statistics.median(inference_samples)})
        model.train()
        # Compute-only uses resident device tensors; transfer-inclusive begins with CPU tensors.
        for transfer in (False, True):
            x = x_cpu.to(device_obj)
            y = y_cpu.to(device_obj)
            def step():
                optimizer.zero_grad(set_to_none=True)
                batch_x = x_cpu.to(device_obj) if transfer else x
                batch_y = y_cpu.to(device_obj) if transfer else y
                logits, _ = model(batch_x)
                loss = torch.nn.functional.cross_entropy(logits, batch_y)
                loss.backward(); optimizer.step()
            for _ in range(warmups): step()
            sync(torch, device)
            samples = []
            for _ in range(repeats):
                sync(torch, device); start = time.perf_counter()
                for _ in range(iterations): step()
                sync(torch, device)
                samples.append((time.perf_counter() - start) / iterations)
            result.append({"device": device, "batch_size": batch,
                           "phase": "update-transfer-inclusive" if transfer else "update-resident",
                           "warmups": warmups, "iterations_per_repeat": iterations,
                           "seconds_per_update_repeats": samples,
                           "median_seconds": statistics.median(samples),
                           "p95_seconds": float(np.percentile(samples, 95)),
                           "updates_per_second": 1 / statistics.median(samples)})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    p.add_argument("--devices", help="comma-separated explicit devices, e.g. cpu,cuda:0")
    p.add_argument("--batches", default="1,5,128,1024,4096")
    p.add_argument("--warmups", type=int, default=50)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("torch-profile.json"))
    p.add_argument("--smoke", action="store_true", help="explicit bounded local smoke profile")
    args=p.parse_args(argv)
    try:
        require_kaggle_for_full_workload("Torch profiling benchmark", smoke=args.smoke)
    except RuntimeError as exc:
        p.error(str(exc))
    if args.smoke:
        if args.devices or args.batches != "1,5,128,1024,4096" or (args.warmups, args.iterations, args.repeats) != (50, 200, 5):
            p.error("--smoke uses fixed limits; omit --devices, --batches, --warmups, --iterations, and --repeats")
        args.devices, args.batches = None, "1,5"
        args.warmups, args.iterations, args.repeats = 1, 2, 1
    torch=require_torch(); set_deterministic(1234)
    devices=args.devices.split(",") if args.devices else [resolve_device(args.device)]
    rows=[]
    for dev in devices:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            rows.append({"device":dev,"status":"unavailable"}); continue
        rows.extend(benchmark(dev,batches=[int(x) for x in args.batches.split(",")],
                              warmups=args.warmups,iterations=args.iterations,repeats=args.repeats))
    doc={"suite":"torch-mlp-update-microbenchmark-v1","python":platform.python_version(),
         "torch":torch.__version__,"cuda_available":torch.cuda.is_available(),
         "gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
         "results":rows,
         "claims_boundary":"Network update measurements only; not end-to-end acceleration or learning evidence."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"result_count":len(rows),"cuda_available":torch.cuda.is_available()}))
if __name__=="__main__": main()
