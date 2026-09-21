"""Run one explicit benchmark cell, or print the matrix without executing anything.

Examples (from repository root):
  python -m experiments.modal_benchmark matrix recording.wav
  python -m experiments.modal_benchmark run recording.wav --backend local --execute
  python -m experiments.modal_benchmark run recording.wav --backend modal --workers 2 --execute
  python -m experiments.modal_benchmark summarize benchmark-results
"""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import re
import shutil
import statistics
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from transcription.audio import decode_pcm, effective_overlap
from transcription.contracts import SAMPLE_RATE
from transcription.storage import job_lock, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def word_errors(reference, hypothesis):
    """Case/punctuation-normalized Levenshtein counts; O(reference * hypothesis) time."""
    reference = re.findall(r"\w+", reference.casefold())
    hypothesis = re.findall(r"\w+", hypothesis.casefold())
    # (distance, substitutions, deletions, insertions), deterministic tie breaking.
    row = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, word in enumerate(reference, 1):
        next_row = [(i, 0, i, 0)]
        for j, other in enumerate(hypothesis, 1):
            if word == other:
                next_row.append(row[j - 1])
            else:
                d, s, de, ins = row[j - 1]
                sub = (d + 1, s + 1, de, ins)
                d, s, de, ins = row[j]
                deletion = (d + 1, s, de + 1, ins)
                d, s, de, ins = next_row[j - 1]
                insertion = (d + 1, s, de, ins + 1)
                next_row.append(min((sub, deletion, insertion), key=lambda item: item[0]))
        row = next_row
    errors, substitutions, deletions, insertions = row[-1]
    return {"reference_words": len(reference), "substitutions": substitutions,
            "deletions": deletions, "insertions": insertions,
            "wer": errors / len(reference) if reference else (0.0 if not hypothesis else None)}


def reserve_budget(root, run_id, budget, reservation):
    """Reservations persist even on failure; only actual billing reconciliation releases them."""
    if not math.isfinite(budget) or not math.isfinite(reservation) or budget <= 0 or reservation <= 0:
        raise ValueError("Budget and reservation must be positive.")
    ledger_path = root / "budget.json"
    with job_lock(root):
        ledger = read_json(ledger_path) if ledger_path.exists() else {"runs": {}}
        allocated = sum(v.get("actual_gross_usd", v["reserved_usd"]) for v in ledger["runs"].values())
        if allocated + reservation > budget:
            raise ValueError(f"Budget gate: ${allocated:.3f} allocated + ${reservation:.3f} reserved > ${budget:.2f}.")
        ledger["runs"][run_id] = {"reserved_usd": reservation, "reserved_at": time.time()}
        write_json(ledger_path, ledger)


def reconcile(root, run_id, gross_cost):
    if not math.isfinite(gross_cost) or gross_cost < 0:
        raise ValueError("Gross cost must be nonnegative.")
    with job_lock(root):
        ledger = read_json(root / "budget.json")
        ledger["runs"][run_id]["actual_gross_usd"] = gross_cost
        write_json(root / "budget.json", ledger)


def peak_concurrency(results):
    events = []
    for result in results:
        execution = result.get("execution", {})
        if "started_at" in execution and "finished_at" in execution:
            events.extend([(execution["started_at"], 1), (execution["finished_at"], -1)])
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def environment():
    versions = {}
    for name in ("openai-whisper", "torch", "numpy", "modal", "tqdm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions}


def matrix(source):
    return [{"source": str(source), "backend": backend, "workers": workers,
             "state": state, "repetition": repetition}
            for backend, workers in (("baseline", 1), ("local", 1), ("modal", 1), ("modal", 2), ("modal", 4))
            for state in ("cold", "warm") for repetition in range(1, 4)]


def run_cell(args):
    specification = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if not args.execute:
        print(json.dumps({"execute": False, "cell": specification}, indent=2))
        return
    source = args.source.resolve()
    if not source.is_file():
        raise ValueError("Recording does not exist.")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    if args.backend == "modal":
        reserve_budget(root, run_id, args.budget_usd, args.reserve_usd)
    run_dir = root / run_id
    run_dir.mkdir()
    write_json(run_dir / "specification.json", specification)
    report = {"run_id": run_id, "specification": specification, "environment": environment(),
              "source": str(source), "state": args.state, "backend": args.backend,
              "workers": args.workers if args.backend == "modal" else 1,
              "gross_cost_usd": None if args.backend == "modal" else 0.0,
              "measurement_status": "running"}
    warm_calls, adapter = [], None
    try:
        if args.backend == "modal":
            from transcription.backends.modal import ModalBackend
            from transcription.pipeline import transcribe_remote
            adapter = ModalBackend(model=args.model, workers=args.workers, gpu=args.gpu,
                                   transfer=args.transfer, pool=run_id)
            if args.state == "warm":
                # Reserve a warm pool during preparation/upload; release in finally even on interruption.
                adapter.worker.update_autoscaler(min_containers=args.workers)
                warm_started = time.perf_counter()
                for _ in range(args.workers):
                    warm_calls.append(adapter.worker.warmup.spawn(30))
                warmed = [call.get(timeout=240) for call in warm_calls]
                report["warmup_seconds"] = time.perf_counter() - warm_started
                report["warmed_worker_ids"] = sorted({w["worker_id"] for w in warmed})
                if len(report["warmed_worker_ids"]) != args.workers:
                    raise ValueError("Warmup did not reach every requested worker; cell is invalid, not warm.")
            metrics = transcribe_remote(source, model=args.model, language=args.language,
                        chunk_duration=args.chunk_duration, overlap=args.overlap, workers=args.workers,
                        directory=run_dir / "job", transport_factory=lambda **kw: adapter,
                        output_path=run_dir / "transcript.txt")
            report.update(metrics)
            executions = [r["execution"] for r in metrics["results"]]
            worker_ids = {e["worker_id"] for e in executions}
            report["observed_peak_inference_concurrency"] = peak_concurrency(metrics["results"])
            report["worker_ids"] = sorted(worker_ids)
            report["aggregate_execution_seconds"] = sum(e["execution_seconds"] for e in executions)
            report["observed_model_loads"] = len(worker_ids)
            if args.state == "warm" and not worker_ids.issubset(set(report["warmed_worker_ids"])):
                raise ValueError("A cold worker entered the warm run; classify this cell as mixed and rerun.")
            if args.state == "cold" and not all(any(e["worker_id"] == wid and e["container_call_index"] == 0
                                                    for e in executions) for wid in worker_ids):
                raise ValueError("Cold pool unexpectedly reused a worker; cell is invalid.")
            # Client/server clocks can differ; do not mislabel this as pure queue latency.
            for key, call in metrics["calls"].items():
                call["submission_round_trip_seconds"] = call["acknowledged_at"] - call["submitted_at"]
                call["remote_wait_and_retrieval_seconds"] = call["received_at"] - call["acknowledged_at"]
            if args.transfer == "direct":
                report["upload"]["bytes"] = int(metrics["processed_audio_seconds"] * SAMPLE_RATE * 2)
        else:
            # Staging protects the original transcript from the frozen baseline's output behavior.
            staged = run_dir / source.name
            shutil.copy2(source, staged)
            import torch
            import whisper
            if args.backend == "baseline":
                path = Path(__file__).with_name("baseline_transcribe.py")
                spec = importlib.util.spec_from_file_location("frozen_transcriber", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                argv = [str(path), str(staged), "--model", args.model, "--chunk-duration",
                        str(args.chunk_duration), "--overlap", str(args.overlap)]
                if args.language:
                    argv += ["--language", args.language]
                original_loader = whisper.load_model
                warm_model = original_loader(args.model, device="cuda" if torch.cuda.is_available() else "cpu") if args.state == "warm" else None
                started = time.perf_counter()
                with patch.object(sys, "argv", argv), patch.object(module.whisper, "load_model",
                        side_effect=lambda *a, **kw: warm_model if warm_model is not None else original_loader(*a, **kw)):
                    module.main()
                report["end_to_end_seconds"] = time.perf_counter() - started
                staged.with_suffix(".txt").replace(run_dir / "transcript.txt")
            else:
                from transcription.backends.local import LocalBackend
                load_started = time.perf_counter()
                backend = LocalBackend(args.model)
                warmup = time.perf_counter() - load_started
                started = load_started if args.state == "cold" else time.perf_counter()
                metrics = backend.transcribe_file(staged, args.language, args.chunk_duration, args.overlap,
                                                 output_path=run_dir / "transcript.txt")
                report.update(metrics)
                report["end_to_end_seconds"] = time.perf_counter() - started
                report["warmup_seconds"] = warmup if args.state == "warm" else 0
            duration = decode_pcm(staged, run_dir / "duration.pcm")["duration"]
            report["original_audio_seconds"] = duration
            report["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            report["fp16"] = torch.cuda.is_available()
            (run_dir / "duration.pcm").unlink()
            staged.unlink()
        if args.reference:
            report["quality"] = word_errors(args.reference.read_text(encoding="utf-8"),
                                            (run_dir / "transcript.txt").read_text(encoding="utf-8"))
        report["measurement_status"] = "complete"
    except BaseException as error:
        report["measurement_status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if adapter is not None and args.state == "warm":
            for call in warm_calls:
                try:
                    call.cancel()
                except Exception:
                    pass
            try:
                adapter.worker.update_autoscaler(min_containers=0)
            except Exception as error:
                report["cleanup_error"] = str(error)
                print("WARNING: warm pool reset failed; stop the benchmark app in Modal now.", file=sys.stderr)
        write_json(run_dir / "report.json", report)
        print(f"Benchmark report: {run_dir / 'report.json'}")


def summarize(root):
    ledger = read_json(root / "budget.json") if (root / "budget.json").exists() else {"runs": {}}
    groups = {}
    for path in root.glob("*/report.json"):
        report = read_json(path)
        if report["measurement_status"] != "complete":
            continue
        cost = ledger["runs"].get(report["run_id"], {}).get("actual_gross_usd", report.get("gross_cost_usd"))
        report["cost_per_audio_hour"] = cost * 3600 / report["original_audio_seconds"] if cost is not None else None
        spec = report["specification"]
        key = (report["source"], report["backend"], report["workers"], report["state"],
               spec["model"], spec["language"], spec["chunk_duration"], spec["overlap"],
               spec["gpu"] if report["backend"] == "modal" else "local", spec["transfer"],
               json.dumps(report["environment"], sort_keys=True))
        groups.setdefault(key, []).append(report)
    rows = []
    for key, reports in groups.items():
        times = [r["end_to_end_seconds"] for r in reports]
        costs = [r["cost_per_audio_hour"] for r in reports]
        rows.append({"configuration": key, "runs": len(times), "median_seconds": statistics.median(times),
                     "range_seconds": [min(times), max(times)],
                     "median_cost_per_audio_hour": statistics.median(costs) if all(c is not None for c in costs) else None,
                     "quality": [r.get("quality") for r in reports],
                     "acceptance": "pending paired quality/boundary review and complete billing"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("matrix")
    plan.add_argument("source", type=Path)
    run = commands.add_parser("run")
    run.add_argument("source", type=Path)
    run.add_argument("--backend", choices=("baseline", "local", "modal"), default="local")
    run.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    run.add_argument("--gpu", choices=("L4", "T4"), default="L4")
    run.add_argument("--transfer", choices=("volume", "direct"), default="volume")
    run.add_argument("--state", choices=("cold", "warm"), default="cold")
    run.add_argument("--model", default="small")
    run.add_argument("--language")
    run.add_argument("--chunk-duration", type=float, default=300.0)
    run.add_argument("--overlap", type=float, default=30.0)
    run.add_argument("--output", type=Path, default=Path("benchmark-results"))
    run.add_argument("--reference", type=Path)
    run.add_argument("--budget-usd", type=float, default=10.0)
    run.add_argument("--reserve-usd", type=float, default=1.0,
                     help="Conservative estimated cell cost; retained until reconciled with actual billing.")
    run.add_argument("--execute", action="store_true", help="Actually run inference; Modal may incur charges.")
    summary = commands.add_parser("summarize")
    summary.add_argument("root", type=Path)
    cost = commands.add_parser("reconcile-cost")
    cost.add_argument("root", type=Path)
    cost.add_argument("run_id")
    cost.add_argument("gross_cost", type=float)
    args = parser.parse_args()
    if args.command == "matrix":
        print(json.dumps(matrix(args.source), indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize(args.root), indent=2))
    elif args.command == "reconcile-cost":
        reconcile(args.root, args.run_id, args.gross_cost)
    else:
        effective_overlap(args.chunk_duration, args.overlap)
        run_cell(args)


if __name__ == "__main__":
    main()
