# Modal backend: operation, benchmark, and rollout

## Current status

The optional backend and an offline-tested benchmark harness are implemented.
Cloud deployment, GPU inference, model-cache behavior on Modal, and the latency,
quality, and cost gates still require an authorized deployed test with representative
recordings. Local remains the default. There is no claim that the $0.10/audio-hour
target has been achieved.

The actual pre-change implementation is frozen in `experiments/baseline_transcribe.py`.
Do not refactor that snapshot: it is the baseline for the comparison. Run baseline
cells in the original environment, and run the refactored local and Modal cells from
a separate Python 3.12 environment using `requirements-benchmark.txt`. The GPU image
pins Whisper 20250625, PyTorch 2.10.0/cu128, NumPy 2.2.6, and tqdm 4.67.1. The optional
Modal client is pinned to 1.5.5. Record full resolved packages, image identity, GPU,
FFmpeg version, and local hardware alongside each benchmark campaign. The harness
records client Python/package versions; deployment/image and platform billing details
must be exported from Modal for the campaign.

## Execution and data contracts

The CLI owns input discovery. `transcription/audio.py` streams the same FFmpeg s16le,
mono, 16 kHz conversion used by Whisper to disk. Neither MP4 video nor the original
compressed audio is uploaded. Requests contain integer sample boundaries, derived
second offsets, recording/job IDs, chunk index/count, model/checkpoint identity,
language, configuration fingerprint, and per-chunk PCM checksum. The recording ID
is SHA-256 of normalized PCM. Source bytes are separately hashed to reject a changed
source on resume. Jobs receive random IDs, so two independent runs do not share outputs.

The client uploads a single PCM recording with a seven-day lease. A CPU function
checks the complete hash and model-cache availability before GPU calls. The worker
reloads its audio Volume before reading each range and closes files before later
reloads. Weights and audio use separate Volumes. Workers do not write model weights.
The serial CPU provisioner publishes verified checkpoints atomically.

Results repeat request identity and include language, text, chunk-relative segment
timestamps, and execution metadata. `global_segments` adds the chunk offset to a
copy, never to the saved result. Each result is validated and atomically saved before
progress advances. Final assembly sorts by chunk index, requires complete coverage,
and accepts only one result for each identity. An identical duplicate is harmless;
conflicting duplicates are rejected. Execution may occur more than once after platform
recovery; this is not an exactly-once inference guarantee.

The initial merge is the original segment-end overlap rule. It can repeat a word or
phrase where a segment straddles the overlap. Nonempty text without segments now fails
explicitly in multi-chunk output instead of being silently discarded. Silent chunks
may have empty text and segments. Language detection remains independent per chunk
unless `--language` is supplied. No cross-chunk prompting is introduced.

## Limits and failure recovery

- One active inference per container; two outstanding client calls by default, maximum
  four. SDK variants cap the selected worker pool at that count. Model parameters and
  benchmark variants have independent pools, so this is not a workspace-wide cap.
- L4 by default, two physical CPU cores (request and limit), 8 GiB host RAM for
  tiny/base/small, 16 GiB for medium/large. No batching, quantization, engine switching,
  minimum warm pool, or spare buffer in normal use; idle window is two seconds.
- Startup timeout 180 seconds, call timeout 600 seconds, scheduling/execution job
  deadline 3,600 seconds after upload. Unchunked recordings retain these limits.
- Worker audio I/O retries twice after 1 and 2 seconds. Inference/validation/OOM errors
  are terminal. SDK function retries are disabled to avoid multiplying application
  retries. Modal may still reschedule crashes/preemptions; the client deadline and
  cancellation bound an attached job, not an abandoned workspace's total bill.
- Ctrl+C and exceptions cancel pending known calls. If cancellation is unconfirmed,
  the ID remains pending for reconciliation. A hard-killed process cannot cancel work.
  Use the Modal dashboard to inspect/stop abandoned calls or a crash-looping app.
- A local OS lock prevents concurrent access to the same job directory on one machine.
  Do not resume the same job concurrently from multiple machines or synced copies.
- A terminal worker failure stops that invocation. After resolving the cause, an
  explicit resume retries unfinished/failed chunks while retaining completed chunks.
  There is no automatic whole-job retry loop; changed inference settings require a new job.

A network failure during submission can leave an unknown call ID. Such a job remains
in `submitting` state and refuses automatic resubmission. Check the Modal dashboard
and cancel the orphan before changing that entry in `state.json` to `cancelled` and
resuming. This deliberate manual recovery prevents unnoticed duplicate GPU charges.
Do not remove or rewrite pending call IDs to force retries.

Atomic checkpoint writes leave `.tmp` fragments harmless after crashes. A corrupt
committed checkpoint fails closed. The original transcript is left intact on job
failure. Local results outlive the platform's seven-day result retention; unresolved
expired calls are resubmitted. If uploaded PCM has expired, it is uploaded again from
the validated local copy, or regenerated from the unchanged source.

## Benchmark procedure

Use approved 5-, 30-, 60-, and 120-minute recordings, covering clean speech, noisy
multi-speaker meetings, accents, silence, and MP4 audio. Mostly single-language audio
is the primary workload; add language switching as a quality case. Use identical
model weights, decoding defaults, chunk duration, and overlap across configurations.
Local CPU FP32 versus remote GPU FP16 is an explicit hardware/precision difference.
One GPU versus multiple identical GPUs isolates the effect of parallel execution.

Print the matrix or a cell specification without downloads, authentication, or inference:

```bash
python -m experiments.modal_benchmark matrix recording.mp4
python -m experiments.modal_benchmark run recording.mp4 --backend modal --workers 2
```

After deployment and paid execution are authorized, run a short one-worker T4/L4 pilot.
Pick the GPU with lower measured gross cost per audio hour; within 5%, prefer lower
latency. Then run three independent cold and three warm cells for baseline, refactored
local, one GPU, two GPUs, and four GPUs. Run one cell per process:

```bash
python -m experiments.modal_benchmark run recording.mp4 --backend baseline --state cold --execute
python -m experiments.modal_benchmark run recording.mp4 --backend local --state cold --execute
python -m experiments.modal_benchmark run recording.mp4 --backend modal --gpu L4 --workers 2 --state cold --execute
python -m experiments.modal_benchmark run recording.mp4 --backend modal --gpu L4 --workers 2 --state warm --execute
```

Each remote cell uses a unique class parameter to isolate its container pool. Cold
cells start without resident workers but require pre-provisioned weights. Warm cells
explicitly reserve and warm the requested number of workers before the end-to-end
timer, keep them resident through preparation/upload, and release that reservation in
`finally`. Worker IDs verify warm residency; mixed or insufficiently warmed runs fail
validation rather than being reported as warm. Include this warm reservation and
warmup in actual cost even though it is outside the warm latency interval. Hard-killing
a warm benchmark can leave its reservation active: stop the app in Modal afterward.

Local cold cells include model loading; warm cells preload the model. Local process
imports and benchmark-only input staging are excluded from both latency classes.
Local staging prevents the frozen baseline from overwriting an existing transcript.
Normal remote end-to-end timing includes preparation, upload, calls, merge, and output.
First-use image construction and model provisioning are separate campaign measurements.
Record scheduling and container startup from Modal traces: client/server clock deltas
are not reliable measurements of pure scheduling latency.

Use `--transfer direct` on a limited pilot to compare chunk payloads with the default
Volume upload. Direct transfer time is included in per-call submission round trips,
not reported as an isolated upload duration. A one-hour default recording sends
115.2 MB through the Volume strategy or 127.68 MB in direct overlapping chunks. Both
use PCM16; no extra lossy conversion is introduced. External object-storage integration
and FLAC compression are deferred.

Reports contain end-to-end time, preparation/upload/merge durations, chunk inference
and audio-read durations, call submission/retrieval intervals, worker IDs, observed
peak inference concurrency, bytes uploaded, and application retry counts. CUDA timing
is synchronized in the remote worker. Platform crash retries and billable startup/idle
time require Modal traces/billing and are not inferred from successful result timings.
Call return time includes polling and checkpoint overhead. These metrics must not be
added together as if overlapping stages were all serial.

### Budget and cost accounting

The proposed campaign cap is $10 gross spend; it is not authorization to run it. The
harness reserves $1 per remote cell by default (`--reserve-usd`) and refuses the next
cell if accumulated reservations/actual charges would exceed `--budget-usd`. Increase
the reservation for expensive models or slow recordings. Reservations survive failed
runs and are only released by explicit reconciliation. This estimate guard is not a
provider-enforced cap; configure the Modal workspace budget and monitor live usage.
Include first-use provisioning and other campaign charges in the remaining budget.

Reconcile each run with gross platform charges, including startup, CPU/RAM, idle
reservations, retries, applicable storage and transfer, and its warmup. Allocate shared
campaign overhead consistently. Credits must not reduce reported gross costs:

```bash
python -m experiments.modal_benchmark reconcile-cost benchmark-results RUN_ID 0.09
python -m experiments.modal_benchmark summarize benchmark-results
```

Missing billing remains `null`; the report does not manufacture a cost or a pass.
At September 21, 2026 list rates, L4 + two CPU cores + 8 GiB RAM is approximately
$0.000266/container-second; the $0.10 gate allows only about 376 aggregate seconds per
audio hour before other charges. This is a sensitivity estimate, not a benchmark.
Check [current prices](https://modal.com/pricing) before a campaign.

### Quality and adoption gates

Pass `--reference corrected.txt` for full-transcript normalized word-error counts.
Use human-corrected references for each boundary +/-10 seconds and representative
interior passages. Segment-level timestamps cannot reliably clip individual words at
these windows; manually align boundary hypotheses and evaluate those snippets with
`word_errors`. Record insertion/deletion counts and a human duplicate/missing-passage
review alongside reports. Unchunked Whisper output is a diagnostic, not ground truth.

Adopt the smallest worker count satisfying all of:

1. Median end-to-end speedup >=1.5x versus the actual original local implementation
   on each 30/60/120-minute recording, separately for cold and warm runs.
2. Gross cost < $0.10 per original audio hour for cold and warm long-recording cases.
3. Corpus WER increase <= one absolute percentage point; boundary insertion/deletion
   rate increase <= one point; no missing/reordered passages or systematic new duplicates.

Report medians and ranges over three repetitions; do not infer population p95 from
three samples. The summary intentionally leaves acceptance pending until paired quality
and boundary review and actual billing are complete. If no configuration qualifies,
retain local and document that Modal misses this cost target.

## Deployed acceptance checklist and follow-ups

Before recommending opt-in use, verify warm model reuse, cache reuse after scale-to-zero,
Volume visibility from reused workers, 1/2/4 observed concurrency, out-of-order output,
transient I/O retry, explicit timeout, Ctrl+C cancellation, abrupt client termination,
expired-result recovery, and cleanup. Test small recordings before the full matrix.

Then evaluate 180/300/600-second chunks with 10/30-second overlap independently of
the primary comparison. Evaluate recording-level language detection using speech-bearing
samples. A later shared merge experiment may use word timestamps and midpoint ownership
of overlaps; do not change merge behavior only for Modal. Batching, faster-whisper,
quantization, FLAC, and GPU snapshots remain separate follow-ups.

Rollout remains: offline implementation -> authorized prototype and GPU pilot ->
measured go/no-go -> documented opt-in use. Rollback is `--backend local`; stopping the
Modal app also removes active warm reservations. Estimated remaining effort is 1-2 days
for deployed validation/benchmark analysis plus reference annotation and any discovered
fixes; optional quality improvements are separate work.
