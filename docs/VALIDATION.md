# Validation record — 2026-10-23

| Stage | Status | Evidence / boundary |
|---|---|---|
| Source and Hub files | Authored | Dockerfile, native model settings/baked metadata, manifest, Hub config/tests, examples, license and acceptance script |
| Model/source identification | Passed | Pinned HF revisions and inventories; custom code/configs and weight hashes |
| Base image identity | Passed | Registry-resolved Linux/amd64 digest; CUDA 13.0.2 and Python 3.12 from image metadata |
| Queue dependency lock | Passed | Python 3.12.14; Runpod 1.12.0, aiohttp 3.14.5 and huggingface-hub 2.2.0; compatible locked dependencies installed. Existing package pins retained; five dependencies added for startup imports |
| CPU integration/build checks | Passed | 26 unittest cases; see `cpu-tests.txt`. Includes GPU pool exclusions, CUDA floor, preset translation, original handler against local HTTP fixtures and real SDK generator |
| Telemetry conflict regression | Passed locally | Mixed versions reproduced and hash-locked repair verified; see `otel-regression.txt` |
| Prior NVIDIA inference | Limited supplied evidence | Earlier custom-handler, compiled image completed readiness and one queue job; response semantics not independently checked |
| Current image build | Required | Original-startup/eager changes not image-built locally; no Docker builder or final digest |
| GPU fit/performance | Current 48/96GB image acceptance required | Prior compiled image started at 32K and completed a job on RTX PRO 6000 Blackwell MIG 2g.48gb. Prior 32GB test profile failed at 32K. No current eager-image pass for the eligible 48/96GB GPUs or experimental 32GB/2K presets |
| OpenAI gateway and Runpod acceptance | Required | Protocol tested locally; updated image not tested through Runpod's actual `/openai/v1` gateway |
| Hub listing | Latest attempt failed GPU allocation | User reports no instance for the 96GB Server Edition or Workstation/Max-Q fallbacks. Earlier 32GB attempt returned startup-memory errors. No successful current Hub tests or publication evidence |

## Pinned inputs

- Main model: `cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f`
- LiLiCorr: `449d24140160b8e51871bfc0f0a7338abf640aa1`
- Base: `runpod/worker-v1-vllm:v2.30.0`, Linux/amd64 `sha256:437ddeda2917d21fdd16c606d472db47129d0ab3ff55041db1801cbd81de475c`
- Original `/src/handler.py` SHA-256: `69e14fbbe81ad4b27695f48da3089472faf940a69518e75913a5422b65abce00`; Git blob `c35965423027173f066f4c1567917df5b5ced952`
- vLLM: `0.32.0rc1.dev100+g77154ca61`, commit `77154ca6116ae34cbfe314beef1016bde9b63989`; wheel SHA-256 `92e3958a32e65f19ce19dea3cc7a2e1c9edd57565f461617841abfc0a66bcd98`
- B12X 1.5.1 wheel SHA-256: `8731e23571d8d45db26d78e7699ec0b23e7ba20b52487adf8c1063c8d8b53a3d`
- Original `/src/main.py` SHA-256: `3b8b93461216f346f15e56d0633a350cb760f82312051e01ba2a091c575d644a`; helper hashes in `scripts/build_checks.py`
- Runpod SDK 1.12.0, aiohttp 3.14.5 and huggingface-hub 2.2.0; full queue hashes in `requirements.lock`

## Current change and CPU evidence

Hub's required root `handler.py` is now present as a byte-for-byte copy of `worker-vllm` v2.30.0 `src/handler.py`, retaining its MIT license and the hash recorded above. The Docker build copies it to `/src/handler.py` before existing source checks. Startup still runs the original `/src/main.py`. Handler integration checks now exercise the repository copy; configuration checks also require its Docker inclusion. No handler logic or model settings changed.

The deployment selector is now `BLACKWELL_96,ADA_48_PRO,-NVIDIA L40,-NVIDIA L40S,-NVIDIA RTX 6000 Ada Generation`: three 96GB Blackwell variants plus the exact 48GB Blackwell MIG partition. The test GPU remains `NVIDIA RTX PRO 6000 Blackwell Server Edition`; the later capacity reads show LOW Secure Pod stock for it and NONE for the 48GB MIG. Both Hub files allow CUDA 13.0–13.4. Docker/Hub/test context remains 32,768. A 48GB MIG preset is added alongside the 96GB profile and experimental 32GB/2K profiles. Presets only set environment values. See `gpu-catalog-snapshot.json` for separate Serverless, Pod and CUDA-version reads on 2026-10-23.

The entire custom runtime package is removed. The entrypoint directly runs the unchanged `/src/main.py`, which launches the global vLLM CLI and registers the unchanged handler. Baked metadata fixes the model/tokenizer paths; environment variables and `VLLM_EXTRA_ARGS` supply Nemotron flags. The SDK and startup dependencies remain isolated in `/opt/queue` because of the FastAPI conflict.

Tests verify upstream source hashes, native startup registration after health, baked-path precedence/offline preflight, the complete generated Nemotron arguments, matching Docker/Hub/test defaults at 32K, native preset conversion, and the GPU filter's exact four-card result against the catalog. CUDA versions match across Hub files and retain the 13.0 floor; the test GPU remains a single documented ID. They also cover Hub eager/speculation toggles and OpenAI chat envelopes, GET models, raw SSE including completion markers, Responses/Messages routing, native sampling and default model resolution, upstream errors and recovery through the real SDK generator. HTTP responses are synthetic; routing does not prove model support. The acceptance parser checks split SSE and missing completion markers.

Build-only checks cover asset sizes/hashes, pinned revisions, both downloads and failures. No custom startup manifest/GPU validation or readiness generation remains. Upstream startup tests mock process and health operations; actual engine launch, retries, signals and first-request warmup need Linux/GPU acceptance. Custom runtime limits and output transformations are no longer shipped.

## Supplied startup evidence

The latest supplied Hub failure occurred before container testing: no 96GB Server Edition instance or either listed 96GB fallback could be allocated. It is not another model-memory exception. The subsequent catalog audit found an additional architecture-compatible Serverless candidate, the 48GB MIG partition, but its Secure Pod stock was NONE. It is added to deployment eligibility with the incompatible Ada pool members excluded. The CUDA allowlist is broadened to the released 13.0–13.4 versions, without claiming newly available hosts: all eligible stock in the current probes is on 13.2. The documented tests schema offers one primary GPU, so no unsupported fallback array is added and the 96GB test primary remains.

The subsequent Hub results supplied on 2026-10-23 returned the same `startup_error` for all three tests. The stock startup reported an estimated maximum context of about 2,240 tokens, while `.runpod/tests.json` requested 32,768. The error outputs do not identify the actual GPU allocation; that test configuration selected an RTX 5090. A temporary 2,048-token default was added without a subsequent GPU result. The user then requested a full 96GB RTX PRO 6000 test GPU and restored 32K context. The 2K settings are retained only as experimental 32GB presets; neither those presets nor the new 96GB/32K profile has a successful acceptance result. Eager mode, one sequence, the 2,048-token batch budget, FP8 KV cache, 90% memory allocation and LiLiCorr remain unchanged.

The earlier `logs (7).txt` run on 2026-10-23 (UTC) recorded GPU detection at 09:59:21.475, readiness at 10:04:51.427, acceptance around 10:04:54, and completion at 10:04:56.247: about 5 minutes 33 seconds to acceptance, excluding image pulls. Loading reported 26 seconds; engine initialization reported 245.96 seconds including 49.81 seconds compilation. CUDA graph captures reported 76 and 4 seconds.

`ENFORCE_EAGER=true` now skips Torch compilation, CUDA graphs and optional JIT pre-warmup in the pinned engine. Necessary kernel compilation can still occur during execution. The drafter inherits eager mode. Speculation remains enabled and startup uses the original `/health` readiness check. No GPU startup saving or throughput claim is made for the current image.

## Remaining release gates

1. Rebuild; retain the image digest, package inventory and dependency-check results. Review vulnerabilities under deployment policy.
2. Run the updated acceptance script through Runpod, including actual OpenAI models/chat/streaming. Test your SDK client, thinking, structured JSON, tool round trips, media and Responses/Messages if required.
3. Measure memory, startup, latency and throughput on the selected GPU at the configured context. Compare eager/compiled modes. Verify cancellation, recovery, shutdown, replacement, image pulls and offline loading.
4. Review B12X's explicit production/datacenter scope limitation. Operational safeguards do not change that limitation; critical use may require another supported runtime/checkpoint.
5. Complete GitHub release, Hub build/tests and publication review.

No task-owned cloud resources were created. Research/test environments remain under `work/`; deliverables are under `outputs/`.
