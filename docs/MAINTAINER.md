# Maintainer handoff

## Checks without a GPU

Use Python 3.12. The queue SDK runs separately from vLLM because their FastAPI requirements conflict. Tests exercise the original Runpod startup, argument builder and handler. Outside the base image, fetch the pinned upstream test source first:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-test.lock
git clone --depth 1 --branch v2.30.0 https://github.com/runpod-workers/worker-vllm.git .test-upstream
export RUNPOD_SOURCE_DIR="$PWD/.test-upstream/src"
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, use `.venv/Scripts/python.exe` and `$env:RUNPOD_SOURCE_DIR=(Resolve-Path '.test-upstream/src').Path`. In the container, tests default to `/src`. Tests verify hashes for all seven imported upstream files and fail on drift or missing source. The fixture HTTP server uses loopback and synthetic responses, not NVIDIA weights or the Runpod gateway. The original source is only a local test dependency; it is inherited unchanged from the image at deployment.

Regenerate the queue lock deliberately with `uv pip compile requirements.in --python-version 3.12 --universal --generate-hashes --no-annotate --output-file requirements.lock`. The direct SDK-environment requirements are Runpod 1.12.0, aiohttp 3.14.5 and huggingface-hub 2.2.0. The original startup imports Hugging Face helpers even when the baked model runs offline. Review changes and rerun checks. Do not upgrade the inference image or B12X without GPU qualification.

## Build and inspect

Use a Linux/amd64 Docker builder with at least 150 GB free for the framework, 25.08 GB of baked assets, intermediate layers and cache. The base contains approximately 9.47 GB of compressed layers before additions. Final compressed image size is unmeasured; Docker is not installed on the authoring host.

```bash
docker buildx build --platform linux/amd64 --load -t nemotron-bolt-worker:0.4.0 .
docker image inspect nemotron-bolt-worker:0.4.0
docker run --rm --entrypoint /opt/queue/bin/python nemotron-bolt-worker:0.4.0 \
  -c 'import runpod,aiohttp; print(runpod.__version__,aiohttp.__version__)'
docker run --rm --entrypoint cat nemotron-bolt-worker:0.4.0 /app/inference-packages.txt
```

The Dockerfile inherits the Runpod image by digest, installs the exact hash-pinned NVIDIA vLLM nightly and dependency adjustments, and isolates the SDK in `/opt/queue`. Build checks verify the original startup/helper/handler hashes and recursively validate inference dependencies. The SDK interpreter also imports the original startup during the build, catching missing dependencies before the model download. See [base compatibility](BASE-IMAGE.md). No upstream source file is copied over or edited.

The builder downloads locked assets into `/runpod-volume/models/bolt` and `/runpod-volume/models/lilicorr`, verifies every weight SHA-256, writes revision markers and removes download bookkeeping. The final image retains upstream notices and sets model libraries offline. No model code executes during download. B12X can still compile necessary kernels at startup, even in eager mode.

The public repositories need no Hugging Face token. If access changes, use `docker buildx build ... --secret id=HF_TOKEN,env=HF_TOKEN`; never use a token build argument.

Check baked metadata and model presence without networking or a GPU (full asset hashes are checked during the build):

```bash
docker run --rm --network none --entrypoint python3 nemotron-bolt-worker:0.4.0 \
  -c 'import json,pathlib; m=json.load(open("/local_model_args.json")); print(m); assert pathlib.Path(m["MODEL_NAME"],"config.json").is_file(); assert pathlib.Path("/runpod-volume/models/lilicorr/config.json").is_file()'
```

Keep the final image digest, `/app/inference-packages.txt` and model revisions as release evidence. The current image has not been built on the authoring host.

## GPU acceptance

The entrypoint is `/opt/queue/bin/python /src/main.py`. That unchanged original startup reads `/local_model_args.json`, launches the global `vllm` CLI, waits for engine health and starts the original handler. `MAX_CONCURRENCY=1` matches `MAX_NUM_SEQS=1`. There is no custom startup, GPU guard or extra readiness generation. Example on a supported Blackwell GPU:

`.runpod/tests.json` targets one full **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB**, at `MAX_MODEL_LEN=32768`. The deployment selector also admits its 48GB MIG partition via `BLACKWELL_96,ADA_48_PRO,-NVIDIA L40,-NVIDIA L40S,-NVIDIA RTX 6000 Ada Generation`. Keep all three exclusions. Native presets only change environment values: switching to an experimental 32GB profile requires changing GPU selection separately and setting `MAX_MODEL_LEN=2048`. Refresh pool membership before editing the selector. Hub tests document one primary GPU, not an arbitrary fallback list; the latest Secure Cloud Pod read had no 48GB MIG stock, so the primary remains 96GB. Both Hub files allow CUDA 13.0–13.4, preserving the image's CUDA 13.0 floor.

```bash
docker run --rm --gpus all --ipc=host nemotron-bolt-worker:0.4.0 \
  --test_input '{"input":{"openai_route":"/v1/chat/completions","openai_input":{"model":"nemotron-bolt","messages":[{"role":"user","content":"What is 2 plus 2? Reply with one digit."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}}}'
```

Repeat with tools, streaming, image, thinking, invalid input followed by a valid request, and the configured context limit. The standard handler yields error objects and raw SSE strings; see README before interpreting output. `MAX_OUTPUT_TOKENS` and earlier custom request limits no longer apply. SDK CLI arguments are passed through the original startup as in the upstream image; invoking it can load the model before SDK argument handling.

Measure fresh image pulls separately from engine startup. Compare fresh containers on the same GPU with the default `ENFORCE_EAGER=true` and `-e ENFORCE_EAGER=false` (before the image name). Record startup, first-token latency, throughput, peak memory and cancellation/recovery. Keep the upstream health check and no network volume attached. Measure the first request separately because it can include remaining kernel warmup. No eager-mode speedup has been measured yet.

## Runpod and Hub acceptance

1. Build/push the image or connect the repository to Runpod's GitHub builder using the root Dockerfile. Check builder disk/time limits for the large image.
2. Use a queue endpoint with README GPU/settings, no Model Store and no network volume. Preserve endpoint ID, image digest, model revisions and GPU name.
3. Export `RUNPOD_API_KEY`, then run `python scripts/acceptance.py --endpoint YOUR_ENDPOINT_ID --output validation-runpod.json`. This incurs real GPU usage: eight queue jobs (three text, invalid input, recovery, streaming, tools and image), then checks of the real `/openai/v1/models`, chat and streaming gateway routes. It validates contents and records evidence. Test Responses/Messages separately before advertising those routes.
4. Test `/stream/{job_id}` during execution, cancellation, subsequent jobs and worker replacement. Verify GPU work stops after cancellation. Test thinking, structured JSON, representative video and maximum context. Exercise your actual OpenAI SDK client against the rebuilt endpoint.
5. Review B12X's upstream scope and unresolved gates in [VALIDATION.md](VALIDATION.md). Source checks and one successful job do not establish production readiness.
6. Create a GitHub release and submit the repository to Hub. Listing/tests are in `.runpod/`; confirm Hub build/tests and manual review. No Hub listing is published by this task.

The acceptance script cancels only its own queue job if polling fails or times out. It does not stop workers or delete the endpoint. Gateway calls follow normal HTTP behavior; check Runpod for work remaining after a disconnected client. Keep minimum workers at zero unless ongoing warm-worker cost is intentional, and remove disposable resources after testing.
