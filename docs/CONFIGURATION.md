# Advanced configuration and API reference

For everyday setup, start with the [user guide](../README.md). This reference covers environment settings, GPU pools, request formats and implementation details.

## GPU and context settings

Start with **one RTX PRO 6000 Blackwell 96GB and the default 32K preset**. These are configured starting points, not measured maximum capacities.

| GPU / Hub preset | VRAM | `MAX_MODEL_LEN` | Initial request `max_tokens` | Validation status |
|---|---:|---:|---:|---|
| **RTX PRO 6000 Blackwell 96GB — 32K (default)** | 96 GB | **32768** | 4096 or less | Current Hub test target; GPU acceptance pending |
| RTX PRO 6000 Blackwell **MIG 2g.48gb** — 32K | 48 GB | **32768** | 4096 or less | Earlier compiled image started and completed a job; current image retest required |
| RTX 5090 — 2K (experimental) | 32 GB | **2048** | 256 or less | Earlier 32K profile ran out of memory; revised 2K profile unverified |
| RTX PRO 4500 Blackwell — 2K (experimental) | 32 GB | **2048** | 256 or less | Capacity unverified |

The 96GB profile covers the **Server Edition, Workstation Edition and Max-Q Workstation Edition**. Hub tests target the full 96GB Server Edition. The default deployment selection permits those three cards and the exact 48GB MIG partition; 32GB cards are excluded from that selection.

Every preset keeps **eager mode enabled, LiLiCorr enabled, one active sequence and one job per worker**. Short acceptance requests check basic functionality; they do not prove that a full 32K conversation, large tool schema or media workload fits.

### Apply the right preset

1. Choose the GPU hardware in Runpod.
2. Select the matching Hub preset, or set `MAX_MODEL_LEN` to the value above in the endpoint environment.
3. Check existing environment overrides, then restart/redeploy workers so the engine reads the new settings.
4. Run acceptance, then test representative inputs near your intended context limit before increasing concurrency or traffic.

**Presets change environment values only. They do not select or detect the GPU.** If an existing endpoint has `MAX_MODEL_LEN=2048`, switching its image or GPU does not remove that override. Change it to `32768` when moving to the 32K profile.

This image's B12X kernels require **SM120/SM121**. H100/H200, A100, B200/B300, L40/L40S, RTX 6000 Ada and RTX 4090 are not compatible with this build. VRAM alone is not sufficient. See [hardware compatibility and dated availability](HARDWARE.md).

<details>
<summary>Advanced: GPU pools and Hub test fallbacks</summary>

The default deployment selector is:

```text
BLACKWELL_96,ADA_48_PRO,-NVIDIA L40,-NVIDIA L40S,-NVIDIA RTX 6000 Ada Generation
```

The 48GB Blackwell MIG shares `ADA_48_PRO` with incompatible Ada cards, so **keep all three exclusions**. Never select that pool unfiltered. To require a full 96GB card, use `BLACKWELL_96` alone. Experimental RTX 5090 and RTX PRO 4500 profiles need separate hardware selection (`ADA_32_PRO` and `BLACKWELL_32`, respectively).

Hub's [test configuration](https://docs.runpod.io/hub/publishing-guide#test-environment-configuration) accepts one primary `gpuTypeId`; Runpod manages its fallbacks. Changing deployment pools does not change the test runner's primary GPU. Both Hub files permit host CUDA compatibility versions **13.0–13.4**; the container contains CUDA 13.0.2. A broader allowlist does not guarantee available instances. See the [capacity findings](HARDWARE.md).

</details>

## How context size works

**`MAX_MODEL_LEN` is the total token budget for input plus generated output.** It is an engine setting, applied when the worker starts. **`max_tokens` is the output limit for one API request.** Lowering `max_tokens` cannot fix an engine that fails to start with an oversized context allocation.

```text
tokenized input + requested maximum output <= MAX_MODEL_LEN
```

Input includes system instructions, chat history, tool definitions, chat-template overhead and image/video tokens. If thinking is enabled, thinking and the final answer share the output budget. Tokens are not characters; use the model's tokenizer/template to estimate input length.

| Context setting | Output reserved with `max_tokens` | Budget for the complete tokenized input |
|---|---:|---:|
| `MAX_MODEL_LEN=32768` | 4096 | 28672 tokens |
| `MAX_MODEL_LEN=2048` | 256 | 1792 tokens |

These are budget calculations, not memory guarantees. Include template overhead when estimating from raw text.

### Why a 32K worker has a 2K batch setting

`MAX_NUM_BATCHED_TOKENS=2048` controls the tokens scheduled in a batch. With `ENABLE_CHUNKED_PREFILL=true`, the engine processes longer prompts in smaller chunks. **It does not reduce the context window to 2K.** Keep this conservative batch setting while validating GPU memory use.

### Changing context safely

- **Start at 32K on 96GB.** Larger windows need their own startup and workload tests. The Hub field's `262144` ceiling is an input limit, not a promise that 256K fits any listed GPU.
- **Retest the 48GB MIG profile.** Earlier 32K startup evidence came from a different launcher/compiled configuration and a short job.
- **Treat 2K on 32GB as experimental.** The previous 32GB test profile estimated about 2240 tokens of capacity after 32K startup failed; 2K has not subsequently passed GPU acceptance.
- Keep `MAX_NUM_SEQS=1` and `MAX_CONCURRENCY=1` initially. Increasing either changes memory pressure and needs separate testing.

The model's “24GB” name describes compact weights, **not total running GPU memory**. The drafter, KV cache, Mamba state, activations and runtime overhead also need memory. If a future context profile requires 96GB, restrict that endpoint to the 96GB pool so it cannot land on a 48GB partition.

## Deploy

The listing is not published yet. Connect this repository through Runpod's GitHub deployment flow using the root `Dockerfile`, or follow the [image build instructions](MAINTAINER.md).

1. Create a **queue Serverless** endpoint and choose a GPU/preset from the table above.
2. Leave **Model / Model Store empty** and **do not attach a network volume**. A mount at `/runpod-volume` would hide the baked models.
3. Use **80 GB container disk** and a host compatible with CUDA **13.0 or newer**. Suggested host resources: at least **8 vCPUs / 64 GB RAM**; verify the allocation.
4. Start with **minimum workers 0, maximum workers 1, idle timeout 30 seconds**. Suggested execution timeout: **900 seconds**; the worker request timeout is 600 seconds.
5. Run the [acceptance script](../scripts/acceptance.py) against the deployed endpoint, including OpenAI models, chat and streaming. These are real GPU jobs and can incur charges. Test longer prompts and required features before opening up traffic.

No runtime Hugging Face token is required. Model loading is offline; the worker still needs network access to Runpod. Original startup launches vLLM, waits for `/health`, then starts the original handler. No public port 8000 or load-balancer configuration is needed.

## OpenAI-compatible API

Use your **Runpod API key** and the usual endpoint base URL:

```text
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1
```

The served model name is **`nemotron-bolt`**. With the OpenAI Python SDK installed:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RUNPOD_API_KEY"],
    base_url=f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}/openai/v1",
)
print(client.models.list())
response = client.chat.completions.create(
    model="nemotron-bolt",
    messages=[{"role": "user", "content": "Explain binary search."}],
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

Set `stream=True` and iterate over the returned stream using the SDK's normal chat streaming interface. Set an explicit output budget. Tools, structured output and thinking fields are forwarded to vLLM; your application executes tool calls. Media uses standard content parts, with one image and one video allowed per request by the current engine settings.

The original handler also forwards Responses and Messages routes. Forwarding does not guarantee model support: validate the features your application needs. This updated image has not yet passed live gateway acceptance. See the [upstream API contract](https://github.com/runpod-workers/worker-vllm/blob/v2.30.0/README.md#usage-openai-compatibility).

### Native Runpod requests

Submit to `/run` or `/runsync` using the original handler's request envelope. Retrieve asynchronous results through `/status/{job_id}` or `/stream/{job_id}`:

```json
{
  "input": {
    "openai_route": "/v1/chat/completions",
    "openai_input": {
      "model": "nemotron-bolt",
      "messages": [{"role": "user", "content": "Explain binary search."}],
      "max_tokens": 256,
      "chat_template_kwargs": {"enable_thinking": false}
    }
  }
}
```

Complete request files: [text](../examples/text.json), [streaming](../examples/stream.json), [tool call](../examples/tool-call.json).

A nonstreaming native result is in `output[0]`. Native streaming returns raw SSE strings that may split event boundaries; concatenate and parse them as SSE. Check both job status and output: the handler may return an `error` inside a `COMPLETED` job. OpenAI SDK clients receive normal API responses and streams through the gateway instead.

## Settings reference

These defaults are already in the image. Hub controls expose context, eager mode and LiLiCorr; other settings can be supplied through the endpoint environment.

| Setting | Default | What it controls |
|---|---|---|
| `MAX_MODEL_LEN` | `32768` | Total input + output window; see the GPU table |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Scheduling batch budget, not the context window |
| `ENABLE_CHUNKED_PREFILL` | `true` | Processes long prompts in smaller batches |
| `MAX_NUM_SEQS` / `MAX_CONCURRENCY` | `1` / `1` | Engine sequence limit / Runpod job concurrency |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM GPU-memory allocation fraction; raising it leaves less safety margin |
| `ENFORCE_EAGER` | `true` | Skips Torch compilation and CUDA graph capture |
| `KV_CACHE_DTYPE` | `fp8` | KV cache format for this Blackwell profile |
| `SPECULATIVE_CONFIG` | LiLiCorr JSON from the Dockerfile | Hub toggle controls the drafter; set an **empty string** to disable manually |
| `VLLM_STARTUP_TIMEOUT` | `1800` | Engine health deadline in seconds per upstream startup attempt |
| `REQUEST_TIMEOUT` | `600` | Handler HTTP request timeout in seconds |
| `SERVED_MODEL_NAME` | `nemotron-bolt` | Name sent by API clients |
| `VLLM_EXTRA_ARGS` | Nemotron backend/parser flags | Required model-specific flags; preserve them |

### Startup speed and speculation

Eager mode skips Torch compilation, CUDA graphs and optional JIT pre-warmup in the pinned engine. Model loading, health checks and necessary B12X/Triton/FlashInfer kernel compilation remain. The first request may still include warmup, and generation may be slower than compiled mode. Set `ENFORCE_EAGER=false` to compare on the same GPU; actual eager startup time and throughput have not yet been measured.

LiLiCorr uses **7 speculative draft tokens**. Switching it off stops loading the drafter and changes memory/performance behavior, so retest the profile. It does not remove the baked files.

Compiled caches depend on GPU architecture, software and engine settings, not one physical card's identity. Caches here live in disposable container storage; fresh containers cannot assume they exist. CUDA graphs are process-local. Baked weights avoid model downloads, but fresh hosts must still pull the larger image. See [startup measurements](VALIDATION.md#supplied-startup-evidence).

## Troubleshooting

| Symptom | What to check |
|---|---|
| “No instances available” before testing | Allocation failed before the model started. Recheck stock and allowed CUDA versions. Deployment pools do not change Hub's test GPU. See [hardware notes](HARDWARE.md). |
| `startup_error` / out of GPU memory | Verify the actual GPU and `MAX_MODEL_LEN`, including endpoint overrides. Keep eager mode on, batch budget at 2048 and concurrency at 1. Use a larger compatible GPU or lower context; disabling LiLiCorr is another profile to retest. |
| Request exceeds context length | Reduce input/history or `max_tokens` so the complete input plus output budget fits. Raising the engine limit requires a restart and GPU validation. |
| Still at 2K after switching to 96GB | Update the endpoint's `MAX_MODEL_LEN=2048` override, then restart workers. |
| Model path missing / unexpected download | Leave Model Store empty and remove any volume mount covering `/runpod-volume`; it hides the baked files. |
| `COMPLETED` job contains no answer | Inspect `output` for an upstream error. Queue completion alone does not prove successful inference. |

## Baked models and build details

| Model | Fixed image path |
|---|---|
| Nemotron Bolt | `/runpod-volume/models/bolt` |
| LiLiCorr drafter | `/runpod-volume/models/lilicorr` |

`/local_model_args.json` points original startup directly at Bolt for model and tokenizer, taking precedence over environment path overrides. There is no runtime cache search. Both repositories are pinned in [assets.lock.json](../assets.lock.json); build checks validate file sizes, code/config hashes and every weight shard's SHA-256. Files total **25.08 GB / 23.36 GiB** before image compression; this is storage size, not VRAM usage. Keep `/cache` writable for disposable library caches.

Stock startup and handler files are hash-checked and inherited unchanged. A separate queue environment handles the SDK dependency conflict while global vLLM runs inference. See [base-image compatibility](BASE-IMAGE.md) and the [maintainer guide](MAINTAINER.md) for pins and build steps.

<details>
<summary>Migrating from the earlier custom worker</summary>

- Use the complete `openai_input` body above, or put shorthand generation options inside `sampling_params`. A flat `{"prompt":"...","max_tokens":256}` no longer applies that limit. Shorthand `messages` uses chat; `prompt` uses completions without a chat template.
- Replace `STARTUP_TIMEOUT_SECONDS` with `VLLM_STARTUP_TIMEOUT`, `REQUEST_TIMEOUT_SECONDS` with `REQUEST_TIMEOUT`, and `ENABLE_SPECULATIVE` with `SPECULATIVE_CONFIG`.
- Custom `MAX_OUTPUT_TOKENS`, the 6 MB application cap and inline-media-only restriction are gone. The original handler and vLLM validate requests; Runpod's payload limits still apply. Local-file media access remains disabled.
- The custom supervisor, GPU precheck, runtime manifest validator and extra readiness completion are removed. Original startup retries, health checks, error reporting and signal handling apply.

</details>

## License

Worker project: [Apache-2.0](../LICENSE). NVIDIA weights and bundled model code retain [OpenMDW-1.1](https://openmdw.ai/license/1-1/) terms; inherited Runpod code retains its MIT license. See [NOTICE](../NOTICE). The [B12X 1.5.1](https://pypi.org/project/b12x/1.5.1/) intended-use limitation is separate from these licenses and remains a release consideration.
