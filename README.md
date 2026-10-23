# Nemotron 3.5 Bolt on Runpod

Deploy [NVIDIA Nemotron 3.5 Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) on Runpod and connect it to an OpenAI-compatible chat app. The model is included in the image, with **32K context by default** and streaming responses.

## Choose your GPU

| GPU | Memory | Context preset | Status |
|---|---:|---:|---|
| **RTX PRO 6000 Blackwell** | **96 GB** | **32K** | Default target |
| RTX PRO 6000 Blackwell MIG 2g.48gb | 48 GB | 32K | Retest required |
| RTX 5090 | 32 GB | 2K | Experimental |
| RTX PRO 4500 Blackwell | 32 GB | 2K | Experimental |

The 96GB option includes Server, Workstation and Max-Q editions. All profiles still need validation with the current image. Use a GPU from this table; other GPU families require a different build.

**Choose the GPU and its matching preset separately.** A preset changes settings, not hardware. The 32GB options are not suitable for the default 32K setting.

## Get started

1. In Runpod, deploy this repository as a **queue Serverless** endpoint.
2. Choose your GPU and matching preset. Allow CUDA versions **13.0–13.4** and use **80 GB container disk**.
3. Leave **Model / Model Store empty** and **do not attach a network volume**. The model is already included.
4. Start with **0 minimum workers** and **1 maximum worker**. Leave **Faster startup** and **LiLiCorr speculative decoding** enabled.
5. Deploy, wait for the worker to become ready, then send a short test message.

The first launch can take longer while the image loads. You do not need a Hugging Face token. Runpod GPU usage is billed to your account.

## Connect your chat app

Choose your app's **OpenAI-compatible** or **custom OpenAI** connection and enter:

| Field | Value |
|---|---|
| API base URL | `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1` |
| API key | Your Runpod API key |
| Model | `nemotron-bolt` |

Replace `YOUR_ENDPOINT_ID` with the ID from your Runpod endpoint. Streaming uses the app's normal streaming option. For direct API use, see the [examples and API reference](docs/CONFIGURATION.md#openai-compatible-api).

## Set the context size

Context is the space available for **your conversation plus the model's reply**. It includes instructions, chat history and any attached media. Longer conversations need more context and GPU memory.

- **32K:** 32,768 tokens in total. Reserving 4,096 for the reply leaves 28,672 for the complete input.
- **2K:** 2,048 tokens in total. Reserving 256 for the reply leaves 1,792 for the complete input.

To change it, edit **Context window** in the endpoint settings (`MAX_MODEL_LEN`), then restart the workers. If you upgraded from a 32GB GPU, check that an old `2048` override is not still applied.

Start with the preset for your GPU. Larger context sizes are not yet validated, and the model's “24GB” name does not include all the memory needed to run it.

## Having trouble?

- **No GPUs available:** retry later or choose another listed GPU with its matching preset.
- **Out of memory:** check the GPU and context setting. Reduce context or choose a larger compatible GPU.
- **Conversation too long:** start a new chat, shorten the history or reduce the reply limit.

**Preview status:** live GPU/API validation is still pending. The current inference backend is not intended for production/datacenter use; see [status and limitations](docs/VALIDATION.md) before deploying.

[Advanced settings](docs/CONFIGURATION.md) · [Build and maintenance](docs/MAINTAINER.md) · [License](LICENSE) · [Model and dependency notices](NOTICE)
