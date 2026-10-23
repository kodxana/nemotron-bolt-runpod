# GPU selection

The 32K deployment profile admits the three **RTX PRO 6000 Blackwell 96GB** variants and the exact **RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb** partition. The Hub test primary remains the full 96GB Server Edition. One sequence, FP8 KV cache, LiLiCorr and eager mode remain unchanged.

The earlier supplied GPU log identifies the 48GB MIG device as SM120 with 47.4 GiB visible memory, configures `max_model_len=32768`, reaches readiness and finishes a queue job. That run used the older custom launcher and compiled mode; it supports the candidate selection but does not qualify the current eager image. The 32GB profile previously failed startup at 32K and is excluded from the default pool.

## Capacity observed on 2026-10-23

Runpod `list_gpu_types` was read separately for SERVERLESS and POD, Secure Cloud, one GPU and minimum CUDA 13.0. `get_capacity` also probed Secure Cloud across CUDA 13.0–13.4. These are snapshots, not reservations or a guarantee that Hub can allocate its test Pod.

| GPU | VRAM | Serverless pool | Serverless availability | Secure Pod availability | Serverless hourly price |
|---|---:|---|---|---|---:|
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB | `BLACKWELL_96` | LOW | LOW | $3.49 |
| NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 96 GB | `BLACKWELL_96` | NONE | NONE | $3.49 |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 96 GB | `BLACKWELL_96` | NONE | NONE | $3.49 |
| NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb | 48 GB | `ADA_48_PRO`, filtered | MEDIUM | NONE | $1.75 |

The 48GB MIG read reports capacity for Serverless and no capacity for Secure Cloud Pods; these are different product contexts. The user's Hub attempt could not allocate any of the three 96GB variants. The later Pod read reports LOW stock for the Server Edition on CUDA 13.2, with no stock on 13.0, 13.1, 13.3 or 13.4. Keeping the 96GB test primary avoids switching to a candidate whose Pod stock is NONE. Recheck at the time of a new run; no current allocation success is claimed.

Prices are the returned `price.serverless` values, not invoice estimates. Query parameters, per-product results and driver probes are retained in [gpu-catalog-snapshot.json](gpu-catalog-snapshot.json).

## Pool filters and test limits

The deployment selector is:

```text
BLACKWELL_96,ADA_48_PRO,-NVIDIA L40,-NVIDIA L40S,-NVIDIA RTX 6000 Ada Generation
```

The 48GB MIG partition shares `ADA_48_PRO` with L40, L40S and RTX 6000 Ada. All three exclusions are required; an unfiltered pool can schedule an incompatible architecture. CPU regression checks resolve the selector against the captured catalog and require exactly the four eligible GPU IDs above. Refresh the catalog and review new pool members before changing this filter.

[Hub's publishing schema](https://docs.runpod.io/hub/publishing-guide#test-environment-configuration) documents one `gpuTypeId` for tests. The supplied allocation error demonstrates Runpod's automatic fallbacks to the Workstation and Max-Q variants; no custom multi-GPU fallback field is documented. Expanding `hub.json` affects deployed endpoint workers, not the test runner's primary GPU selection. No unsupported `gpuTypeIds` array or comma-separated test ID is used.

Both Hub files permit CUDA **13.0, 13.1, 13.2, 13.3 and 13.4**. The image uses CUDA 13.0.2; newer drivers support older CUDA runtimes through [NVIDIA's backward compatibility](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html). This removes unnecessary version exclusions but the present reads found eligible stock only on 13.2. CUDA 12.x is not allowed.

## Other GPUs

The pinned [B12X 1.5.1](https://pypi.org/project/b12x/1.5.1/) requires SM120/SM121. [NVIDIA's compute capability table](https://developer.nvidia.com/cuda/gpus) places RTX PRO Blackwell on SM120; H100/H200, A100, L40/L40S, RTX 6000 Ada, B200 and B300 have different architectures. Extra VRAM does not make those GPUs compatible with this image.

RTX PRO 5000 Blackwell 48GB is architecture-compatible, but the catalog supplies no Serverless pool or Serverless price and reports no Secure Pod capacity. It is not advertised as a new deployable pool. The RTX 5090 and RTX PRO 4500 Blackwell retain experimental 2K presets, requiring manual hardware selection and independent GPU validation. Their 32GB memory does not qualify them for the default 32K profile. The 24GB Blackwell MIG partition is also excluded.

Native presets set environment values; they do not select hardware. Do not extend context or concurrency beyond the tested profile merely because a full 96GB GPU is present. If a later profile requires 96GB specifically, remove the 48GB pool from that endpoint.

The main baked assets occupy 24,119,509,141 bytes (22.46 GiB); the drafter occupies 962,510,929 bytes (0.90 GiB). Storage size is not VRAM consumption: CUDA context, vision activations, Mamba state, KV cache and allocator overhead add to it.

Suggested host resources remain at least 8 vCPUs, 64 GB RAM and 80 GB container disk. They are not encoded as unsupported Hub keys. Measure image pulls separately from engine startup and first-request kernel warmup. No network volume is required. B12X's production/datacenter scope limitation remains independent of GPU capacity.
