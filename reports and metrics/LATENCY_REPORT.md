# 📊 Enhanced Infrastructure Performance & Latency Report

**Project:** Clinical Diagnostic Assistant (v1.0)  
**Model:** Llama-3-8B-Instruct (4-bit NF4 Quantization + LoRA)  
**Hardware:** NVIDIA A10G Tensor Core GPU (16GB VRAM)  
**Host:** Modal Serverless Infrastructure

---

## 1. Verified Performance Metrics (Ground Truth)

Based on the final telemetry run with a **926-token** medical context:

| Metric | Observed Value | Interpretation |
| :--- | :--- | :--- |
| **Input Prompt Size** | 926 Tokens | Significant "pre-fill" load due to USMLE context chunks. |
| **TTFT (Time to First Token)** | 1.358s | The "Reading" speed. Passable for RAG on A10G. |
| **Throughput (TPS)** | 12.51 tokens/sec | **PRIMARY BOTTLENECK.** 4-bit dequantization overhead. |
| **Total GPU Execution** | 18.386s | The "Writing" speed. Limits real-time interactivity. |

---

## 2. Hardware Evidence: Visual Analysis

### A. GPU Memory Saturation
![[Insert image_be0ef3.png here]]

* **Observation:** VRAM is locked at 15.13 GiB.
* **Analysis:** While 4-bit weights only require ~5.5 GB, the system has pre-allocated nearly the entire 16 GB for the **KV-Cache**. This is necessary to handle the high input prompt size (926 tokens) without crashing mid-generation, but it proves we are at the physical memory limit of the A10G.

### B. The "Dequantization Tax" Stutter
![[Insert image_be0ef3.png here - Referencing the Utilization % graph]]

* **Observation:** GPU Utilization fluctuates between 31% and 40%.
* **Analysis:** The "valleys" in the utilization graph are the smoking gun for **Memory Bound** behavior. The GPU Tensor Cores are finishing the math so quickly that they must pause to wait for the `bitsandbytes` library to "unpack" the next set of 4-bit weights.

---

## 3. The "Mirroring" Bug Resolution
![[Insert image_ca45db.png here]]

* **The Issue:** The model previously echoed user input ("um its high temperature") instead of diagnosing.
* **The Root Cause:** Invalid prompt formatting. Llama-3-8B-Instruct is a strict instruction-follower. Without the required double-newlines (`\n\n`) after the header tags, the model reverted to **Base Autocomplete Mode**.
* **The Fix:** Enforced strict Meta-standard whitespace, triggering the "Assistant" weights.

---

## 4. Strategic Roadmap: Path to 50+ Tokens/Sec

To transition from 12.5 t/s to production speeds, we must move past the "Memory Wall" using these three steps:

1.  **vLLM Implementation:** Use **PagedAttention** to manage that 15GB VRAM usage more efficiently, allowing for faster TTFT.
2.  **Quantization Migration:** Swap `bitsandbytes` (NF4) for **AWQ** or **GPTQ**. These use "fused kernels" that dequantize and compute math simultaneously.
3.  **Prompt Caching:** Since USMLE contexts are large (926 tokens), we should cache the prompt prefix so the GPU doesn't re-read the entire textbook for every turn in a conversation.

![alt text](gpu_metrics_modal_bitsandbytes.png)

![alt text](gpu_metrics_modal_bitsandbytes_2.png)