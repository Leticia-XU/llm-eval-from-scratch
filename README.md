
# LLM Evaluation from Scratch on T4 (vLLM + GSM8K)

## 1. Project Goal

Build an end-to-end LLM inference and evaluation pipeline from scratch:

- Deploy a 7B model with vLLM
- Run OpenAI-compatible API server
- Evaluate on GSM8K
- Measure accuracy and latency

---

## 2. Environment

- Ubuntu 24.04
- NVIDIA Tesla T4 (16GB)
- Driver 590
- CUDA 13.1
- Python 3.10

---

## 3. Model Deployment

Model: Qwen2-7B-Instruct  
Framework: vLLM  
Quantization: 4-bit  
Attention backend: TRITON_ATTN  

Key concepts learned:

- KV Cache
- PagedAttention
- Autotuning
- CUDA memory management

---

## 4. Evaluation Setup

Dataset: GSM8K (Grade School Math 8K)

- Test subset: 50 samples
- Temperature: 0.0
- Metric: Accuracy

Result:

Accuracy: 0.68  
Average latency: ~8.9s per question

---

## 5. What I Learned

- GPU memory analysis with nvidia-smi
- Dataset loading and split slicing
- tqdm progress interpretation
- Deterministic decoding for evaluation
- Handling CUDA backend issues

---

## 6. Engineering Challenges

- CUDA OOM during vLLM warmup
  → Solved by using AWQ quantized model

- FlashInfer required nvcc
  → Switched attention backend to TRITON_ATTN

- Conda TOS issue
  → Accepted channel terms manually

---

## 7. Next Steps

- Full 1319 test evaluation
- Few-shot prompting
- Compare multiple models
- Add latency profiling
