# Sample Benchmark Results

Pre-computed benchmark results from an NVIDIA RTX 5090 (sm_120, 24GB VRAM).

These results power the Tradeoff Dashboard without requiring you to run benchmarks yourself.
Load the dashboard with `streamlit run dashboard/app.py` to explore them interactively.

## Models Benchmarked

### Qwen/Qwen2.5-7B-Instruct (7B parameters)

| Method | Size | Perplexity | TPS | Quality % | Fits RTX 5090 |
|--------|------|-----------|-----|-----------|--------------|
| BF16 | 14.5 GB | 8.42 | 178 | 100% | ✓ |
| INT8 | 7.3 GB | 8.71 | 161 | 97% | ✓ |
| GPTQ 4-bit | 4.1 GB | 8.87 | 148 | 95% | ✓ |
| GGUF Q4_K_M | 4.7 GB | 9.04 | 135 | 92% | ✓ |
| INT4 NF4 | 4.2 GB | 9.23 | 157 | 90% | ✓ |

**Recommended:** GPTQ 4-bit — best quality/size ratio (95% quality at 3.5x compression)

### microsoft/Phi-4-mini-instruct (3.8B parameters)

| Method | Size | Perplexity | TPS | Quality % | Fits RTX 5090 |
|--------|------|-----------|-----|-----------|--------------|
| BF16 | 7.6 GB | 7.83 | 242 | 100% | ✓ |
| INT4 NF4 | 2.2 GB | 8.41 | 228 | 95% | ✓ |

**Key insight:** Phi-4-Mini INT4 NF4 at 2.2GB fits on Jetson Nano (4GB) — enabling
state-of-the-art reasoning at the edge with zero cloud dependency.

## How to Add Your Own Results

Run benchmarks with the CLI:
```bash
edgemind benchmark Qwen/Qwen2.5-7B-Instruct --method bf16
edgemind benchmark ./quantized/qwen-7b-int4 --method int4_nf4
```

Results are saved to `./results/` and automatically appear in the dashboard.
