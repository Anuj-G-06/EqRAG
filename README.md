# EqRAG — LLM Fine-Tuning for Financial Stock Prediction

> Can LLMs predict stock price movements from news headlines and financial ratios?

This project explores applying Large Language Models to **fundamental stock analysis and price movement prediction** on Dow 30 stocks. We compare In-Context Learning (ICL) against Parameter-Efficient Fine-Tuning (PEFT) methods — LoRA and IA3 — on math-specialized 7B models.

**Authors:** [Anuj Gupta](mailto:anujg2@andrew.cmu.edu), [Aditya Motwani](mailto:amotwani@andrew.cmu.edu) — Carnegie Mellon University

---

## Results at a Glance

```
┌─────────────────────────────────┬──────────┬──────────┐
│ Configuration                   │ Accuracy │ F1 Macro │
├─────────────────────────────────┼──────────┼──────────┤
│ DeepSeek-Math-7B (0-shot ICL)   │  12.0%   │  0.299   │
│ Qwen-2.5-Math-7B (0-shot ICL)  │  12.3%   │  0.254   │
│ Qwen 1.5B (Full Fine-Tune)     │  46.7%   │  0.483   │
│ DeepSeek-Math-7B (LoRA r=16)   │  43.0%   │  0.510   │
│ Qwen-2.5-Math-7B (IA3)         │  12.0%   │  0.286   │
│ Qwen-2.5-Math-7B (LoRA r=8)    │  48.7%   │  0.489   │
│ ★ Qwen-2.5-Math-7B (LoRA r=16) │  51.7%   │  0.525   │
│ Qwen-2.5-Math-7B (LoRA r=32)   │  49.7%   │  0.518   │
│ Qwen-2.5-Math-7B (LoRA r=64)   │  46.7%   │  0.507   │
└─────────────────────────────────┴──────────┴──────────┘
```

> In financial forecasting, accuracy above 50% is statistically significant due to market stochasticity. LoRA fine-tuning improved accuracy from ~12% → **51.7%**.

---

## Task

Given a company profile, recent news headlines, and financial metrics (P/E, ROE, Asset Turnover, etc.), predict the stock's movement for the upcoming week:

```
Input:  Company intro + News summaries + Financial ratios
Output: [Positive Developments] → [Potential Concerns] → [Prediction & Analysis] → [Action]: BUY / SELL / HOLD
```

**Dataset:** [FinGPT/fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) — 1,230 train / 300 test examples covering Dow 30 components over one year.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   FinGPT Dataset    │
                    │  (Dow 30, 1 year)   │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │   dataset.py        │
                    │  • Add [Action]     │
                    │  • Extract % pred   │
                    │  • Tokenize         │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  icl.py    │  │  train.py  │  │  train.py  │
     │  0-shot    │  │  LoRA      │  │  IA3 / FFT │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           └───────┬───────┴───────┬───────┘
                   ▼               ▼
          ┌──────────────┐  ┌────────────┐
          │ inference.py │  │  Metrics   │
          │ Batch eval   │  │ Acc, F1    │
          └──────────────┘  └────────────┘
```

---

## Key Findings

1. **ICL fails completely** (~12% accuracy) — models cannot perform structured financial reasoning zero-shot, often hallucinating output formats
2. **LoRA fine-tuning** transforms performance: 12% → **51.7%** with just 1 epoch on 1,230 examples
3. **Math-specialized base models** (Qwen-2.5-Math) outperform instruction-tuned variants (DeepSeek-Math-Instruct) — non-instruction-tuned models adapt better to domain-specific formats
4. **LoRA rank r=16** is optimal — higher ranks (32, 64) degrade performance due to overfitting on limited data
5. **IA3 fails entirely** (12% accuracy) — its lightweight vector rescaling cannot handle the domain shift from math to finance

---

## Quick Start

### Setup

```bash
# Clone the repo
git clone https://github.com/Anuj-G-06/EqRAG.git
cd EqRAG

# Install dependencies
pip install -r requirements.txt
```

### Training (LoRA Fine-Tuning)

```bash
# Best configuration: Qwen-2.5-Math-7B with LoRA (r=16)
python train.py configs/finsight_math_lora_tm.json
```

### In-Context Learning (0-shot Baseline)

```bash
python icl.py configs/qwen_icl_0.json
```

### Inference & Evaluation

```bash
python inference.py configs/finsight_math_lora_tm.json
```

Results are saved to `outputs/` as JSON with per-example predictions and aggregate metrics.

---

## Experiment Configs

All experiment configurations are in `configs/`:

| Config File | Model | Method | LoRA Rank |
|---|---|---|---|
| `qwen_icl_0.json` | Qwen-2.5-Math-7B | 0-shot ICL | — |
| `ds_icl_0.json` | DeepSeek-Math-7B | 0-shot ICL | — |
| `finsight_math_lora_tm.json` | Qwen-2.5-Math-7B | LoRA | 16 |
| `finsight_math_lora_tm-r8.json` | Qwen-2.5-Math-7B | LoRA | 8 |
| `finsight_math_lora_tm-r32.json` | Qwen-2.5-Math-7B | LoRA | 32 |
| `finsight_math_lora_tm-r64.json` | Qwen-2.5-Math-7B | LoRA | 64 |
| `finsight_math_ia3_tm.json` | Qwen-2.5-Math-7B | IA3 | — |
| `finsight_math_fft.json` | Qwen-2.5-Math-1.5B | Full Fine-Tune | — |
| `finsight_dsmi_lora_tm.json` | DeepSeek-Math-7B | LoRA | 16 |
| `finsight_dsmi_ia3_tm.json` | DeepSeek-Math-7B | IA3 | — |

---

## Project Structure

```
EqRAG/
├── train.py              # Fine-tuning script (LoRA, IA3, FFT)
├── icl.py                # In-context learning inference
├── inference.py          # Batch inference & evaluation
├── dataset.py            # Dataset loading, action labeling, tokenization
├── requirements.txt      # Python dependencies
├── configs/              # Experiment configurations (JSON)
│   ├── finsight_math_lora_tm.json    # ★ Best config
│   ├── qwen_icl_0.json              # ICL baseline
│   └── ...
├── outputs/              # Inference results (JSON)
├── Report.pdf            # Full research report
└── EqRAG.pptx            # Presentation slides
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-Math-7B |
| Learning Rate | 3e-4 |
| Batch Size | 4 |
| Epochs | 1 |
| Sequence Length | 2048 |
| Precision | bf16 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| LR Scheduler | Cosine |
| Optimizer | AdamW |

---

## References

- Hu, E. J. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv:2106.09685.
- Liu, H. et al. (2022). [Few-shot PEFT is Better and Cheaper than ICL](https://arxiv.org/abs/2205.05638). arXiv:2205.05638.
- Tang, Z. et al. (2025). [FinanceReasoning: Benchmarking Financial Numerical Reasoning](https://arxiv.org/abs/2506.05828). arXiv:2506.05828.
- Wei, J. et al. (2023). [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903).
- Lopez-Lira, A. et al. (2025). [Can We Trust LLMs' Economic Forecasts?](https://arxiv.org/abs/2504.14765). arXiv:2504.14765.
