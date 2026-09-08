# Model Optimization Across Transformer Architectures

Cross-domain investigation of model optimization techniques on two
Transformer-based architectures: **BERT-base-uncased** (NLP, AG News text
classification) and **ViT-Base-Patch16-224** (vision, CIFAR-10 image
classification).

**Research question:** How do model optimization techniques behave across
Transformer architectures in NLP and computer vision, and does the best
optimization strategy depend on the architecture and domain?

## What's implemented

| Technique | Module | Applied to |
|---|---|---|
| Quantization (PTQ dynamic, PTQ static + calibration, QAT scaffold) | `src/optimizations/quantization.py` | BERT, ViT |
| Pruning (unstructured L1 magnitude, structured channel) | `src/optimizations/pruning.py` | BERT, ViT |
| Knowledge Distillation (soft + hard loss, temperature-scaled) | `src/optimizations/distillation.py` | BERT, ViT |
| Low-Rank Factorization (SVD, bonus technique) | `src/optimizations/low_rank.py` | BERT (FFN layers) |
| Combined: Pruning + Quantization | `scripts/run_experiments.py` | BERT, ViT |

All techniques are scored with the same shared utilities
(`src/evaluation/metrics.py`, `src/evaluation/benchmark.py`) so before/after
and cross-technique comparisons are fair: accuracy, macro-F1, real serialized
size (MB), total vs. non-zero parameters, sparsity %, compression ratio,
performance retention %, and warmed-up/repeated-run latency + throughput.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires internet access to Hugging Face Hub (to download
`google-bert/bert-base-uncased`, `google/vit-base-patch16-224`, `ag_news`,
`cifar10`) and, for practical training times, a CUDA-capable GPU.

## Reproduction

```bash
# 1. Fine-tune both task-specific baselines (Priority 1)
python scripts/train_bert.py --epochs 3 --batch_size 16
python scripts/train_vit.py  --epochs 5 --batch_size 32

# 2. Run every optimization technique against each baseline (Priorities 3-7)
python scripts/run_experiments.py --model bert
python scripts/run_experiments.py --model vit

# 3. Open the notebook for tables, figures, and discussion (Priority 8)
jupyter lab notebooks/final_experiments.ipynb
```

Results land in `results/metrics.csv` (one row per experiment, appended by
`src/evaluation/model_stats.log_experiment`), `results/<experiment>.json`
(full per-experiment detail), and `results/figures/`.

## Repository structure

```
model-optimization/
├── notebooks/
│   └── final_experiments.ipynb   # experiment driver, tables, figures, discussion
├── src/
│   ├── data/          # nlp_data.py (AG News/BERT), vision_data.py (CIFAR-10/ViT)
│   ├── models/        # bert.py, vit.py — baseline construction
│   ├── optimizations/ # quantization.py, pruning.py, distillation.py, low_rank.py
│   └── evaluation/    # metrics.py, benchmark.py, model_stats.py (shared, fair scoring)
├── scripts/
│   ├── train_bert.py       # Stage 1: BERT baseline fine-tuning
│   ├── train_vit.py        # Stage 1: ViT baseline fine-tuning
│   └── run_experiments.py  # Stage 2+: applies every technique, priority-ordered
├── results/
│   ├── metrics.csv
│   └── figures/
├── requirements.txt
└── README.md
```

## Experimental fairness

- Same test split used for baseline and every optimized variant.
- Same tokenizer/image-processor preprocessing throughout.
- Latency benchmarks warm up (discarded runs) before timing, then average
  over repeated runs (`src/evaluation/benchmark.py`).
- Random seeds fixed and recorded (`src/evaluation/model_stats.set_all_seeds`).
- Exact framework/library versions and hardware logged with every experiment
  (`src/evaluation/model_stats.record_environment`).
- All optimization hyperparameters (quantization backend, pruning
  amount/criterion, distillation temperature/alpha, factorization rank)
  are passed explicitly and logged, never hardcoded silently.

## Known limitations (documented, not hidden)

- Unstructured pruning here is mask-based (zeros in a dense tensor); it
  raises `sparsity_pct` but does **not** shrink `size_mb` or latency on
  standard hardware — that's the expected, documented behavior, not a bug
  (see the "Structured vs. Unstructured" trade-off in `pruning.py`'s
  docstring). Structured pruning is implemented separately for the cases
  where a real speedup is the goal.
- PyTorch's native dynamic INT8 quantization runs on **CPU only**; a
  GPU-accelerated INT4 path via `bitsandbytes` is documented in
  `quantization.py` as `quantize_bitsandbytes_4bit` but not exercised here.
- Knowledge distillation uses a fixed 4-layer student architecture as a
  first pass rather than a searched/tuned student size.
