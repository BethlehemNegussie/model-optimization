# Model Optimization Across Transformer Architectures

Cross-domain investigation of model optimization techniques on two Transformer-based architectures:

- **BERT-base-uncased** for NLP text classification on **AG News**
- **ViT-Base-Patch16-224** for image classification on **CIFAR-10**

## Research Question

> How do model optimization techniques behave across Transformer architectures in NLP and computer vision, and does the best optimization strategy depend on the architecture and domain?

The project evaluates model compression and optimization techniques using accuracy, macro-F1, model size, parameter sparsity, compression ratio, and performance retention.

---

## Project Status

The project is being completed in two experimental stages.

### BERT — Completed

The BERT baseline and the main optimization experiments have been successfully executed.

Completed experiments:

- Baseline BERT
- Dynamic INT8 quantization
- 30% unstructured L1 pruning
- 50% unstructured L1 pruning
- Knowledge distillation
- Low-rank SVD factorization

The resulting metrics are stored in:

```text
results/metrics.csv
results/bert_*.json
```

### ViT — In Progress

The ViT implementation is complete, but the full ViT experimental evaluation is still being completed.

The remaining work is to:

1. Fine-tune the ViT baseline on CIFAR-10.
2. Run the optimization techniques against the trained ViT baseline.
3. Record the resulting metrics.
4. Compare BERT and ViT results across domains.
5. Update the final notebook and discussion.

The ViT experiments are being run in a GPU environment such as Kaggle because GPU availability in the original Google Colab environment became limited.

---

## What's Implemented

| Technique | Module | Architecture |
|---|---|---|
| Dynamic INT8 quantization | `src/optimizations/quantization.py` | BERT, ViT |
| Static INT8 quantization + calibration | `src/optimizations/quantization.py` | BERT, ViT |
| QAT scaffold | `src/optimizations/quantization.py` | BERT, ViT |
| Unstructured L1 magnitude pruning | `src/optimizations/pruning.py` | BERT, ViT |
| Structured channel pruning | `src/optimizations/pruning.py` | BERT, ViT |
| Knowledge distillation | `src/optimizations/distillation.py` | BERT, ViT |
| Low-rank SVD factorization | `src/optimizations/low_rank.py` | BERT |
| Combined pruning + quantization | `scripts/run_experiments.py` | BERT, ViT |

**Important:** Implementation does not necessarily mean that every technique has already been experimentally evaluated. The completed experiments are those represented in `results/metrics.csv`.

---

## Evaluation Metrics

The project uses shared evaluation utilities in:

```text
src/evaluation/
```

The experiments evaluate:

- Accuracy
- Macro-F1
- Serialized model size (MB)
- Total parameters
- Non-zero parameters
- Parameter sparsity (%)
- Compression ratio
- Performance retention (%)

Latency and throughput benchmarking utilities are also implemented in:

```text
src/evaluation/benchmark.py
```

However, the current experiment runner does **not** record latency and throughput in `results/metrics.csv`. Therefore, latency/throughput results are not presented as completed experimental results unless they are explicitly benchmarked and recorded later.

---

## BERT Results

The BERT experiments were conducted using:

- **Model:** BERT-base-uncased
- **Dataset:** AG News
- **Task:** 4-class text classification
- **Baseline training:** 3 epochs

### Completed Results

| Method | Accuracy | Macro-F1 | Size (MB) | Sparsity | Compression | Retention |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 94.87% | 94.87% | 417.73 | 0.00% | 1.00× | 100.00% |
| Dynamic INT8 | 93.88% | 93.87% | 173.09 | 0.00% | 2.41× | 98.96% |
| 30% Pruning | 94.84% | 94.84% | 417.73 | 23.44% | 1.00× | 99.97% |
| 50% Pruning | 94.41% | 94.41% | 417.73 | 39.06% | 1.00× | 99.51% |
| Distillation | 91.93% | 91.92% | 73.14 | ~0.00% | 5.71× | 96.91% |
| Low-Rank SVD | 61.51% | 60.13% | 363.75 | 0.00% | 1.15× | 64.84% |

### BERT Observations

- **Dynamic INT8 quantization** reduced serialized model size substantially while retaining approximately 99% of baseline performance.
- **30% pruning** produced almost no accuracy degradation, although the dense serialized model size remained unchanged because the pruning is mask-based.
- **50% pruning** produced greater sparsity with only a small reduction in accuracy.
- **Knowledge distillation** achieved the largest model-size reduction among the completed BERT experiments, with approximately 5.71× compression.
- **Low-rank SVD with the tested rank configuration** caused a substantial accuracy drop and provided only modest size reduction. This demonstrates that low-rank factorization is sensitive to the chosen factorization configuration.

---

## ViT Experiments

The vision component uses:

- **Model:** ViT-Base-Patch16-224
- **Dataset:** CIFAR-10
- **Task:** 10-class image classification

The pretrained ImageNet ViT checkpoint is adapted to the 10-class CIFAR-10 classification task.

The planned ViT evaluation mirrors the BERT evaluation where technically appropriate:

1. Train the ViT baseline.
2. Apply quantization.
3. Apply unstructured pruning.
4. Apply structured pruning where supported.
5. Apply knowledge distillation.
6. Evaluate combined optimization techniques where supported.
7. Record accuracy, macro-F1, size, sparsity, compression, and performance retention.

The final cross-domain analysis will compare whether the effectiveness of each optimization strategy changes between NLP and computer vision Transformer architectures.

---

## Reproduction

### 1. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project requires internet access to download models and datasets from Hugging Face.

A CUDA-capable GPU is recommended for practical training times.

---

### 2. Train the BERT Baseline

```bash
python scripts/train_bert.py --epochs 3 --batch_size 16
```

### 3. Train the ViT Baseline

```bash
python scripts/train_vit.py --epochs 5 --batch_size 32
```

For GPU-based execution, Kaggle or another CUDA-enabled environment can be used.

### 4. Run BERT Experiments

```bash
python scripts/run_experiments.py --model bert
```

### 5. Run ViT Experiments

```bash
python scripts/run_experiments.py --model vit
```

The ViT pipeline should be run after the ViT baseline has been successfully trained and the environment-specific dataset loading requirements have been satisfied.

### 6. Open the Analysis Notebook

```bash
jupyter lab notebooks/final_experiments.ipynb
```

The notebook is intended for result tables, visualization, cross-technique comparisons, cross-domain analysis, and final discussion.

---

## Repository Structure

```text
model-optimization/
├── notebooks/
│   └── final_experiments.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── optimizations/
│   └── evaluation/
├── scripts/
│   ├── train_bert.py
│   ├── train_vit.py
│   └── run_experiments.py
├── results/
│   ├── metrics.csv
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Experimental Fairness

To make comparisons as consistent as possible:

- The same test split is used for a baseline and its optimized variants.
- The same tokenizer/image preprocessing is used within each task.
- Random seeds are fixed where applicable.
- Optimization hyperparameters are explicitly passed to the relevant functions.
- Evaluation uses shared metric utilities.
- Baseline and optimized models are evaluated using the same task-specific evaluation procedure.

Hardware and software environment information can also be recorded through the utilities in `src/evaluation/model_stats.py`.

---

## Known Limitations

### Unstructured Pruning

Unstructured pruning creates zero-valued weights in otherwise dense tensors.

Therefore, although sparsity increases, the serialized model size and inference latency may remain approximately unchanged on standard dense hardware.

This is expected behavior rather than a bug.

Structured pruning is implemented separately for cases where actual architectural reduction or hardware speedup is desired.

### Dynamic INT8 Quantization

PyTorch's native dynamic INT8 quantization is primarily intended for CPU execution.

The repository also documents a GPU-oriented 4-bit `bitsandbytes` path, but that path is not treated as part of the completed results unless explicitly evaluated.

### Knowledge Distillation

The current distillation experiments use a fixed smaller student architecture rather than searching over multiple student configurations.

### Low-Rank Factorization

Low-rank factorization is sensitive to the selected rank and the layers being factorized. The completed BERT experiment demonstrates that an unsuitable rank/configuration can cause substantial performance degradation.

### GPU Availability

Training Transformer models can require substantial GPU resources. The experiments may therefore be executed in different GPU environments, such as Google Colab or Kaggle.

---

## Checkpoints

Large trained model checkpoints are intentionally excluded from the Git repository because standard GitHub repositories have a 100 MB per-file limit.

The repository contains the code required to reproduce the models and the recorded experimental results.

If trained checkpoints are required, they should be distributed separately through an appropriate model/file hosting service or Git LFS.

---

## Current Goal

The remaining experimental objective is to complete the ViT evaluation and use the resulting measurements to answer the central research question:

> **Does the most effective model optimization strategy depend on the Transformer architecture and application domain?**

The final analysis will compare the BERT and ViT results and identify the trade-offs between model size, sparsity, compression, and predictive performance.
