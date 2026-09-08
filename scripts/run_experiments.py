"""
Master experiment runner. Applies every core technique to both fine-tuned
baselines, independently, then combined

  1. Both baselines fully working and evaluated   (train_bert.py / train_vit.py)
  2. Shared evaluation utilities                   (src/evaluation/*)
  3. Quantization on both models
  4. Pruning on both models
  5. Knowledge distillation on both models
  6. Bonus low-rank experiment (BERT only)
  7. Combined optimizations (pruning + quantization)
  8. Visualizations / cleanup / report            (notebooks/final_experiments.ipynb)

Usage: python scripts/run_experiments.py --model bert   (or --model vit, --model all)
Requires the corresponding baseline checkpoint to already exist under
results/checkpoints/, produced by train_bert.py / train_vit.py.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModelForImageClassification, BertConfig, ViTConfig

from src.data.nlp_data import load_agnews_tokenized
from src.data.vision_data import load_cifar10_processed, vision_collate_fn
from src.optimizations.quantization import dynamic_quantize_int8
from src.optimizations.pruning import unstructured_magnitude_prune
from src.optimizations.distillation import DistillationConfig, train_distilled_student
from src.optimizations.low_rank import apply_low_rank_to_bert_ffn
from src.evaluation.benchmark import evaluate_classifier, benchmark_latency
from src.evaluation.metrics import (
    compute_classification_metrics, count_parameters, compute_sparsity,
    measure_serialized_size_mb, compression_ratio, performance_retention,
)
from src.evaluation.model_stats import set_all_seeds, log_experiment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def nlp_collate_fn(batch):
    return {
        "input_ids": torch.tensor(
            [x["input_ids"] for x in batch], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [x["attention_mask"] for x in batch], dtype=torch.long
        ),
        "labels": torch.tensor(
            [x["label"] for x in batch], dtype=torch.long
        ),
    }


def score(model, loader, input_keys, baseline_size_mb=None, baseline_acc=None, device=DEVICE):
    preds, labels = evaluate_classifier(model, loader, device, input_keys=input_keys)
    metrics = compute_classification_metrics(labels, preds)
    metrics.update(count_parameters(model))
    metrics["sparsity_pct"] = compute_sparsity(model)
    size_mb = measure_serialized_size_mb(model)
    metrics["size_mb"] = size_mb
    if baseline_size_mb:
        metrics["compression_ratio"] = compression_ratio(baseline_size_mb, size_mb)
    if baseline_acc:
        metrics["performance_retention_pct"] = performance_retention(baseline_acc, metrics["accuracy"])
    return metrics


def run_nlp_experiments(baseline_dir="results/checkpoints/bert_baseline"):
    set_all_seeds(42)
    data = load_agnews_tokenized()
    test_loader = DataLoader(data.test, batch_size=16, collate_fn=nlp_collate_fn)
    keys = ("input_ids", "attention_mask")

    baseline = AutoModelForSequenceClassification.from_pretrained(baseline_dir).to(DEVICE)
    baseline_metrics = score(baseline, test_loader, keys)
    log_experiment("bert_baseline_eval", {}, baseline_metrics)
    print("BERT baseline:", baseline_metrics)

    # ---  Quantization ---
    quantized = dynamic_quantize_int8(baseline)
    q_metrics = score(quantized, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"], device="cpu")
    log_experiment("bert_quantized", {"method": "dynamic_int8"}, q_metrics)
    print("BERT quantized:", q_metrics)

    # ---  Pruning (two sparsity levels) ---
    for amount in (0.3, 0.5):
        pruned = unstructured_magnitude_prune(baseline, amount=amount).to(DEVICE)
        p_metrics = score(pruned, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"])
        log_experiment(f"bert_pruned_{amount}", {"method": "unstructured_l1", "amount": amount}, p_metrics)
        print(f"BERT pruned ({amount}):", p_metrics)

    # ---  Knowledge Distillation ---
    student_config = BertConfig(
        vocab_size=baseline.config.vocab_size, hidden_size=384,
        num_hidden_layers=4, num_attention_heads=6, intermediate_size=1536,
        max_position_embeddings=baseline.config.max_position_embeddings,
        type_vocab_size=baseline.config.type_vocab_size, num_labels=4,
    )
    student = AutoModelForSequenceClassification.from_config(student_config)
    student, history = train_distilled_student(
        student, baseline, data.train, DistillationConfig(epochs=3),
        input_keys=keys, collate_fn=nlp_collate_fn
    )
    d_metrics = score(student, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"])
    log_experiment("bert_distilled", {"student_layers": 4, "student_hidden_size": 384,
                                       "student_heads": 6, "history": history}, d_metrics)
    print("BERT distilled:", d_metrics)

    # ---  Bonus low-rank factorization ---
    lr_model, replaced = apply_low_rank_to_bert_ffn(baseline, rank=128)
    lr_metrics = score(lr_model, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"])
    log_experiment("bert_low_rank", {"rank": 128, "layers_replaced": len(replaced)}, lr_metrics)
    print("BERT low-rank:", lr_metrics)

    # --- Combined (Pruning + Quantization) ---
    pruned_then_quantized = dynamic_quantize_int8(unstructured_magnitude_prune(baseline, amount=0.3))
    c_metrics = score(pruned_then_quantized, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"], device="cpu")
    log_experiment("bert_pruned_quantized", {"prune_amount": 0.3, "method": "dynamic_int8"}, c_metrics)
    print("BERT pruned+quantized:", c_metrics)


def run_vision_experiments(baseline_dir="results/checkpoints/vit_baseline"):
    set_all_seeds(42)
    data = load_cifar10_processed()
    test_loader = DataLoader(data.test, batch_size=32, collate_fn=vision_collate_fn)
    keys = ("pixel_values",)

    baseline = AutoModelForImageClassification.from_pretrained(baseline_dir).to(DEVICE)
    baseline_metrics = score(baseline, test_loader, keys)
    log_experiment("vit_baseline_eval", {}, baseline_metrics)
    print("ViT baseline:", baseline_metrics)

    quantized = dynamic_quantize_int8(baseline)
    q_metrics = score(quantized, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"], device="cpu")
    log_experiment("vit_quantized", {"method": "dynamic_int8"}, q_metrics)
    print("ViT quantized:", q_metrics)

    for amount in (0.3, 0.5):
        pruned = unstructured_magnitude_prune(baseline, amount=amount).to(DEVICE)
        p_metrics = score(pruned, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"])
        log_experiment(f"vit_pruned_{amount}", {"method": "unstructured_l1", "amount": amount}, p_metrics)
        print(f"ViT pruned ({amount}):", p_metrics)

    from transformers import ViTConfig
    student_config = ViTConfig(
        image_size=baseline.config.image_size, patch_size=baseline.config.patch_size,
        num_channels=baseline.config.num_channels, hidden_size=384,
        num_hidden_layers=4, num_attention_heads=6, intermediate_size=1536,
        num_labels=10, id2label=baseline.config.id2label, label2id=baseline.config.label2id,
    )
    student = AutoModelForImageClassification.from_config(student_config)
    student, history = train_distilled_student(
        student, baseline, data.train, DistillationConfig(epochs=3),
        input_keys=keys, collate_fn=vision_collate_fn
    )
    d_metrics = score(student, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"])
    log_experiment("vit_distilled", {"student_layers": 4, "student_hidden_size": 384,
                                      "student_heads": 6, "history": history}, d_metrics)
    print("ViT distilled:", d_metrics)

    pruned_then_quantized = dynamic_quantize_int8(unstructured_magnitude_prune(baseline, amount=0.3))
    c_metrics = score(pruned_then_quantized, test_loader, keys, baseline_metrics["size_mb"], baseline_metrics["accuracy"], device="cpu")
    log_experiment("vit_pruned_quantized", {"prune_amount": 0.3, "method": "dynamic_int8"}, c_metrics)
    print("ViT pruned+quantized:", c_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "vit", "all"], default="all")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.model in ("bert", "all"):
        run_nlp_experiments()
    if args.model in ("vit", "all"):
        run_vision_experiments()
