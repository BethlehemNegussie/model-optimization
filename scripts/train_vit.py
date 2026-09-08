"""Stage 1 fine-tune ViT-Base-Patch16-224 on CIFAR-10 and save the
task-specific baseline checkpoint. Run from the repo root:
`python scripts/train_vit.py`.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.vision_data import load_cifar10_processed, vision_collate_fn
from src.models.vit import build_vit_classifier, record_pretrained_checkpoint_info
from src.evaluation.benchmark import evaluate_classifier
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.model_stats import set_all_seeds, log_experiment


def main(args):
    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Stage 0: recording pretrained checkpoint info...")
    checkpoint_info = record_pretrained_checkpoint_info()
    print(checkpoint_info)

    print("Loading + preprocessing CIFAR-10...")
    data = load_cifar10_processed()

    train_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=True, collate_fn=vision_collate_fn)
    val_loader = DataLoader(data.val, batch_size=args.batch_size, collate_fn=vision_collate_fn)
    test_loader = DataLoader(data.test, batch_size=args.batch_size, collate_fn=vision_collate_fn)

    model = build_vit_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"ViT epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds, labels = evaluate_classifier(model, val_loader, device, input_keys=("pixel_values",))
        val_metrics = compute_classification_metrics(labels, preds)
        print(f"epoch {epoch + 1} val: {val_metrics}")

    preds, labels = evaluate_classifier(model, test_loader, device, input_keys=("pixel_values",))
    test_metrics = compute_classification_metrics(labels, preds)
    print(f"ViT baseline test: {test_metrics}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    data.image_processor.save_pretrained(args.output_dir)

    log_experiment(
        name="vit_baseline",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, **checkpoint_info},
        results=test_metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/checkpoints/vit_baseline")
    main(parser.parse_args())
