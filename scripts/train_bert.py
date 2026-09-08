"""Stage 1 fine-tune BERT-base-uncased on AG News and save the task-specific
baseline checkpoint. Run from the repo root: `python scripts/train_bert.py`.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.nlp_data import load_agnews_tokenized
from src.models.bert import build_bert_classifier, record_pretrained_checkpoint_info
from src.evaluation.benchmark import evaluate_classifier
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.model_stats import set_all_seeds, log_experiment


def main(args):
    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Stage 0: recording pretrained checkpoint info...")
    checkpoint_info = record_pretrained_checkpoint_info()
    print(checkpoint_info)

    print("Loading + tokenizing AG News...")
    data = load_agnews_tokenized()

    def collate(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }

    train_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(data.val, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(data.test, batch_size=args.batch_size, collate_fn=collate)

    model = build_bert_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"BERT epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds, labels = evaluate_classifier(model, val_loader, device)
        val_metrics = compute_classification_metrics(labels, preds)
        print(f"epoch {epoch + 1} val: {val_metrics}")

    preds, labels = evaluate_classifier(model, test_loader, device)
    test_metrics = compute_classification_metrics(labels, preds)
    print(f"BERT baseline test: {test_metrics}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    data.tokenizer.save_pretrained(args.output_dir)

    log_experiment(
        name="bert_baseline",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, **checkpoint_info},
        results=test_metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/checkpoints/bert_baseline")
    main(parser.parse_args())
