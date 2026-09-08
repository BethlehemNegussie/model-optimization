
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class DistillationConfig:
    temperature: float = 4.0   # tau: softens the teacher's distribution
    alpha: float = 0.5         # weight on hard-label loss vs soft-target loss
    lr: float = 5e-5
    epochs: int = 3
    batch_size: int = 32


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, alpha: float):
    """L_student = alpha * L_hard + (1 - alpha) * L_soft.

    L_hard: standard cross-entropy against the ground-truth label.
    L_soft: KL divergence between temperature-softened student and teacher
    distributions, scaled by T^2 to keep gradient magnitudes
    comparable across temperature settings.
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss, hard_loss.item(), soft_loss.item()


def train_distilled_student(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    train_dataset,
    config: DistillationConfig,
    input_keys=("input_ids", "attention_mask"),
    collate_fn=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Teacher-guided distillation training loop.

    The teacher is frozen and run in eval/no_grad mode throughout -- only the
    student's parameters are updated. Works for either the NLP (BERT teacher
    -> smaller BERT/DistilBERT-style student) or vision (ViT teacher ->
    smaller ViT student) branch, since both consume the same {inputs, label}
    batch shape and produce `.logits`.
    """
    teacher.eval().to(device)
    student.train().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    history = []
    for epoch in range(config.epochs):
        epoch_loss, epoch_hard, epoch_soft, n_batches = 0.0, 0.0, 0.0, 0
        for batch in tqdm(loader, desc=f"distill epoch {epoch + 1}/{config.epochs}"):
            inputs = {k: batch[k].to(device) for k in input_keys if k in batch}
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            with torch.no_grad():
                teacher_logits = teacher(**inputs).logits

            student_logits = student(**inputs).logits
            loss, hard, soft = distillation_loss(
                student_logits, teacher_logits, labels, config.temperature, config.alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_hard += hard
            epoch_soft += soft
            n_batches += 1

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss / n_batches,
            "hard_loss": epoch_hard / n_batches,
            "soft_loss": epoch_soft / n_batches,
        })
        print(history[-1])

    return student, history
