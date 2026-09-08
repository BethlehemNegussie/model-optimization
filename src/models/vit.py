"""ViT baseline: load pretrained checkpoint, adapt classification head for CIFAR-10."""
from transformers import AutoModelForImageClassification, AutoConfig

from src.data.vision_data import CHECKPOINT, NUM_LABELS, LABEL_NAMES


def build_vit_classifier(checkpoint: str = CHECKPOINT, num_labels: int = NUM_LABELS):
    """Stage 0 instantiate ViT with the head swapped for CIFAR-10's 10 classes.

    ignore_mismatched_sizes=True is required because the pretrained checkpoint's
    head is sized for ImageNet-1k (1000 classes), not CIFAR-10 (10 classes).
    """
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def record_pretrained_checkpoint_info(checkpoint: str = CHECKPOINT) -> dict:
    """Stage 0 record info about the original pretrained checkpoint."""
    config = AutoConfig.from_pretrained(checkpoint)
    n_params = sum(p.numel() for p in build_vit_classifier(checkpoint).parameters())
    return {
        "checkpoint": checkpoint,
        "architecture": config.architectures[0] if config.architectures else "ViTForImageClassification",
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "patch_size": config.patch_size,
        "image_size": config.image_size,
        "total_parameters": n_params,
    }
