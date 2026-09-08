"""
Data preparation for the vision branch of the project.

Model:   ViT Base Patch16 224 (google/vit-base-patch16-224)
Task:    Multi-class image classification
Dataset: CIFAR-10 (10 classes)
"""
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor

CHECKPOINT = "google/vit-base-patch16-224"
NUM_LABELS = 10
LABEL_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class VisionData:
    train: object
    val: object
    test: object
    image_processor: object


def load_cifar10_processed(
    checkpoint: str = CHECKPOINT,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> VisionData:
    """Download CIFAR-10 and resize/normalize images for ViT's 224x224 input."""
    raw = load_dataset("uoft-cs/cifar10")
    split = raw["train"].train_test_split(
    test_size=val_fraction,
    seed=seed
)
    train_raw, val_raw, test_raw = split["train"], split["test"], raw["test"]

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["img"]]
        processed = image_processor(images, return_tensors="pt")
        batch["pixel_values"] = processed["pixel_values"]
        batch["label"] = batch["label"]
        return batch

    train = train_raw.with_transform(transform)
    val = val_raw.with_transform(transform)
    test = test_raw.with_transform(transform)

    return VisionData(train=train, val=val, test=test, image_processor=image_processor)


def vision_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == "__main__":
    data = load_cifar10_processed()
    print(f"train={len(data.train)} val={len(data.val)} test={len(data.test)}")
