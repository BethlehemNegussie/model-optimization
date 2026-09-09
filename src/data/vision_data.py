"""
Data preparation for the vision branch of the project.

Model:   ViT Base Patch16 224 (google/vit-base-patch16-224)
Task:    Multi-class image classification
Dataset: CIFAR-10 (10 classes)
"""
from dataclasses import dataclass

import torch
from datasets import load_dataset #ds from HF datasets lib
from transformers import AutoImageProcessor #pretrained ViT expects images in a particular format , 
#which is handled by AutoImageProcessor  

CHECKPOINT = "google/vit-base-patch16-224" # ViT receives an image of  224 * 224 px and divides it into patches of 16 * 16
NUM_LABELS = 10 #so there are 224/16 (14 * 14 img patches), ViT treats those patches somewhat like tokens in NLP
LABEL_NAMES = [ #human-readable mapping of the 10 labels
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass #makes it convenient to create a class that mainly holds data
class VisionData: #creates a little container for our data
    train: object
    val: object
    test: object
    image_processor: object


def load_cifar10_processed( #load CIFAR-10 and process it for the ViT
    checkpoint: str = CHECKPOINT,
    val_fraction: float = 0.1,#10% of the training data to become validation data
    seed: int = 42, #same seed to reproduce the same train/validation split  
) -> VisionData: #function returns a VisionData object.
    """Download CIFAR-10 and resize/normalize images for ViT's 224x224 input."""
    raw = load_dataset("uoft-cs/cifar10") #downloads/loads CIFAR-10 from HF
    split = raw["train"].train_test_split( #takes the original training set and splits it into   train/ valid set
    test_size=val_fraction,
    seed=seed
)
    train_raw, val_raw, test_raw = split["train"], split["test"], raw["test"] #gives us three datasets
    #Get the image preprocessing config associated with this pretrained ViT
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    
    def transform(batch): #What should happen to images before they are given to ViT?
        images = [img.convert("RGB") for img in batch["img"]]
        processed = image_processor(images, return_tensors="pt") #processor takes the images and prepares them for PyTorch/ViT, returns the processed data as a tensor
        batch["pixel_values"] = processed["pixel_values"] #what the ViT will actually receive as its image input
        batch["label"] = batch["label"]
        return batch
    #with_transform() so the preprocessing is applied when the data is accessed, rather than permanently rewriting the entire dataset,
    # not to create another giant processed copy of the imgs, on-the-fly preprocessing
    train = train_raw.with_transform(transform)
    val = val_raw.with_transform(transform)
    test = test_raw.with_transform(transform)
    #return everything packaged into our VisionData dataclass.
    return VisionData(train=train, val=val, test=test, image_processor=image_processor)

# fn to take individual examples and assembling them into a batch.
def vision_collate_fn(batch):
    #Take the pixel_values from every example and stack them together
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch]) #Make labels into a tensor too
    return {"pixel_values": pixel_values, "labels": labels}

#Only execute the following code if this file is run directly
if __name__ == "__main__":
    data = load_cifar10_processed()
    print(f"train={len(data.train)} val={len(data.val)} test={len(data.test)}")
