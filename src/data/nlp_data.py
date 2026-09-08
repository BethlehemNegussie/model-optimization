"""
Data preparation for the NLP branch of the project.

Model:   BERT-base-uncased (google-bert/bert-base-uncased)
Task:    Multi-class text classification
Dataset: AG News (4 classes: World, Sports, Business, Sci/Tech)
"""
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoTokenizer

CHECKPOINT = "google-bert/bert-base-uncased"
NUM_LABELS = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAX_LENGTH = 128


@dataclass
class NLPData:
    train: object
    val: object
    test: object
    tokenizer: object


def load_agnews_tokenized(
    checkpoint: str = CHECKPOINT,
    max_length: int = MAX_LENGTH,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> NLPData:
    """Download AG News, tokenize it for BERT, and carve out a val split.

    The official AG News split only ships train/test, so we hold out
    `val_fraction` of train (stratified by label via `datasets`'
    train_test_split, which shuffles with a fixed seed for reproducibility)
    to use for model selection, keeping test fully held out for final
    reporting.
    """
    raw = load_dataset("fancyzhx/ag_news")
    split = raw["train"].train_test_split(
        test_size=val_fraction, seed=seed, stratify_by_column="label"
    )
    train_raw, val_raw, test_raw = split["train"], split["test"], raw["test"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    cols_to_remove = ["text"]
    train = train_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)
    val = val_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)
    test = test_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)

    return NLPData(train=train, val=val, test=test, tokenizer=tokenizer)


def get_representative_calibration_batch(nlp_data: NLPData, n: int = 256):
    """Small held-out subset used for PTQ calibration."""
    return nlp_data.train.select(range(min(n, len(nlp_data.train))))


if __name__ == "__main__":
    data = load_agnews_tokenized()
    print(f"train={len(data.train)} val={len(data.val)} test={len(data.test)}")
