"""
Data preparation for the NLP branch of the project.

Model:   BERT-base-uncased (google-bert/bert-base-uncased)
Task:    Multi-class text classification
Dataset: AG News (4 classes: World, Sports, Business, Sci/Tech)
"""
from dataclasses import dataclass
#used later to create a simple container class
from datasets import load_dataset #HuggingFace's datasets library function for downloading/loading datasets
from transformers import AutoTokenizer #auto-loads the correct tokenizer for a given model checkpoint

CHECKPOINT = "google-bert/bert-base-uncased" #HF Hub id for the pretrained BERT model
#ag news 4 classes, store in human readable names
NUM_LABELS = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAX_LENGTH = 128 #Max token sequence length

# lightweight struct to bundle three dataset splits &  tokenizer together, 
# so callers can pass around one object instead of four separate variables
@dataclass
class NLPData:
    train: object
    val: object
    test: object
    tokenizer: object

# signature with sensible defaults
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
    #Downloads/ loads from cache AG News dataset, which comes with only train and test splits
    split = raw["train"].train_test_split( #carves 10% out of train to create validation split, since no valid slpit in ag news ds
        test_size=val_fraction, seed=seed, stratify_by_column="label"
    )
    train_raw, val_raw, test_raw = split["train"], split["test"], raw["test"]
    # unpacks and keeps og test set separated avoiding any leakage between model-selection data and the final reported metric

    tokenizer = AutoTokenizer.from_pretrained(checkpoint) #Loads BERT's tokenizer
    # local helper applied to batches of examples to tokenize text field, truncate anything longer 
    # than max_length, and pads shorter seq up to max_length
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    # Applies tokenization to all three splits in batched mode (fast, vectorized), 
    # and drops the raw text column afterward since only token IDs are needed for training/eval
    cols_to_remove = ["text"]
    train = train_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)
    val = val_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)
    test = test_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)
    #packages everything into the dataclass and returns it
    return NLPData(train=train, val=val, test=test, tokenizer=tokenizer)

#Returns the first n (default 256) examples from the training set for PTQ[many quantization 
# schemes need to run a small representative batch through the model to estimate activation 
# ranges/statistics before converting weights to lower precision, min(n, len(...)) guards 
# against requesting more samples than exist]
def get_representative_calibration_batch(nlp_data: NLPData, n: int = 256):
    """Small held-out subset used for PTQ calibration."""
    return nlp_data.train.select(range(min(n, len(nlp_data.train))))

#script entry point, If run directly (not imported), this loads the data and prints split sizes as a quick sanity check
if __name__ == "__main__":
    data = load_agnews_tokenized()
    print(f"train={len(data.train)} val={len(data.val)} test={len(data.test)}")
