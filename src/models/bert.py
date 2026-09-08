"""BERT baseline: load pretrained checkpoint, attach classification head."""
from transformers import AutoModelForSequenceClassification, AutoConfig

from src.data.nlp_data import CHECKPOINT, NUM_LABELS, LABEL_NAMES


def build_bert_classifier(checkpoint: str = CHECKPOINT, num_labels: int = NUM_LABELS):
    """Stage 0 instantiate BERT with a fresh task specific classification head."""
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    config = AutoConfig.from_pretrained(
        checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    return model


def record_pretrained_checkpoint_info(checkpoint: str = CHECKPOINT) -> dict:
    """Stage 0 record info about the original pretrained checkpoint before any
    fine-tuning or optimization, for the reproducibility log."""
    config = AutoConfig.from_pretrained(checkpoint)
    n_params = sum(p.numel() for p in build_bert_classifier(checkpoint).parameters())
    return {
        "checkpoint": checkpoint,
        "architecture": config.architectures[0] if config.architectures else "BertForSequenceClassification",
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "total_parameters": n_params,
    }
