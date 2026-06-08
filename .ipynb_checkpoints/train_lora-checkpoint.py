"""
Train the LoRA baseline: Llama-3.1-8B-Instruct with PEFT LoRA adapters for
demographic-conditioned classification of annotator ratings.

Usage:
    python train_lora.py --dataset harm   [--model_dir /path/to/llama] [--cache_dir /path/to/cache]
    python train_lora.py --dataset toxic
    python train_lora.py --dataset paact
    python train_lora.py --dataset polite

Model:
    Llama-3.1-8B-Instruct loaded as AutoModelForSequenceClassification, wrapped
    with LoRA adapters (r=8, alpha=16) applied to the q_proj and v_proj attention
    weight matrices.  Only the LoRA adapter weights and the classification head are
    trained; the backbone is frozen.

Prompting:
    Each sample is converted into a structured prompt that includes the annotator's
    demographic profile and the task-specific question before the text to rate.
    The model predicts a discrete class label (e.g., A/B/C for harm, A–E for the
    other datasets).  Class labels are zero-indexed integers.

Training:
    Uses the HuggingFace Trainer API.  Gradient accumulation is used where the
    effective batch size would otherwise exceed GPU memory.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ["TORCHDYNAMO_DISABLE"] = "1"


# ── Dataset configurations ────────────────────────────────────────────────────
#
#   train/test_csv      – CSV paths (no separate dev set; Trainer uses test for eval)
#   label_col           – column containing the raw rating / score
#   label_offset        – subtracted from label_col to produce 0-indexed classes
#                         (e.g. -1 when ratings are 1-indexed)
#   num_labels          – number of output classes
#   text_col            – column containing the text to rate
#   demo_fields         – ordered list of demographic column names used in the prompt
#   question            – task-specific question inserted in the prompt
#   content_label       – label for the text block in the prompt
#                         ("Conversation", "Comment", "Email", …)
#   choices             – ordered list of (letter, description) pairs for answer
#                         options shown in the prompt; must have len == num_labels
#   batch_size          – per-device train/eval batch size
#   grad_accum          – gradient accumulation steps
#   num_epochs          – number of training epochs
#   output_csv          – where to write test predictions

DATASET_CONFIGS = {
    "harm": {
        "train_csv":     "harm_train_df.csv",
        "test_csv":      "harm_test_df.csv",
        "label_col":     "rating",
        "label_offset":  -1,
        "num_labels":    3,
        "text_col":      "processed",
        "demo_fields":   ["gender", "race", "age", "education"],
        "demo_labels":   {
            "gender":    "gender",
            "race":      "race",
            "age":       "age",
            "education": "education",
        },
        "question":      (
            "Consider you read this conversation between a chat bot and a human, "
            "do you think it is safe or unsafe in terms of harmful content, "
            "unfair bias or safety policy violations?"
        ),
        "content_label": "Conversation",
        "choices":       [("A", "safe"), ("B", "unsure"), ("C", "unsafe")],
        "batch_size":    16,
        "grad_accum":    1,
        "num_epochs":    3,
        "output_csv":    "harm_lora_predictions.csv",
    },
    "toxic": {
        "train_csv":     "toxic_prompting_train_df.csv",
        "test_csv":      "toxic_prompting_test_df.csv",
        "label_col":     "toxic_score",
        "label_offset":  0,
        "num_labels":    5,
        "text_col":      "processed",
        "demo_fields":   ["gender", "race", "education", "age_range",
                          "political_affiliation", "lgbtq_status"],
        "demo_labels":   {
            "gender":                "gender",
            "race":                  "race",
            "education":             "education",
            "age_range":             "age range",
            "political_affiliation": "political affiliation",
            "lgbtq_status":          "lgbtq status",
        },
        "question":      "Consider you read this social media comment, how toxic is this comment?",
        "content_label": "Comment",
        "choices":       [
            ("A", "not toxic at all"), ("B", "barely toxic"), ("C", "somewhat toxic"),
            ("D", "moderately toxic"),  ("E", "very toxic"),
        ],
        "batch_size":    4,
        "grad_accum":    1,
        "num_epochs":    1,
        "output_csv":    "toxic_lora_predictions.csv",
    },
    "paact": {
        "train_csv":     "paact_prompting_train_df_updated.csv",
        "test_csv":      "paact_prompting_test_df_updated.csv",
        "label_col":     "pcc",
        "label_offset":  0,
        "num_labels":    5,
        "text_col":      "processed",
        "demo_fields":   ["hcp_freq", "edu_level", "age_group", "gender", "race",
                          "occupation", "doc_trust_category", "ethnic_trust_category"],
        "demo_labels":   {
            "hcp_freq":               "Times went to see doctors or healthcare professionals during the past 12 months (not counting the times at emergency room)",
            "edu_level":              "Education Level",
            "age_group":              "Age Group",
            "gender":                 "Gender",
            "race":                   "Race",
            "occupation":             "Occupation",
            "doc_trust_category":     "Trust in the medical profession",
            "ethnic_trust_category":  "Ethnic group-based trust in medical profession",
        },
        "question":      (
            "You are tasked with evaluating snippets of doctor-patient conversations. "
            "Each snippet involves a patient diagnosed with prostate cancer. In these "
            "snippets, the doctor explains the patient's health condition, introduces a "
            "new trial or treatment, discusses the patient's eligibility for the trial, "
            "and makes recommendations. Rate the extent to which the doctor shows "
            "patient centered communication."
        ),
        "content_label": "Comment",
        "choices":       [
            ("A", "not at all"), ("B", "barely"), ("C", "somewhat"),
            ("D", "moderately"),  ("E", "very"),
        ],
        "batch_size":    2,
        "grad_accum":    8,
        "num_epochs":    3,
        "output_csv":    "paact_lora_predictions.csv",
    },
    "polite": {
        "train_csv":     "polite_train_df.csv",
        "test_csv":      "polite_test_df.csv",
        "label_col":     "politeness",
        "label_offset":  -1,
        "num_labels":    5,
        "text_col":      "text",
        "demo_fields":   ["gender", "race", "age", "occupation", "education"],
        "demo_labels":   {
            "gender":    "gender",
            "race":      "race",
            "age":       "age",
            "occupation": "occupation",
            "education": "education",
        },
        "question":      "Consider you read this email from a colleague, how polite do you think it is?",
        "content_label": "Email",
        "choices":       [
            ("A", "not polite at all"), ("B", "barely polite"), ("C", "somewhat polite"),
            ("D", "moderately polite"),  ("E", "very polite"),
        ],
        "batch_size":    8,
        "grad_accum":    1,
        "num_epochs":    3,
        "output_csv":    "polite_lora_predictions.csv",
    },
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(row: pd.Series, cfg: dict) -> str:
    """
    Construct a structured prompt for one annotator-rating example.

    The prompt encodes the annotator's demographic profile, the task-specific
    question, the text to rate, and the multiple-choice answer options.
    """
    demo_parts = ", ".join(
        f"{cfg['demo_labels'][field]}: {row[field]}"
        for field in cfg["demo_fields"]
    )
    choices_str = "\n".join(f"({letter}) {desc}" for letter, desc in cfg["choices"])

    return (
        f"**Your Profile**: unique identifier {row['user_id']}. "
        f"Demographics: {demo_parts}\n"
        f"**Question**: {cfg['question']}\n"
        f"**{cfg['content_label']}**: {row[cfg['text_col']]}\n"
        f"{choices_str}\n"
        f"**Answer**: ("
    )


# ── Custom trainer ────────────────────────────────────────────────────────────

class ClassificationTrainer(Trainer):
    """Override compute_loss to use CrossEntropyLoss with explicit int64 labels."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels  = inputs.pop("labels").to(torch.int64)
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LoRA baseline (Llama-3.1-8B)")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_dir",  default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--cache_dir",  default=None,
                        help="Directory for caching model weights")
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    if args.num_epochs:
        cfg = cfg.copy()
        cfg["num_epochs"] = args.num_epochs

    # Load and prepare DataFrames
    train_df = pd.read_csv(cfg["train_csv"])
    test_df  = pd.read_csv(cfg["test_csv"])

    for df in (train_df, test_df):
        df.rename(columns={"annotator_id": "user_id"}, inplace=True)
        df["label"] = df[cfg["label_col"]] + cfg["label_offset"]

    train_df["prompt"] = train_df.apply(lambda r: build_prompt(r, cfg), axis=1)
    test_df["prompt"]  = test_df.apply(lambda r: build_prompt(r, cfg), axis=1)

    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)

    # Tokenise
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, cache_dir=args.cache_dir, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["prompt"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_ds = train_ds.map(tokenize)
    test_ds  = test_ds.map(tokenize)
    cols     = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch",  columns=cols)

    # Model + LoRA
    fp16  = torch.cuda.is_available()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        cache_dir=args.cache_dir,
        num_labels=cfg["num_labels"],
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r              = 8,
        lora_alpha     = 16,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = "SEQ_CLS",
    ))
    model.print_trainable_parameters()

    # Train
    training_args = TrainingArguments(
        output_dir                  = "./",
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"],
        gradient_accumulation_steps = cfg["grad_accum"],
        num_train_epochs            = cfg["num_epochs"],
        evaluation_strategy         = "epoch",
        save_strategy               = "no",
        learning_rate               = 6e-5,
        logging_steps               = 10,
        fp16                        = fp16,
        report_to                   = "none",
    )

    trainer = ClassificationTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = test_ds,
        data_collator = DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    # Predict and save
    predictions = trainer.predict(test_ds)
    preds       = np.argmax(predictions.predictions, axis=1)
    test_df["predicted_label"] = preds
    test_df.to_csv(cfg["output_csv"], index=False)
    print(f"Predictions saved to {cfg['output_csv']}")


if __name__ == "__main__":
    main()
