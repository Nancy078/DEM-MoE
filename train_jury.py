"""
Train the Jury baseline: JuryRegressor (ModernBERT + annotator embedding + DCN).

Usage:
    python train_jury.py --dataset harm
    python train_jury.py --dataset toxic
    python train_jury.py --dataset paact
    python train_jury.py --dataset polite

Pre-requisites per dataset:
    - The train/dev/test CSVs listed in DATASET_CONFIGS
    - One-hot encoded demographic / survey columns present in the CSVs
      (see survey_cols in each dataset config)

Model:
    JuryRegressor concatenates the ModernBERT [CLS] embedding with a learnable
    annotator embedding and a linear projection of one-hot survey features, then
    passes the result through a Deep & Cross Network (DCN) before a linear head.

Training:
    - AdamW with a linear warmup schedule
    - MSELoss; early stopping (patience=3) on dev MSE
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm as tqdm_module
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from models import EarlyStopping, JuryDataset, JuryRegressor

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


# ── Dataset configurations ────────────────────────────────────────────────────
#
#   train/dev/test_csv  – CSV paths (must contain 'processed', 'scaled_rating',
#                         'instance_id', 'annotator_id' plus all survey_cols)
#   survey_cols         – list of one-hot encoded demographic / survey columns
#                         already present in the CSV (no additional encoding done)
#   batch_size          – DataLoader batch size
#   num_epochs          – max training epochs
#   lr                  – AdamW learning rate
#   warmup_frac         – fraction of total steps used for LR warmup
#   num_cross_layers    – depth of the DCN cross network in JuryRegressor
#   dropout_rate        – dropout used inside JuryRegressor

DATASET_CONFIGS = {
    "harm": {
        "train_csv":       "harm_jury_train.csv",
        "dev_csv":         "harm_jury_dev.csv",
        "test_csv":        "harm_jury_test.csv",
        "survey_cols":     [
            "gender_0", "gender_1",
            "race_0", "race_1", "race_2", "race_3",
            "age_0", "age_1", "age_2",
            "education_0", "education_1", "education_2",
        ],
        "batch_size":      32,
        "num_epochs":      10,
        "lr":              5e-6,
        "warmup_frac":     0.2,
        "num_cross_layers": 5,
        "dropout_rate":    0.2,
        "output_csv":      "harm_jury_test_preds.csv",
        "model_path":      "jury_harm.pth",
    },
    "toxic": {
        "train_csv":       "toxic_jury_train.csv",
        "dev_csv":         "toxic_jury_dev.csv",
        "test_csv":        "toxic_jury_test.csv",
        "survey_cols":     [
            "gender_0", "gender_1",
            "race_0", "race_1", "race_2", "race_3",
            "education_0", "education_1", "education_2",
            "age_range_0", "age_range_1", "age_range_2",
            "political_affiliation_0", "political_affiliation_1",
            "political_affiliation_2", "political_affiliation_3",
            "lgbtq_status_0", "lgbtq_status_1",
        ],
        "batch_size":      16,
        "num_epochs":      10,
        "lr":              5e-6,
        "warmup_frac":     0.2,
        "num_cross_layers": 5,
        "dropout_rate":    0.2,
        "output_csv":      "toxic_jury_test_preds.csv",
        "model_path":      "jury_toxic.pth",
    },
    "paact": {
        "train_csv":       "paact_jury_train_updated.csv",
        "dev_csv":         "paact_jury_dev_updated.csv",
        "test_csv":        "paact_jury_test_updated.csv",
        "survey_cols":     [
            "hcp_freq_0", "hcp_freq_1", "hcp_freq_2",
            "edu_level_0", "edu_level_1", "edu_level_2",
            "age_group_0", "age_group_1", "age_group_2", "age_group_4",
            "gender_0", "gender_1",
            "race_0", "race_1", "race_2", "race_3",
            "occupation_0", "occupation_1", "occupation_2", "occupation_3",
            "doc_trust_category_0", "doc_trust_category_1",
            "doc_trust_category_2", "doc_trust_category_3",
            "ethnic_trust_category_0", "ethnic_trust_category_1",
            "ethnic_trust_category_2", "ethnic_trust_category_3",
        ],
        "batch_size":      16,
        "num_epochs":      5,
        "lr":              5e-6,
        "warmup_frac":     0.2,
        "num_cross_layers": 5,
        "dropout_rate":    0.2,
        "output_csv":      "paact_jury_test_preds.csv",
        "model_path":      "jury_paact.pth",
    },
    "polite": {
        "train_csv":       "polite_jury_train.csv",
        "dev_csv":         "polite_jury_dev.csv",
        "test_csv":        "polite_jury_test.csv",
        "survey_cols":     [
            "gender_0", "gender_1",
            "race_0", "race_1", "race_2", "race_3",
            "age_0", "age_1", "age_2",
            "occupation_0", "occupation_1", "occupation_2",
            "education_0", "education_1", "education_2",
        ],
        "batch_size":      16,
        "num_epochs":      10,
        "lr":              5e-6,
        "warmup_frac":     0.2,
        "num_cross_layers": 5,
        "dropout_rate":    0.2,
        "output_csv":      "polite_jury_test_preds.csv",
        "model_path":      "jury_polite.pth",
    },
    "off": {
        "train_csv":       "jury_off_train.csv",
        "dev_csv":         "jury_off_dev.csv",
        "test_csv":        "jury_off_test.csv",
        "survey_cols":     [
            "gender_Non-binary", "gender_Woman",
            "race_Asian", "race_Black or African American",
            "race_Hispanic or Latino", "race_Native American", "race_White",
            "age_25-29", "age_30-34", "age_35-39", "age_40-44",
            "age_45-49", "age_50-54", "age_54-59", "age_60-64", "age_>65",
            "occupation_Homemaker", "occupation_Other",
            "occupation_Prefer not to disclose", "occupation_Retired",
            "occupation_Self-employed", "occupation_Student",
            "occupation_Unemployed",
            "education_Graduate degree",
            "education_High school diploma or equivalent",
            "education_Less than a high school diploma", "education_Other",
        ],
        "batch_size":      16,
        "num_epochs":      10,
        "lr":              5e-6,
        "warmup_frac":     0.2,
        "num_cross_layers": 5,
        "dropout_rate":    0.2,
        "output_csv":      "off_jury_test_preds.csv",
        "model_path":      "jury_off.pth",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def core(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def load_data(cfg: dict):
    train_df  = pd.read_csv(cfg["train_csv"])
    dev_df    = pd.read_csv(cfg["dev_csv"])
    test_df   = pd.read_csv(cfg["test_csv"])
    concat_og = pd.concat([train_df, dev_df, test_df])

    # Ensure all one-hot columns present (fill missing with 0)
    for df in (train_df, dev_df, test_df):
        for col in cfg["survey_cols"]:
            if col not in df.columns:
                df[col] = 0

    le = LabelEncoder().fit(concat_og["annotator_id"])
    for df in (train_df, dev_df, test_df):
        df["encoded_annotator_id"] = le.transform(df["annotator_id"])

    seen_annotators     = set(train_df["annotator_id"])
    test_df["seen_flag"] = test_df["annotator_id"].isin(seen_annotators)

    return {
        "train_df":       train_df,
        "dev_df":         dev_df,
        "test_df":        test_df,
        "num_annotators": len(le.classes_),
    }


def make_loaders(cfg, data):
    survey_cols = cfg["survey_cols"]
    bs          = cfg["batch_size"]
    tokenizer_name = "answerdotai/ModernBERT-large"

    train_ds = JuryDataset(data["train_df"], survey_cols, tokenizer_name=tokenizer_name)
    dev_ds   = JuryDataset(data["dev_df"],   survey_cols, tokenizer_name=tokenizer_name)
    test_ds  = JuryDataset(data["test_df"],  survey_cols, tokenizer_name=tokenizer_name)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(dev_ds,   batch_size=bs, shuffle=False),
        DataLoader(test_ds,  batch_size=bs, shuffle=False),
    )


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)
        survey_info    = batch["survey_info"].to(device)
        annotator_id   = batch["annotator_id"].to(device)
        targets        = batch["targets"].to(device)

        outputs = model(input_ids, attention_mask, survey_info, annotator_id)
        loss    = criterion(outputs.squeeze(-1), targets.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()

        del input_ids, attention_mask, survey_info, annotator_id, targets, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

    return total / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds_all = 0.0, []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            survey_info    = batch["survey_info"].to(device)
            annotator_id   = batch["annotator_id"].to(device)
            targets        = batch["targets"].to(device)

            outputs = model(input_ids, attention_mask, survey_info, annotator_id)
            loss    = criterion(outputs.squeeze(-1), targets.float())
            total  += loss.item()
            preds_all.extend(outputs.squeeze(-1).cpu().tolist())

            del batch, input_ids, attention_mask, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

    return total / len(loader), preds_all


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Jury baseline")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()
    cfg  = DATASET_CONFIGS[args.dataset]
    if args.num_epochs:
        cfg = cfg.copy()
        cfg["num_epochs"] = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = load_data(cfg)
    train_loader, dev_loader, test_loader = make_loaders(cfg, data)

    survey_size = len(cfg["survey_cols"])
    model = JuryRegressor(
        annotator_size   = data["num_annotators"],
        survey_size      = survey_size,
        num_cross_layers = cfg["num_cross_layers"],
        dropout_rate     = cfg["dropout_rate"],
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion   = nn.MSELoss()
    optimizer   = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_frac"])
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    early_stop  = EarlyStopping(patience=3)

    for epoch in tqdm_module.trange(cfg["num_epochs"], desc=args.dataset):
        train_loss         = train_epoch(model, train_loader, optimizer, scheduler,
                                         criterion, device)
        dev_loss, dev_preds = evaluate(model, dev_loader, criterion, device)

        dev_mae = mean_absolute_error(
            data["dev_df"]["scaled_rating"].tolist(), dev_preds
        )
        print(f"  Epoch {epoch+1:02d}  train_mse={train_loss:.4f}  "
              f"dev_mse={dev_loss:.4f}  dev_mae={dev_mae:.4f}")

        torch.save(core(model).state_dict(), cfg["model_path"])

        early_stop(dev_loss)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    # Final test predictions
    _, test_preds = evaluate(model, test_loader, criterion, device)
    test_df = data["test_df"]
    test_df["pred"] = test_preds
    test_df.to_csv(cfg["output_csv"], index=False)
    print(f"Test predictions saved to {cfg['output_csv']}")

    seen_mae   = mean_absolute_error(
        test_df.loc[test_df["seen_flag"],  "scaled_rating"],
        test_df.loc[test_df["seen_flag"],  "pred"],
    )
    unseen_mae = mean_absolute_error(
        test_df.loc[~test_df["seen_flag"], "scaled_rating"],
        test_df.loc[~test_df["seen_flag"], "pred"],
    )
    print(f"Test MAE  seen={seen_mae:.4f}  unseen={unseen_mae:.4f}")


if __name__ == "__main__":
    main()
