"""
Train the mBERT baseline: plain ModernBERT-large fine-tuned for regression.

Usage:
    python train_mbert.py --dataset harm
    python train_mbert.py --dataset toxic
    python train_mbert.py --dataset paact
    python train_mbert.py --dataset polite

Pre-requisites per dataset:
    - The train/dev/test CSVs listed in DATASET_CONFIGS

Model:
    BERTModel is a thin wrapper around ModernBERT-large: the [CLS] token
    representation is passed through a single linear head to produce a scalar
    regression prediction. No annotator information is used, making this the
    text-only lower-bound baseline.

Training:
    - AdamW with linear warmup schedule
    - MSELoss, mixed-precision (GradScaler)
    - Early stopping (patience=5) on dev MAE
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm as tqdm_module
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from models import BERTModel, EarlyStopping

os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0,1,2,3,4,5,6,7"


# ── Dataset configurations ────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "harm": {
        "train_csv":  "harm_train_df.csv",
        "dev_csv":    "harm_dev_df.csv",
        "test_csv":   "harm_test_df.csv",
        "batch_size": 32,
        "num_epochs": 50,
        "lr":         1e-4,
        "warmup_frac": 0.2,
        "output_csv": "harm_mbert_test_preds.csv",
        "model_path": "mbert_harm.pth",
    },
    "toxic": {
        "train_csv":  "toxic_moe_train_updated.csv",
        "dev_csv":    "toxic_moe_dev_updated.csv",
        "test_csv":   "toxic_moe_test_updated.csv",
        "batch_size": 32,
        "num_epochs": 50,
        "lr":         1e-4,
        "warmup_frac": 0.2,
        "output_csv": "toxic_mbert_test_preds.csv",
        "model_path": "mbert_toxic.pth",
    },
    "paact": {
        "train_csv":  "paact_moe_train_updated_1.csv",
        "dev_csv":    "paact_moe_dev_updated_1.csv",
        "test_csv":   "paact_moe_test_updated_1.csv",
        "batch_size": 32,
        "num_epochs": 50,
        "lr":         1e-4,
        "warmup_frac": 0.2,
        "output_csv": "paact_mbert_test_preds.csv",
        "model_path": "mbert_paact.pth",
    },
    "polite": {
        "train_csv":  "polite_train_df.csv",
        "dev_csv":    "polite_dev_df.csv",
        "test_csv":   "polite_test_df.csv",
        "batch_size": 32,
        "num_epochs": 50,
        "lr":         1e-4,
        "warmup_frac": 0.2,
        "output_csv": "polite_mbert_test_preds.csv",
        "model_path": "mbert_polite.pth",
    },
    "off": {
        "train_csv":  "off_train_df.csv",
        "dev_csv":    "off_dev_df.csv",
        "test_csv":   "off_test_df.csv",
        "batch_size": 32,
        "num_epochs": 50,
        "lr":         1e-4,
        "warmup_frac": 0.2,
        "output_csv": "off_mbert_test_preds.csv",
        "model_path": "mbert_off.pth",
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(cfg: dict, device: torch.device):
    train_df  = pd.read_csv(cfg["train_csv"])
    dev_df    = pd.read_csv(cfg["dev_csv"])
    test_df   = pd.read_csv(cfg["test_csv"])

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    def tokenise(df, max_length=512):
        return tokenizer(
            df["processed"].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_enc = tokenise(train_df)
    dev_enc   = tokenise(dev_df)
    test_enc  = tokenise(test_df)

    def target(df):
        return torch.tensor(df["scaled_rating"].tolist(), dtype=torch.float32).unsqueeze(1).to(device)

    def make_ds(enc, df):
        return TensorDataset(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            target(df),
        )

    bs = cfg["batch_size"]
    seen_annotators     = set(train_df["annotator_id"])
    test_df["seen_flag"] = test_df["annotator_id"].isin(seen_annotators)

    return {
        "train_loader": DataLoader(make_ds(train_enc, train_df), batch_size=bs, shuffle=True),
        "dev_loader":   DataLoader(make_ds(dev_enc,   dev_df),   batch_size=bs, shuffle=False),
        "test_loader":  DataLoader(make_ds(test_enc,  test_df),  batch_size=bs, shuffle=False),
        "test_df":      test_df,
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device):
    model.train()
    total = 0.0
    for input_ids, attention_mask, targets in loader:
        input_ids, attention_mask, targets = (
            input_ids.to(device), attention_mask.to(device), targets.to(device)
        )
        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask)
            loss    = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for input_ids, attention_mask, targets in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with autocast():
                outputs = model(input_ids, attention_mask)
            preds_all.extend(outputs.cpu().numpy().flatten())
            targets_all.extend(targets.cpu().numpy().flatten())
    return np.array(preds_all), np.array(targets_all)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train mBERT baseline")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()
    cfg  = DATASET_CONFIGS[args.dataset]
    if args.num_epochs:
        cfg = cfg.copy()
        cfg["num_epochs"] = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = load_data(cfg, device)

    model = BERTModel(model_name="answerdotai/ModernBERT-large").to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion    = nn.MSELoss()
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    total_steps  = len(data["train_loader"]) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_frac"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler       = GradScaler()
    early_stop   = EarlyStopping(patience=5)

    for epoch in tqdm_module.trange(cfg["num_epochs"], desc=args.dataset):
        train_loss = train_epoch(model, data["train_loader"], optimizer,
                                  criterion, scaler, scheduler, device)
        dev_preds, dev_targets = evaluate(model, data["dev_loader"], device)
        dev_mae = mean_absolute_error(dev_targets, dev_preds)
        print(f"  Epoch {epoch+1:02d}  train_mse={train_loss:.4f}  dev_mae={dev_mae:.4f}")

        early_stop(dev_mae)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    torch.save(
        (model.module if hasattr(model, "module") else model).state_dict(),
        cfg["model_path"],
    )
    print(f"Saved to {cfg['model_path']}")

    test_preds, test_targets = evaluate(model, data["test_loader"], device)
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
