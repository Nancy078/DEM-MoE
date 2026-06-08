"""
Train the UWUA (Annotator-Aware) baseline: AnnotatorAwareRegressor.

Usage:
    python train_uwua.py --dataset harm
    python train_uwua.py --dataset toxic
    python train_uwua.py --dataset paact
    python train_uwua.py --dataset polite

Pre-requisites per dataset:
    - The train/dev/test CSVs listed in DATASET_CONFIGS
    - A discrete integer `rating` column (or `scaled_rating` for PAACT which
      uses continuous labels)

Model:
    AnnotatorAwareRegressor injects two annotator signals into the ModernBERT
    input embeddings before the forward pass:
      Ea – learnable annotator embedding
      En – leave-one-out running mean of past label embeddings per annotator
    At inference, unseen annotators fall back to the global-mean En.

Training schedule:
    - Epochs 0..(freeze_epochs-1): backbone frozen; only head/embedding params train
    - Epoch freeze_epochs onward : backbone unfrozen with a much lower LR
    - Scheduler: ReduceLROnPlateau (patience=3) on dev MAE
    - Mixed-precision with GradScaler
    - Early stopping (patience=5) on dev MAE
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm as tqdm_module
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, ModernBertModel

from models import AnnotatorAwareRegressor, EarlyStopping

os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0,1,2,3,4,5,6,7"


# ── Dataset configurations ────────────────────────────────────────────────────
#
#   train/dev/test_csv      – CSV paths
#   backbone                – HuggingFace model name for the BERT encoder
#   hidden_size             – embedding dim for annotator / label embeddings;
#                             should match backbone hidden size (1024 = large,
#                             768 = base)
#   rating_col              – column with integer (or float for PAACT) ratings
#                             used to build the label history buffer
#   use_continuous_labels   – if True, encode labels via a small MLP instead
#                             of nn.Embedding (for continuous-score datasets)
#   mode                    – signals injected: "Ea", "En", or "Ea+En"
#   batch_size              – DataLoader batch size
#   num_epochs              – max training epochs
#   freeze_epochs           – number of epochs to keep backbone frozen at start
#   lr_bert                 – learning rate for backbone parameters
#   lr_head                 – learning rate for all non-backbone parameters
#   dropout                 – dropout before the regression head

DATASET_CONFIGS = {
    "harm": {
        "train_csv":             "harm_train_df.csv",
        "dev_csv":               "harm_dev_df.csv",
        "test_csv":              "harm_test_df.csv",
        "backbone":              "answerdotai/ModernBERT-large",
        "hidden_size":           1024,
        "rating_col":            "rating",
        "use_continuous_labels": False,
        "mode":                  "Ea+En",
        "batch_size":            32,
        "num_epochs":            10,
        "freeze_epochs":         0,
        "lr_bert":               2e-5,
        "lr_head":               2e-5,
        "dropout":               0.1,
        "output_csv":            "harm_uwua_test_preds.csv",
        "model_path":            "uwua_harm.pth",
    },
    "toxic": {
        "train_csv":             "toxic_moe_train_updated.csv",
        "dev_csv":               "toxic_moe_dev_updated.csv",
        "test_csv":              "toxic_moe_test_updated.csv",
        "backbone":              "answerdotai/ModernBERT-base",
        "hidden_size":           768,
        "rating_col":            "toxic_score",
        "use_continuous_labels": False,
        "mode":                  "En",
        "batch_size":            400,
        "num_epochs":            10,
        "freeze_epochs":         3,
        "lr_bert":               2e-6,
        "lr_head":               2e-5,
        "dropout":               0.5,
        "output_csv":            "toxic_uwua_test_preds.csv",
        "model_path":            "uwua_toxic.pth",
    },
    "paact": {
        "train_csv":             "paact_jury_train_updated.csv",
        "dev_csv":               "paact_jury_dev_updated.csv",
        "test_csv":              "paact_jury_test_updated.csv",
        "backbone":              "answerdotai/ModernBERT-large",
        "hidden_size":           1024,
        "rating_col":            "scaled_rating",
        "use_continuous_labels": True,
        "mode":                  "Ea+En",
        "batch_size":            125,
        "num_epochs":            10,
        "freeze_epochs":         0,
        "lr_bert":               2e-6,
        "lr_head":               2e-5,
        "dropout":               0.1,
        "output_csv":            "paact_uwua_test_preds.csv",
        "model_path":            "uwua_paact.pth",
    },
    "polite": {
        "train_csv":             "polite_train_df.csv",
        "dev_csv":               "polite_dev_df.csv",
        "test_csv":              "polite_test_df.csv",
        "backbone":              "answerdotai/ModernBERT-large",
        "hidden_size":           1024,
        "rating_col":            "rating",
        "use_continuous_labels": False,
        "mode":                  "Ea+En",
        "batch_size":            32,
        "num_epochs":            10,
        "freeze_epochs":         0,
        "lr_bert":               2e-5,
        "lr_head":               2e-5,
        "dropout":               0.1,
        "output_csv":            "polite_uwua_test_preds.csv",
        "model_path":            "uwua_polite.pth",
    },
    "off": {
        "train_csv":             "off_train_df.csv",
        "dev_csv":               "off_dev_df.csv",
        "test_csv":              "off_test_df.csv",
        "backbone":              "answerdotai/ModernBERT-large",
        "hidden_size":           1024,
        "rating_col":            "offensiveness",
        "use_continuous_labels": False,
        "mode":                  "Ea+En",
        "batch_size":            100,
        "num_epochs":            10,
        "freeze_epochs":         0,
        "lr_bert":               2e-5,
        "lr_head":               2e-5,
        "dropout":               0.1,
        "output_csv":            "off_uwua_test_preds.csv",
        "model_path":            "uwua_off.pth",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def core(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def load_data(cfg: dict, device: torch.device):
    import pandas as pd

    train_df  = pd.read_csv(cfg["train_csv"])
    dev_df    = pd.read_csv(cfg["dev_csv"])
    test_df   = pd.read_csv(cfg["test_csv"])
    concat_og = pd.concat([train_df, dev_df, test_df])

    rating_col = cfg["rating_col"]

    # Cast rating to int (discrete) or keep as float (continuous)
    if cfg["use_continuous_labels"]:
        for df in (train_df, dev_df, test_df):
            df["_rating"] = df[rating_col].astype(float)
        rating_dtype = torch.float
    else:
        for df in (train_df, dev_df, test_df):
            df["_rating"] = df[rating_col].astype(int)
        rating_dtype = torch.long

    # Annotator IDs → contiguous integers
    le = LabelEncoder().fit(concat_og["annotator_id"])
    for df in (train_df, dev_df, test_df):
        df["annot_idx"] = le.transform(df["annotator_id"])

    # Tokenise text
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])

    def encode(df, max_length=512):
        return tokenizer(
            df["processed"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_enc = encode(train_df)
    dev_enc   = encode(dev_df)
    test_enc  = encode(test_df)

    def make_ds(enc, df):
        return TensorDataset(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            torch.tensor(df["annot_idx"].values, dtype=torch.long,  device=device),
            torch.tensor(df["_rating"].values,   dtype=rating_dtype, device=device),
            torch.tensor(df["scaled_rating"].values, dtype=torch.float, device=device),
        )

    bs = cfg["batch_size"]
    train_ds = make_ds(train_enc, train_df)
    dev_ds   = make_ds(dev_enc,   dev_df)
    test_ds  = make_ds(test_enc,  test_df)

    seen_annotators     = set(train_df["annotator_id"])
    test_df["seen_flag"] = test_df["annotator_id"].isin(seen_annotators)

    return {
        "train_loader":  DataLoader(train_ds, batch_size=bs, shuffle=True),
        "dev_loader":    DataLoader(dev_ds,   batch_size=bs, shuffle=False),
        "test_loader":   DataLoader(test_ds,  batch_size=bs, shuffle=False),
        "train_df":      train_df,
        "dev_df":        dev_df,
        "test_df":       test_df,
        "le":            le,
        "num_annotators": len(le.classes_),
        "num_labels":    0 if cfg["use_continuous_labels"]
                         else int(train_df["_rating"].nunique()),
    }


def init_buffers(model, train_df, cfg):
    """Pre-fill sumEmb / count from the training set."""
    m = core(model)
    with torch.no_grad():
        m.sumEmb.zero_()
        m.count.zero_()
        for a, r in zip(train_df["annot_idx"].tolist(), train_df["_rating"].tolist()):
            if cfg["use_continuous_labels"]:
                r_t = torch.tensor([[r]], dtype=torch.float, device=m.sumEmb.device)
                emb = m.label_mlp(r_t).squeeze(0)
            else:
                emb = m.label_embeddings.weight[int(r)]
            m.sumEmb[a] += emb
            m.count[a]  += 1


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, loss_fn, mode, device):
    model.train()
    total = 0.0
    for input_ids, attn_mask, a_ids, l_ids, y_sc in loader:
        optimizer.zero_grad()
        with autocast():
            preds = model(input_ids, attn_mask, a_ids, l_ids, mode)
            loss  = loss_fn(preds, y_sc)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * input_ids.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, loss_fn, mode):
    model.eval()
    total, preds_all = 0.0, []
    with torch.no_grad():
        for input_ids, attn_mask, a_ids, l_ids, y_sc in loader:
            p     = model(input_ids, attn_mask, a_ids, l_ids, mode)
            total += loss_fn(p, y_sc).item() * input_ids.size(0)
            preds_all.extend(p.cpu().tolist())
    return total / len(loader.dataset), preds_all


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train UWUA baseline")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override num_epochs from config")
    args   = parser.parse_args()
    cfg    = DATASET_CONFIGS[args.dataset]
    if args.num_epochs:
        cfg = cfg.copy()
        cfg["num_epochs"] = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = load_data(cfg, device)

    backbone = ModernBertModel.from_pretrained(cfg["backbone"])
    model    = AnnotatorAwareRegressor(
        bert_model             = backbone,
        num_annotators         = data["num_annotators"],
        num_labels             = data["num_labels"],
        hidden_size            = cfg["hidden_size"],
        dropout_prob           = cfg["dropout"],
        use_continuous_labels  = cfg["use_continuous_labels"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    init_buffers(model, data["train_df"], cfg)

    loss_fn = nn.L1Loss()
    scaler  = GradScaler()

    optimizer = torch.optim.AdamW([
        {"params": core(model).bert.parameters(),       "lr": cfg["lr_bert"]},
        {"params": [p for n, p in core(model).named_parameters()
                    if "bert" not in n],                 "lr": cfg["lr_head"]},
    ], weight_decay=0.01)

    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stop    = EarlyStopping(patience=5)
    mode          = cfg["mode"]
    num_epochs    = cfg["num_epochs"]
    freeze_epochs = cfg["freeze_epochs"]

    for epoch in tqdm_module.trange(num_epochs, desc=args.dataset):
        # Freeze / unfreeze backbone
        if epoch < freeze_epochs:
            for p in core(model).bert.parameters():
                p.requires_grad_(False)
        elif epoch == freeze_epochs and freeze_epochs > 0:
            for p in core(model).bert.parameters():
                p.requires_grad_(True)

        train_loss = train_epoch(model, data["train_loader"], optimizer, scaler,
                                 loss_fn, mode, device)

        dev_loss, dev_preds = evaluate(model, data["dev_loader"], loss_fn, mode)
        scheduler.step(dev_loss)
        print(f"  Epoch {epoch+1:02d}  train_mae={train_loss:.4f}  dev_mae={dev_loss:.4f}")

        early_stop(dev_loss)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    # Save model
    torch.save(core(model).state_dict(), cfg["model_path"])
    print(f"Saved to {cfg['model_path']}")

    # Final test predictions
    _, test_preds = evaluate(model, data["test_loader"], loss_fn, mode)
    test_df = data["test_df"]
    test_df["pred"] = test_preds
    test_df.to_csv(cfg["output_csv"], index=False)
    print(f"Test predictions saved to {cfg['output_csv']}")

    # Quick seen/unseen MAE
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
