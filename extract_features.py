"""
Extract ModernBERT-large [CLS] embeddings for each dataset split.

These feature files are required by train_moe.py before running MoE training.

Usage:
    python extract_features.py --dataset harm
    python extract_features.py --dataset toxic
    python extract_features.py --dataset paact
    python extract_features.py --dataset polite
    python extract_features.py --dataset off

Output:
    A .pt file (path set in DATASET_CONFIGS["output_pt"]) containing a dict:
        {"train": FloatTensor(N_train, 1024),
         "dev":   FloatTensor(N_dev,   1024),
         "test":  FloatTensor(N_test,  1024)}
"""

import argparse
import os

import pandas as pd
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

DEFAULT_BACKBONE = "answerdotai/ModernBERT-large"

DATASET_CONFIGS = {
    "harm": {
        "train_csv": "data/harm_train_df.csv",
        "dev_csv":   "data/harm_dev_df.csv",
        "test_csv":  "data/harm_test_df.csv",
        "text_col":  "data/processed",
        "output_pt": "HARM_MOE_FEATS_NEW.pt",
    },
    "toxic": {
        "train_csv": "data/toxic_moe_train_updated.csv",
        "dev_csv":   "data/toxic_moe_dev_updated.csv",
        "test_csv":  "data/toxic_moe_test_updated.csv",
        "text_col":  "processed",
        "output_pt": "toxic_text_features_og.pt",
    },
    "paact": {
        "train_csv": "data/paact_moe_train_updated_1.csv",
        "dev_csv":   "data/paact_moe_dev_updated_1.csv",
        "test_csv":  "data/paact_moe_test_updated_1.csv",
        "text_col":  "processed",
        "output_pt": "PAACT_MOE_FEATS.pt",
    },
    "polite": {
        "train_csv": "data/polite_train_df.csv",
        "dev_csv":   "data/polite_dev_df.csv",
        "test_csv":  "data/polite_test_df.csv",
        "text_col":  "processed",
        "output_pt": "POLITE_MOE_FEATS.pt",
    },
    "off": {
        "train_csv": "data/off_train_df.csv",
        "dev_csv":   "data/off_dev_df.csv",
        "test_csv":  "data/off_test_df.csv",
        "text_col":  "processed",
        "output_pt": "OFF_MOE_FEATS_NEW.pt",
    },
}


def extract_cls(texts, tokenizer, model, device, batch_size):
    """Return [CLS] embeddings as a float32 tensor (N, hidden_dim)."""
    model.eval()
    parts = []
    for i in tqdm.trange(0, len(texts), batch_size, desc="batches", leave=False):
        enc = tokenizer(
            texts[i : i + batch_size],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        parts.append(out.last_hidden_state[:, 0, :].cpu())
    return torch.cat(parts, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Extract ModernBERT [CLS] features")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--backbone",   default=DEFAULT_BACKBONE,
                        help="HuggingFace model name or local path to ModernBERT")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Tokenization + inference batch size")
    args = parser.parse_args()

    cfg    = DATASET_CONFIGS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.backbone} …")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model     = AutoModel.from_pretrained(args.backbone).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    backbone = model.module if hasattr(model, "module") else model

    splits = {}
    for split_name, csv_key in [("train", "train_csv"), ("dev", "dev_csv"), ("test", "test_csv")]:
        df    = pd.read_csv(cfg[csv_key])
        texts = df[cfg["text_col"]].tolist()
        print(f"{split_name}: {len(texts)} examples")
        splits[split_name] = extract_cls(texts, tokenizer, backbone, device, args.batch_size)

    torch.save(splits, cfg["output_pt"])
    print(f"\nSaved → {cfg['output_pt']}")
    for k, v in splits.items():
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
