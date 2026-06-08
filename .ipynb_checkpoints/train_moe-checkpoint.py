"""
Train a Top-K Mixture-of-Experts (MoE) model with Bayesian annotator/identity
embeddings and orthogonality + variance + demographic-KL auxiliary losses.

Usage:
    python train_moe.py --dataset harm
    python train_moe.py --dataset toxic
    python train_moe.py --dataset paact
    python train_moe.py --dataset polite

Pre-requisites per dataset:
    - The train/dev/test CSVs listed in DATASET_CONFIGS
    - A .pt file of pre-extracted ModernBERT-large [CLS] embeddings
      (call get_text_features_batched() and torch.save() to create one)
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import logging
import os
import warnings
from collections import Counter, defaultdict

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm_module
from scipy import stats as st
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, ModernBertModel, get_linear_schedule_with_warmup

# ── Local ─────────────────────────────────────────────────────────────────────
from models import (
    BayesianAnnotatorEmbedding,
    BayesianIdentityEmbedding,
    TopKMoE,
    EarlyStopping,
    EarlyStoppingCorr,
)

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=DeprecationWarning,
    module=r"pandas\.core\.groupby",
)


# ── Dataset configurations ────────────────────────────────────────────────────
#
# Top-level config keys
# ─────────────────────
#   train/dev/test_csv  – paths to the annotator-rating CSV files
#   features_pt         – .pt file of pre-extracted ModernBERT-large [CLS] embeddings;
#                         keys "train", "dev", "test" each map to a float32 tensor
#   demo_feats          – demographic attribute columns used as annotator identity features
#   batch_size          – DataLoader batch size (smaller for larger datasets)
#   num_epochs          – maximum training epochs per Optuna trial (early stopping may
#                         terminate sooner)
#   num_experts         – number of MoE experts; we set this equal to the number of
#                         demographic attributes so each expert can specialize on one group
#   direction           – Optuna direction: "minimize" for MAE-based objectives,
#                         "maximize" for correlation-based objectives
#   score_metric        – which metric drives the composite dev score:
#                           "mae"  → score = seen_w * seen_MAE  + unseen_w * unseen_MAE  (minimize)
#                           "corr" → score = seen_w * seen_corr + unseen_w * unseen_corr (maximize)
#   score_weights       – (seen_weight, unseen_weight) for the composite score; e.g. (0.7, 0.3)
#                         means we weight seen-annotator performance 70% and unseen 30%
#   baseline_mae        – per-demographic-group MAE of a baseline model, used for
#                         Δ-logging during training (set to None if not available)
#   params              – best hyperparameters found by prior Optuna search runs
#
#
# Hyperparameter ("params") keys
# ───────────────────────────────
# Learning rates
#   bias_lr      – step size for the per-expert load-balancing bias correction; updated
#                  every batch proportional to the deviation from uniform expert usage
#   lr_gate      – learning rate for the expert selector (gating network) parameters;
#                  kept separate from lr_main to avoid over-fitting the routing too quickly
#   lr_main      – learning rate for all other parameters (encoder projection, Bayesian
#                  annotator/identity embeddings, expert MLPs, output head)
#
# Gating / top-k selection
#   topk_exp     – number of experts each token is routed to (k in top-k gating)
#   T_init       – initial temperature for Gumbel-softmax gating; high values make the
#                  routing distribution more uniform at the start of training, encouraging
#                  all experts to be explored
#   T_decay      – multiplicative per-epoch decay applied to T after the warmup period;
#                  anneals the temperature toward T_min to sharpen expert assignments
#   T_min        – minimum (floor) temperature; prevents the gating from collapsing to a
#                  hard argmax, preserving some gradient signal for all experts
#   T_warmup_epochs – number of epochs to hold T_init before beginning decay; gives the
#                  expert embeddings time to differentiate before routing becomes sharp
#
# KL regularisation on Bayesian embeddings
#   kl_identity_w – weight on the KL divergence term for the Bayesian identity
#                   (demographic) embeddings; keeps the posterior close to the prior
#   annot_weight  – weight on the KL divergence term for the Bayesian annotator
#                   embeddings; annealed via kl_anneal() during early training
#
# Capacity / overflow routing
#   overflow_on    – if True, tokens routed to an over-capacity expert are reassigned
#                    to their second-choice expert, preventing expert collapse
#   capacity_scale – fraction of the uniform per-expert token budget used as the hard
#                    capacity limit (< 1.0 = stricter, encourages spread)
#
# Phased loss-weight schedule (A → B → C)
#   Training uses three phases to progressively increase specialisation pressure:
#     Phase A – light auxiliary penalties; gating learns to use multiple experts
#     Phase B – heavier penalties; experts begin to specialise by demographic group
#     Phase C – weights held constant; routing and predictions stabilise
#   Transitions are triggered by thresholds on load standard deviation and entropy.
#
#   load_w_{A,B,C} – weight on the load standard deviation penalty in each phase;
#                    penalises uneven expert utilisation to prevent expert collapse
#   orth_w_{A,B,C} – weight on the orthogonality loss L_o (Eq. 6 in the paper);
#                    penalises aligned hidden representations across experts assigned
#                    to the same token, encouraging functional diversity
#   var_w_{A,B,C}  – weight on the variance loss L_v (Eq. 7 in the paper);
#                    rewards spread in the gating weight distribution across experts
#   cap_mult_{A,B,C} – max-norm used when renormalising the expert selector weight
#                    matrix at the end of each step; constrains selector magnitude
#
#   load_w / orth_w / var_w – running (current-epoch) values of the above weights;
#                    initialised to the Phase-A values and exponentially interpolated
#                    toward the phase target each epoch
#
# Phase-C encoder fine-tuning
#   encoder_bump – multiplicative LR boost applied to encoder parameters (lr_main)
#                  when entering Phase C; the gating network is frozen at this point,
#                  allowing each expert MLP to fine-tune on its specialised subset
#
# Demographic routing regularisation (optional)
#   demog_kl_w   – weight on the symmetric KL divergence between expert routing
#                  distributions of different demographic groups; encourages the model
#                  to route tokens to similar experts regardless of annotator group,
#                  reducing routing-induced demographic bias (set to 0.0 to disable)

DATASET_CONFIGS = {
    "harm": {
        "train_csv":      "harm_train_df.csv",
        "dev_csv":        "harm_dev_df.csv",
        "test_csv":       "harm_test_df.csv",
        "features_pt":    "HARM_MOE_FEATS_NEW.pt",
        "demo_feats":     ["gender", "race", "age", "education"],
        "batch_size":     256,
        "num_epochs":     20,
        "num_experts":    4,
        "direction":      "minimize",
        "score_metric":   "mae",
        "score_weights":  (0.7, 0.3),
        "baseline_mae":   {"gender": 0.705, "race": 0.726, "age": 0.710, "education": 0.716},
        "params": {
            "bias_lr":         0.03063423530385468,
            "lr_gate":         3e-4,
            "lr_main":         3.0650225540848456e-4,
            "topk_exp":        2,
            "T_init":          38.8474154085148,
            "T_decay":         0.9319178682131377,
            "T_min":           0.6392579408952574,
            "T_warmup_epochs": 7,
            "kl_identity_w":   1e-4,
            "annot_weight":    1e-3,
            "overflow_on":     True,
            "capacity_scale":  0.8,
            "cap_mult":        1.14399,
            "cap_mult_A":      1.0,
            "cap_mult_B":      1.14399,
            "cap_mult_C":      1.14399,
            "load_w_A":        0.27812950557155747,
            "orth_w_A":        0.10805951233833459,
            "var_w_A":         0.17231175333507015,
            "load_w_B":        0.5276725632026085,
            "orth_w_B":        0.12248629048060601,
            "var_w_B":         0.14293366266666038,
            "load_w_C":        0.615037591342064,
            "orth_w_C":        0.6519258439085289,
            "var_w_C":         0.3482795884766893,
            "encoder_bump":    1.5,
            "load_w":          0.27812950557155747,
            "orth_w":          0.10805951233833459,
            "var_w":           0.17231175333507015,
            "demog_kl_w":      0.18645484213205016,
        },
    },

    "toxic": {
        "train_csv":      "toxic_moe_train_updated.csv",
        "dev_csv":        "toxic_moe_dev_updated.csv",
        "test_csv":       "toxic_moe_test_updated.csv",
        "features_pt":    "toxic_text_features_og.pt",
        "demo_feats":     ["gender", "race", "education", "age_range",
                           "political_affiliation", "lgbtq_status"],
        "batch_size":     128,
        "num_epochs":     50,
        "num_experts":    6,
        "direction":      "maximize",
        "score_metric":   "corr",
        "score_weights":  (0.7, 0.3),
        "baseline_mae":   None,
        "params": {
            "bias_lr":         0.03063423530385468,
            "lr_gate":         3.1907174942580015e-4,
            "lr_main":         3.0650225540848456e-4,
            "topk_exp":        2,
            "T_init":          17.47093418619315,
            "T_decay":         0.9081737943009455,
            "T_min":           4.102966005661472,
            "T_warmup_epochs": 3,
            "kl_identity_w":   1e-4,
            "annot_weight":    1e-3,
            "overflow_on":     True,
            "capacity_scale":  0.9247424649829217,
            "cap_mult":        1.14399,
            "cap_mult_A":      1.0,
            "cap_mult_B":      1.14399,
            "cap_mult_C":      1.14399,
            "load_w_A":        0.26109281932159034,
            "orth_w_A":        0.05068470934369791,
            "var_w_A":         0.09828150002076738,
            "load_w_B":        0.46399034544215156,
            "orth_w_B":        0.2524388225960015,
            "var_w_B":         0.10151431614764068,
            "load_w_C":        0.5566644133241406,
            "orth_w_C":        0.3043039078358462,
            "var_w_C":         0.36338047527924056,
            "encoder_bump":    2.0,
            "load_w":          0.26109281932159034,
            "orth_w":          0.05068470934369791,
            "var_w":           0.09828150002076738,
            "demog_kl_w":      0.0,
        },
    },

    "paact": {
        "train_csv":      "paact_moe_train_updated_1.csv",
        "dev_csv":        "paact_moe_dev_updated_1.csv",
        "test_csv":       "paact_moe_test_updated_1.csv",
        "features_pt":    "PAACT_MOE_FEATS.pt",
        "demo_feats":     ["hcp_freq", "edu_level", "age_group", "gender",
                           "race", "occupation", "doc_trust_category", "ethnic_trust_category"],
        "batch_size":     32,
        "num_epochs":     50,
        "num_experts":    8,
        "direction":      "maximize",
        "score_metric":   "corr",
        "score_weights":  (0.6, 0.4),
        "baseline_mae":   None,
        "params": {
            "bias_lr":         0.06801475991423374,
            "lr_gate":         5.9353062832292536e-5,
            "lr_main":         1.5807097743457212e-3,
            "topk_exp":        3,
            "T_init":          13.769005307125093,
            "T_decay":         0.8916691503526311,
            "T_min":           3.678212844929815,
            "T_warmup_epochs": 4,
            "kl_identity_w":   1.365023943716972e-4,
            "annot_weight":    1.169830581156675e-3,
            "overflow_on":     True,
            "capacity_scale":  0.5293782452350007,
            "cap_mult":        2.296414797647836,
            "cap_mult_A":      1.0799149702576643,
            "cap_mult_B":      1.0071371103823163,
            "cap_mult_C":      1.43256115827256,
            "load_w_A":        0.13037907487296954,
            "orth_w_A":        0.051300535116646374,
            "var_w_A":         0.0390314611999813,
            "load_w_B":        0.49467580411164125,
            "orth_w_B":        0.25630241157548445,
            "var_w_B":         0.29632085067749364,
            "load_w_C":        0.7452463562455109,
            "orth_w_C":        0.6303900660399638,
            "var_w_C":         0.6900527376302559,
            "encoder_bump":    2.178096324411714,
            "load_w":          0.13037907487296954,
            "orth_w":          0.051300535116646374,
            "var_w":           0.0390314611999813,
            "demog_kl_w":      0.015078847071585442,
        },
    },

    "polite": {
        "train_csv":      "polite_train_df.csv",
        "dev_csv":        "polite_dev_df.csv",
        "test_csv":       "polite_test_df.csv",
        "features_pt":    "POLITE_MOE_FEATS.pt",
        "demo_feats":     ["gender", "race", "age", "occupation", "education"],
        "batch_size":     128,
        "num_epochs":     50,
        "num_experts":    5,
        "direction":      "maximize",
        "score_metric":   "corr",
        "score_weights":  (0.7, 0.3),
        "baseline_mae":   None,
        "params": {
            "bias_lr":         0.03063423530385468,
            "lr_gate":         3.1907174942580015e-4,
            "lr_main":         3.0650225540848456e-4,
            "topk_exp":        2,
            "T_init":          17.47093418619315,
            "T_decay":         0.9081737943009455,
            "T_min":           4.102966005661472,
            "T_warmup_epochs": 3,
            "kl_identity_w":   1e-4,
            "annot_weight":    1e-3,
            "overflow_on":     True,
            "capacity_scale":  0.9247424649829217,
            "cap_mult":        1.14399,
            "cap_mult_A":      1.0,
            "cap_mult_B":      1.14399,
            "cap_mult_C":      1.14399,
            "load_w_A":        0.26109281932159034,
            "orth_w_A":        0.05068470934369791,
            "var_w_A":         0.09828150002076738,
            "load_w_B":        0.46399034544215156,
            "orth_w_B":        0.2524388225960015,
            "var_w_B":         0.10151431614764068,
            "load_w_C":        0.5566644133241406,
            "orth_w_C":        0.3043039078358462,
            "var_w_C":         0.36338047527924056,
            "encoder_bump":    2.0,
            "load_w":          0.26109281932159034,
            "orth_w":          0.05068470934369791,
            "var_w":           0.09828150002076738,
            "demog_kl_w":      0.0,
        },
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def core(m: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DDP wrapper if present."""
    return m.module if hasattr(m, "module") else m


def get_biases(m: nn.Module):
    return core(m).expert_biases, core(m).bias_lr


def renorm_selector(model: nn.Module, max_norm: float = 1.0):
    with torch.no_grad():
        W      = core(model).expert_selector.weight
        factor = torch.clamp(W.norm(dim=1, keepdim=True) / max_norm, min=1.0)
        W.div_(factor)


def get_text_features_batched(text_data, tokenizer, model, device="cpu", batch_size=128):
    """Extract [CLS] embeddings in batches. Returns np.ndarray of shape (N, D)."""
    model.to(device)
    model.eval()
    all_features = []
    for i in tqdm_module.trange(0, len(text_data), batch_size):
        batch  = text_data[i : i + batch_size].tolist()
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        all_features.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(all_features, axis=0)


# ── Metrics and logging ───────────────────────────────────────────────────────

def slice_metrics(df_slice: pd.DataFrame):
    """Return (indiv_corr, indiv_mae, snip_corr, snip_mae, emd) for a data slice."""
    preds   = df_slice["pred"].to_numpy()
    targets = df_slice["scaled_rating"].to_numpy()

    indiv_r   = np.corrcoef(preds, targets)[1, 0]
    indiv_mae = mean_absolute_error(targets, preds)

    g_pred = df_slice.groupby("instance_id")["pred"].mean()
    g_true = df_slice.groupby("instance_id")["scaled_rating"].mean()
    snip_r   = np.corrcoef(g_pred, g_true)[1, 0]
    snip_mae = mean_absolute_error(g_true, g_pred)

    emd = df_slice.groupby("instance_id").apply(
        lambda row: wasserstein_distance(row["pred"].tolist(), row["scaled_rating"].tolist())
    ).mean()

    return indiv_r, indiv_mae, snip_r, snip_mae, emd


def compute_score(seen_metrics, unseen_metrics, cfg: dict) -> float:
    """Composite score used by Optuna. Returns a value consistent with cfg['direction']."""
    sw, uw = cfg["score_weights"]
    if cfg["score_metric"] == "mae":
        return sw * seen_metrics[1] + uw * unseen_metrics[1]   # lower is better
    else:
        return sw * seen_metrics[2] + uw * unseen_metrics[2]   # higher is better


def demographic_specialization(df: pd.DataFrame, identity_col: str,
                                topk_col: str = "topk_expert_0", eps: float = 1e-8) -> float:
    """Mean pairwise symmetric KL divergence over expert distributions of demographic groups."""
    dist = (
        df.groupby(identity_col)[topk_col]
          .value_counts(normalize=True)
          .unstack(fill_value=0)
    )
    if dist.shape[0] < 2:
        return 0.0
    groups    = dist.index.tolist()
    kl_values = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            p = dist.loc[groups[i]].values + eps
            q = dist.loc[groups[j]].values + eps
            p /= p.sum(); q /= q.sum()
            kl_values.append(sum(rel_entr(p, q)) + sum(rel_entr(q, p)))
    return float(np.mean(kl_values))


def log_expert_usage(df: pd.DataFrame, group_name: str, epoch: int,
                     model: nn.Module, logger):
    expert_cols = [c for c in df.columns if c.startswith("topk_expert_")]
    count  = Counter(df[expert_cols].to_numpy().flatten().tolist())
    header = f"\n[EPOCH {epoch+1}] [{group_name}] Expert-usage counts:"
    print(header); logger.info(header)
    for i in range(core(model).num_experts):
        line = f"  Expert {i:2d}: {count.get(i, 0)}"
        print(line); logger.info(line)


def log_expert_specialization(df: pd.DataFrame, group_name: str, epoch: int,
                               model: nn.Module, logger):
    n_tokens = len(df)
    if n_tokens == 0:
        msg = f"\n[EPOCH {epoch+1}] [{group_name}] No tokens for specialization."
        print(msg); logger.info(msg)
        return
    expert_cols = [c for c in df.columns if c.startswith("topk_expert_")]
    primary_cnt = Counter(df["topk_expert_0"].tolist())
    any_cnt     = Counter(df[expert_cols].to_numpy().flatten().tolist())
    header = (f"\n[EPOCH {epoch+1}] [{group_name}] Expert specialization "
              f"(lower any_ratio => more specialized):")
    print(header); logger.info(header)
    for j in range(core(model).num_experts):
        p, a = primary_cnt.get(j, 0), any_cnt.get(j, 0)
        line = (f"  Expert {j:2d}: primary={p:4d} ({p/n_tokens:.2%})  "
                f"any={a:4d} ({a/n_tokens:.2%})")
        print(line); logger.info(line)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(cfg: dict):
    """Load CSVs, encode IDs, build DataLoaders. Returns a data bundle dict."""
    train_df  = pd.read_csv(cfg["train_csv"])
    test_df   = pd.read_csv(cfg["test_csv"])
    dev_df    = pd.read_csv(cfg["dev_csv"])
    concat_og = pd.concat([train_df, test_df, dev_df])

    demo_feats = cfg["demo_feats"]
    target_col = "scaled_rating"

    num_identity_cats = [concat_og[k].nunique() for k in demo_feats]

    # Annotator IDs → contiguous integers
    ann_enc = LabelEncoder().fit(concat_og["annotator_id"])
    def _ann_ids(df):
        return torch.tensor(ann_enc.transform(df["annotator_id"]), dtype=torch.long)

    train_ann = _ann_ids(train_df)
    test_ann  = _ann_ids(test_df)
    dev_ann   = _ann_ids(dev_df)
    annotator_size = len(ann_enc.classes_)

    # Demographic feature IDs
    def _id_tensor(df):
        parts = []
        for feat in demo_feats:
            enc = LabelEncoder().fit(concat_og[feat])
            parts.append(torch.tensor(enc.transform(df[feat]), dtype=torch.long))
        return torch.stack(parts, dim=1)

    train_id = _id_tensor(train_df)
    test_id  = _id_tensor(test_df)
    dev_id   = _id_tensor(dev_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _target(df):
        return torch.tensor(df[target_col].tolist(), dtype=torch.float32).unsqueeze(1).to(device)

    y_train = _target(train_df)
    y_test  = _target(test_df)
    y_dev   = _target(dev_df)

    # Pre-extracted text features (ModernBERT-large [CLS])
    feats = torch.load(cfg["features_pt"])
    train_txt = feats["train"]
    test_txt  = feats["test"]
    dev_txt   = feats["dev"]

    bs = cfg["batch_size"]
    dl_train = DataLoader(TensorDataset(train_txt, train_ann, train_id, y_train),
                          batch_size=bs, shuffle=True)
    dl_test  = DataLoader(TensorDataset(test_txt,  test_ann,  test_id,  y_test),
                          batch_size=bs, shuffle=False)
    dl_dev   = DataLoader(TensorDataset(dev_txt,   dev_ann,   dev_id,   y_dev),
                          batch_size=bs, shuffle=False)

    seen_annotators     = set(train_df["annotator_id"])
    dev_df["seen_flag"] = dev_df["annotator_id"].isin(seen_annotators)

    return {
        "dl_train":          dl_train,
        "dl_test":           dl_test,
        "dl_dev":            dl_dev,
        "train_df":          train_df,
        "test_df":           test_df,
        "dev_df":            dev_df,
        "annotator_size":    annotator_size,
        "num_identity_cats": num_identity_cats,
        "device":            device,
        "dem_group_dict":    {col: list(dev_df[col].unique()) for col in demo_feats},
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_model(model, dataloader, optimizer, criterion, device,
                scheduler, epoch, phase, sched_params, demo_feats, logger):
    model.train()
    total_loss  = 0.0
    prev_counts = None

    load_w     = sched_params["load_w"]
    cap_mult   = sched_params["cap_mult"]
    kl_id_w    = sched_params["kl_identity_w"]
    ann_w      = sched_params["annot_weight"]
    orth_w     = sched_params["orth_w"]
    var_w      = sched_params["var_w"]
    demog_kl_w = sched_params.get("demog_kl_w", 0.0)

    for batch in dataloader:
        if prev_counts is not None:
            with torch.no_grad():
                biases, eta = get_biases(model)
                step_b = eta * (prev_counts.mean() - prev_counts) / prev_counts.size(0)
                if phase == 0 and 10 <= epoch < 20:
                    step_b *= 3
                biases.add_(step_b).clamp_(-2.5, 2.5)

        text_embeds, annot_ids, id_feats, targets = [x.to(device) for x in batch]
        out, kl_ann, kl_id, topk_idx, topk_scr, softmax_scores, _, per_expert = model(
            text_embeds, annot_ids, id_feats, epoch, return_expert_ids=True
        )

        with torch.no_grad():
            n_exp  = core(model).num_experts
            counts = torch.zeros(n_exp, device=topk_idx.device)
            for j in range(core(model).top_k_experts):
                counts.scatter_add_(0, topk_idx[:, j], topk_scr[:, j])

        load_std = (counts / counts.mean()).clamp(max=1).std()
        entropy  = -(softmax_scores * (softmax_scores + 1e-9).log()).sum(-1).mean()

        # Orthogonality loss L_o (Eq. 6)
        B   = per_expert.shape[0]
        eps = 1e-9
        L_o = torch.tensor(0.0, device=per_expert.device)
        for i in range(B):
            Xi   = per_expert[i]
            idxs = torch.nonzero(Xi.norm(dim=1) > 0.0, as_tuple=False).view(-1)
            for idx_j in idxs:
                for idx_k in idxs:
                    if idx_k == idx_j:
                        continue
                    L_o += (Xi[idx_j] * Xi[idx_k]).sum() / (Xi[idx_k] * Xi[idx_k]).sum().clamp(min=eps)

        # Variance loss L_v (Eq. 7)
        S_mask = torch.zeros_like(softmax_scores)
        for i in range(B):
            for k in range(core(model).top_k_experts):
                S_mask[i, topk_idx[i, k]] = topk_scr[i, k]
        diff = S_mask - S_mask.mean(dim=0, keepdim=True)
        L_v  = -(diff * diff).sum() / (B * n_exp)

        # Demographic routing KL loss (optional)
        demog_kl = torch.tensor(0.0, device=softmax_scores.device)
        if demog_kl_w > 0:
            for dim in range(len(demo_feats)):
                group_dists = []
                for val in torch.unique(id_feats[:, dim]):
                    mask = id_feats[:, dim] == val
                    if mask.sum() < 2:
                        continue
                    d = softmax_scores[mask].mean(dim=0) + 1e-9
                    group_dists.append(d / d.sum())
                for i in range(len(group_dists)):
                    for j in range(i + 1, len(group_dists)):
                        p, q = group_dists[i], group_dists[j]
                        demog_kl += ((p * (p / q).log()) + (q * (q / p).log())).sum() / 2

        loss = (
            criterion(out, targets)
            + kl_id_w    * kl_id.mean()
            + ann_w      * kl_ann.mean()
            + load_w     * load_std
            + orth_w     * L_o
            + var_w      * L_v
            + demog_kl_w * demog_kl
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(core(model).parameters(),                 0.7)
        nn.utils.clip_grad_norm_(core(model).expert_selector.parameters(), 0.5)
        optimizer.step()
        renorm_selector(model, max_norm=cap_mult)
        scheduler.step()

        total_loss  += loss.item()
        prev_counts  = counts

    return total_loss / len(dataloader), load_std.item(), entropy.item()


def evaluate_model(model, dataloader, device):
    core_model = core(model)
    core_model.eval()
    all_preds, all_labels, all_topk = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            text_embeds, annot_ids, id_feats, targets = [x.to(device) for x in batch]
            out, _, _, topk_idx, _, _, _, _ = core_model(
                text_embeds, annot_ids, id_feats, 0, True
            )
            all_preds.append(out.cpu())
            all_labels.append(targets.cpu())
            all_topk.append(topk_idx.cpu())
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_topk).numpy(),
    )


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(cfg: dict, data: dict, logger):
    """Returns a closure so objective() can access cfg and data without globals."""

    best_state = {"score": float("inf") if cfg["direction"] == "minimize" else -float("inf"),
                  "trial": None}

    def is_better(new, old):
        return new < old if cfg["direction"] == "minimize" else new > old

    def objective(trial: optuna.Trial) -> float:
        params = cfg["params"].copy()

        # Hyperparameter search space (narrow refinement around best known values)
        params["T_init"]        = trial.suggest_float("T_init",        0.8 * params["T_init"],  1.2 * params["T_init"])
        params["T_decay"]       = trial.suggest_float("T_decay",       max(0.5, params["T_decay"] - 0.05),
                                                                        min(0.99, params["T_decay"] + 0.05))
        params["T_min"]         = trial.suggest_float("T_min",         0.5 * params["T_min"],   2.0 * params["T_min"])
        params["kl_identity_w"] = trial.suggest_float("kl_identity_w", 0.5 * params["kl_identity_w"],
                                                                        2.0 * params["kl_identity_w"])
        params["annot_weight"]  = trial.suggest_float("annot_weight",  0.5 * params["annot_weight"],
                                                                        2.0 * params["annot_weight"])
        params["load_w_C"]      = trial.suggest_float("load_w_C",      0.8 * params["load_w_C"],  1.2 * params["load_w_C"])
        params["orth_w_C"]      = trial.suggest_float("orth_w_C",      0.8 * params["orth_w_C"],  1.2 * params["orth_w_C"])
        params["var_w_C"]       = trial.suggest_float("var_w_C",       0.8 * params["var_w_C"],   1.2 * params["var_w_C"])
        if params.get("demog_kl_w", 0) > 0:
            params["demog_kl_w"] = trial.suggest_float("demog_kl_w",
                                                        0.5 * params["demog_kl_w"],
                                                        2.0 * params["demog_kl_w"])
        params["encoder_bump"]  = trial.suggest_categorical("encoder_bump", [1.4, 1.5, 1.6, 2.0])

        n_gpus  = torch.cuda.device_count()
        device  = torch.device(f"cuda:{trial.number % max(1, n_gpus)}" if n_gpus else "cpu")

        model = TopKMoE(
            input_dim          = 1408,
            hidden_dim         = 128,
            num_experts        = cfg["num_experts"],
            text_embed_size    = 1024,
            annotator_size     = data["annotator_size"],
            num_identity_cats  = data["num_identity_cats"],
            top_k_experts      = params["topk_exp"],
            bias_lr            = params["bias_lr"],
            T_init             = params["T_init"],
            T_decay            = params["T_decay"],
            T_min              = params["T_min"],
            T_warmup_epochs    = params["T_warmup_epochs"],
            overflow_on        = params["overflow_on"],
            capacity_scale     = params["capacity_scale"],
        ).to(device)

        gate_params  = list(core(model).expert_selector.parameters())
        other_params = [p for n, p in core(model).named_parameters()
                        if not n.startswith("expert_selector")]
        optimizer = optim.AdamW([
            {"params": gate_params,  "lr": params["lr_gate"], "weight_decay": 0.05, "tag": "gate"},
            {"params": other_params, "lr": params["lr_main"], "weight_decay": 0.0,  "tag": "encoder"},
        ])

        num_epochs  = cfg["num_epochs"]
        total_steps = len(data["dl_train"]) * num_epochs
        scheduler   = get_linear_schedule_with_warmup(optimizer, int(0.3 * total_steps), total_steps)
        criterion   = nn.MSELoss()

        # Phase-adaptive loss weight schedule
        params["load_w"] = params["load_w_A"]
        params["orth_w"] = params["orth_w_A"]
        params["var_w"]  = params["var_w_A"]

        PHASE_CFG = {
            0: dict(target=dict(load=params["load_w_B"], orth=params["orth_w_B"],
                                var=params["var_w_B"],   cap=params["cap_mult_B"]), alpha=0.15),
            1: dict(target=dict(load=params["load_w_C"], orth=params["orth_w_C"],
                                var=params["var_w_C"],   cap=params["cap_mult_C"]), alpha=0.1),
            2: dict(target=dict(load=params["load_w_C"], orth=params["orth_w_C"],
                                var=params["var_w_C"],   cap=params["cap_mult_C"]), alpha=0.0),
        }
        w = dict(load=params["load_w_A"], orth=params["orth_w_A"],
                 var=params["var_w_A"],   cap=params["cap_mult_A"])

        current_phase    = 0
        early_stop_phase = EarlyStopping(patience=5)
        early_stopping   = EarlyStopping(patience=5)
        encoder_bumped   = False
        score            = best_state["score"]  # safe default
        dev_df           = data["dev_df"]
        test_df          = data["test_df"]

        logger.info(f"[Trial {trial.number:02d}] PARAMS={params}")

        for epoch in tqdm(range(num_epochs), desc=f"Trial {trial.number}"):
            logger.info(
                f"[Trial {trial.number:02d}][Epoch {epoch:02d}] "
                f"phase={current_phase}  top_k={core(model).top_k_experts}  "
                f"load_w={params['load_w']:.3f}  orth_w={params['orth_w']:.3f}  var_w={params['var_w']:.3f}"
            )

            loss, load_std, entropy = train_model(
                model, data["dl_train"], optimizer, criterion, device,
                scheduler, epoch, current_phase, params, cfg["demo_feats"], logger,
            )

            preds, _, gate_topk = evaluate_model(model, data["dl_dev"], device)
            dev_df = dev_df.copy()
            dev_df["pred"] = preds
            for k in range(core(model).top_k_experts):
                dev_df[f"topk_expert_{k}"] = gate_topk[:, k]

            seen_m   = slice_metrics(dev_df[dev_df["seen_flag"]])
            unseen_m = slice_metrics(dev_df[~dev_df["seen_flag"]])
            seen_r,   seen_mae   = seen_m[2],   seen_m[1]
            unseen_r, unseen_mae = unseen_m[2], unseen_m[1]

            token_hist = np.bincount(gate_topk[:, 0], minlength=cfg["num_experts"])
            max_share  = (token_hist / token_hist.sum()).max()

            # Phase transitions
            if current_phase == 0 and (entropy < 1.3 or load_std > 0.18):
                current_phase = 1
            if current_phase == 1:
                early_stop_phase(unseen_r)
                if early_stop_phase.early_stop or max_share > 0.80:
                    current_phase = 2
            if current_phase == 2 and not encoder_bumped:
                for p in core(model).expert_selector.parameters():
                    p.requires_grad_(False)
                for g in optimizer.param_groups:
                    if g.get("tag") == "encoder":
                        g["lr"] *= params["encoder_bump"]
                encoder_bumped = True

            cfg_ph = PHASE_CFG[current_phase]
            for key in ["load", "orth", "var", "cap"]:
                w[key] += cfg_ph["alpha"] * (cfg_ph["target"][key] - w[key])
            params["load_w"]   = w["load"]
            params["orth_w"]   = w["orth"]
            params["var_w"]    = w["var"]
            params["cap_mult"] = w["cap"]

            # Specialisation logging
            log_expert_usage(dev_df[dev_df["seen_flag"]],   "Seen",   epoch, model, logger)
            log_expert_usage(dev_df[~dev_df["seen_flag"]],  "Unseen", epoch, model, logger)
            log_expert_specialization(dev_df[dev_df["seen_flag"]],   "Seen",   epoch, model, logger)
            log_expert_specialization(dev_df[~dev_df["seen_flag"]],  "Unseen", epoch, model, logger)

            ds_seen   = {c: demographic_specialization(dev_df[dev_df["seen_flag"]],  c)
                         for c in cfg["demo_feats"]}
            ds_unseen = {c: demographic_specialization(dev_df[~dev_df["seen_flag"]], c)
                         for c in cfg["demo_feats"]}
            logger.info(
                f"[Trial {trial.number:02d}][Epoch {epoch:02d}] phase={current_phase} "
                f"entropy={entropy:.3f} load_std={load_std:.3f} "
                f"SEEN R={seen_r:.4f} UNSEEN R={unseen_r:.4f} "
                f"SEEN MAE={seen_mae:.4f} UNSEEN MAE={unseen_mae:.4f} "
                f"DEM_SPEC_SEEN={ds_seen} DEM_SPEC_UNSEEN={ds_unseen}"
            )

            # Per-demographic MAE on test set
            test_preds, _, test_topk = evaluate_model(model, data["dl_test"], device)
            test_df = test_df.copy()
            test_df["pred"] = test_preds
            for k in range(test_topk.shape[1]):
                test_df[f"topk_expert_{k}"] = test_topk[:, k]

            logger.info(f"[Trial {trial.number:02d}][Epoch {epoch:02d}] === Demographic Group MAE ===")
            for dem_cat in cfg["demo_feats"]:
                vals = [
                    mean_absolute_error(test_df.loc[test_df[dem_cat] == v, "scaled_rating"],
                                        test_df.loc[test_df[dem_cat] == v, "pred"])
                    for v in data["dem_group_dict"].get(dem_cat, [])
                    if (test_df[dem_cat] == v).sum() > 0
                ]
                if len(vals) > 1:
                    mean_val     = np.mean(vals)
                    ci_lo, ci_hi = st.t.interval(0.95, len(vals) - 1, loc=mean_val, scale=st.sem(vals))
                    msg = f"  {dem_cat}: mean={mean_val:.3f}  ci=({ci_lo:.3f}, {ci_hi:.3f})"
                    if cfg["baseline_mae"] and dem_cat in cfg["baseline_mae"]:
                        diff   = cfg["baseline_mae"][dem_cat] - mean_val
                        status = "+" if diff > 0 else "-"
                        msg   += f"  vs baseline {cfg['baseline_mae'][dem_cat]:.3f}  delta={diff:.3f} {status}"
                else:
                    msg = f"  {dem_cat}: insufficient data."
                print(msg); logger.info(msg)

            score = compute_score(seen_m, unseen_m, cfg)
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if epoch > 10:
                early_stopping(score if cfg["direction"] == "minimize" else -score)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

        # Save best model
        if is_better(score, best_state["score"]):
            best_state["score"] = score
            best_state["trial"] = trial.number
            tag = cfg.get("dataset_name", "dataset")
            torch.save(model.state_dict(),
                       f"moe_{tag}_trial{trial.number:02d}_score{score:.4f}.pt")
            final_preds, _, final_topk = evaluate_model(model, data["dl_test"], device)
            test_df["pred"] = final_preds
            for k in range(final_topk.shape[1]):
                test_df[f"topk_expert_{k}"] = final_topk[:, k]
            test_df.to_csv(f"moe_{tag}_trial{trial.number:02d}_score{score:.4f}_test_df.csv",
                           index=False)

        logger.info(
            f"[Trial {trial.number:02d}] Final score={score:.4f} "
            f"(seen_r={seen_r:.4f}, unseen_r={unseen_r:.4f}, "
            f"seen_mae={seen_mae:.4f}, unseen_mae={unseen_mae:.4f})"
        )
        return score

    return objective


# ── Entry point ───────────────────────────────────────────────────────────────

def setup_logging(dataset_name: str):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    loggers = {}
    for name, fname, level in [
        ("output",  f"training_output_{dataset_name}.log",      logging.INFO),
        ("vanish",  f"vanishing_gradients_{dataset_name}.log",  logging.INFO),
        ("error",   f"errors_{dataset_name}.log",               logging.ERROR),
    ]:
        lg = logging.getLogger(f"moe_{name}")
        lg.setLevel(level)
        fh = logging.FileHandler(fname)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        lg.addHandler(fh)
        loggers[name] = lg
    return loggers["output"]


def main():
    parser = argparse.ArgumentParser(description="Train Top-K MoE for subjective NLP tasks")
    parser.add_argument("--dataset",  required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to train on")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--n_jobs",   type=int, default=8,
                        help="Parallel Optuna trials (set to 1 for debugging)")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset].copy()
    cfg["dataset_name"] = args.dataset

    logger = setup_logging(args.dataset)
    logger.info(f"Starting training for dataset={args.dataset}")

    data = load_data(cfg)

    objective = make_objective(cfg, data, logger)

    _pbar = tqdm(total=args.n_trials, desc="Optuna trials")
    def _cb(study, trial):
        _pbar.update(1)

    study = optuna.create_study(direction=cfg["direction"])
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, callbacks=[_cb])
    _pbar.close()

    print(f"\nDone! Best {cfg['direction']} score: {study.best_value:.4f}")
    print("Best params:\n", study.best_params)


if __name__ == "__main__":
    main()
