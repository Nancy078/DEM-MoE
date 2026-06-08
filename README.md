# DEM-MoE: Demographic-Aware Mixture of Experts

DEM-MoE routes text representations to expert subnetworks conditioned on annotator demographic identity, enabling structured modeling of group-level annotation disagreement.

**Paper:** [https://arxiv.org/pdf/2508.02853](https://arxiv.org/pdf/2508.02853)

---

## Overview

Subjective NLP tasks — toxicity detection, harm assessment, politeness rating — yield different annotations from different demographic groups. DEM-MoE captures this structure by:

1. **Bayesian annotator and identity embeddings** (reparameterization trick + KL regularization) that encode individual annotators and their demographic attributes.
2. **Top-K Mixture of Experts** with Gumbel-softmax gating and temperature annealing, routing each (text, annotator) pair to the most appropriate expert subnetwork.
3. **Three-phase training** (A → B → C) with progressive orthogonality and variance losses that encourage expert specialization.
4. **Demographic KL regularization** that penalizes routing disparity across demographic groups.

---

## Datasets

| Key      | Task                              | Target column     | Scale                  |
|----------|-----------------------------------|-------------------|------------------------|
| `harm`   | Chatbot harm assessment           | `rating`          | {1, 2, 3}              |
| `toxic`  | Social media toxicity             | `toxic_score`     | {1, 2, 3, 4, 5}        |
| `paact`  | Patient-centered communication    | `pcc` / `scaled_rating` | continuous / z-score |
| `polite` | Email politeness                  | `politeness`      | {1, 2, 3, 4, 5}        |
| `off`    | Reddit offensiveness              | `offensiveness`   | {1, 2, 3, 4, 5}        |

All datasets require pre-split CSVs named `<dataset>_train_df.csv`, `<dataset>_dev_df.csv`, `<dataset>_test_df.csv` (exact names listed in each training script's `DATASET_CONFIGS`).

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and CUDA-capable GPU(s). The MoE model uses `answerdotai/ModernBERT-large` (≈ 400 M parameters); the LoRA baseline requires `meta-llama/Llama-3.1-8B-Instruct` with a HuggingFace access token.

---

## Quickstart

### 1. Extract text features (required for MoE only)

`train_moe.py` reads pre-extracted [CLS] embeddings from `.pt` files. Generate them once per dataset:

```bash
python extract_features.py --dataset harm
python extract_features.py --dataset toxic
python extract_features.py --dataset paact
python extract_features.py --dataset polite
python extract_features.py --dataset off
```

Output files are written to the paths in `DATASET_CONFIGS["output_pt"]` (e.g. `HARM_MOE_FEATS_NEW.pt`).

### 2. Train

```bash
# DEM-MoE (main model)
python train_moe.py   --dataset harm

# Baselines
python train_uwua.py  --dataset harm
python train_jury.py  --dataset harm
python train_mbert.py --dataset harm
python train_lora.py  --dataset harm
```

All scripts accept `--dataset {harm,toxic,paact,polite,off}` and an optional `--num_epochs` override.

The LoRA script additionally accepts:
```
--model_dir   path/to/Llama-3.1-8B-Instruct   (default: HuggingFace hub ID)
--cache_dir   path/to/weight/cache
```

### 3. Evaluate

Each training script saves a prediction CSV (e.g. `harm_moe_test_preds.csv`) containing `pred`, `scaled_rating`, `instance_id`, and `seen_flag`. Pass it to `evaluate.py`:

```bash
python evaluate.py --pred_csv harm_moe_test_preds.csv
python evaluate.py --pred_csv harm_moe_test_preds.csv --n_bootstrap 1000
```

Reported metrics (separately for seen and unseen annotators):

| Metric        | Description                                        |
|---------------|----------------------------------------------------|
| `indiv_corr`  | Pearson *r* between individual predictions & labels |
| `indiv_mae`   | MAE on individual ratings                          |
| `snip_corr`   | Pearson *r* on per-snippet mean predictions         |
| `snip_mae`    | MAE on per-snippet means                           |
| `emd`         | Mean Earth Mover's Distance per snippet            |

#### Demographic subgroup analysis

Pass `--demo_cols` to additionally compute per-attribute `snip_agg_mae` with 95% CI:

```bash
python evaluate.py --pred_csv paact_moe_test_preds.csv \
    --demo_cols "hcp_freq,edu_level,age_group,gender,race,occupation,doc_trust_category,ethnic_trust_category"
```

For each attribute, the script independently bootstraps every subgroup (e.g. Man / Woman / Non-binary for `gender`), pools all bootstrap `snip_agg_mae` values across subgroups, and reports a Student's-t 95% CI on the pool:

```
  DEMOGRAPHIC SUBGROUP ANALYSIS  (snip_agg_mae)
  ────────────────────────────────────────────────────────────
  Category                    mean   95% CI              error
  ──────────────────────────────────────────────────────────────────
  hcp_freq                   0.735  (0.729, 0.741)        0.006
  edu_level                  0.788  (0.779, 0.797)        0.009
  age_group                  0.740  (0.735, 0.746)        0.005
  gender                     0.887  (0.871, 0.902)        0.016
  race                       0.760  (0.756, 0.764)        0.004
  ...
```

The demographic columns must be present in the prediction CSV; all training scripts preserve the original CSV columns alongside `pred`.

---

## Repository Structure

```
DEM-MoE/
├── models.py             # All model classes
│   ├── BayesianAnnotatorEmbedding
│   ├── BayesianIdentityEmbedding
│   ├── TopKMoE            (main model)
│   ├── AnnotatorAwareRegressor  (UWUA baseline)
│   ├── JuryDataset / JuryRegressor  (Jury baseline)
│   ├── BERTModel          (mBERT baseline)
│   └── EarlyStopping / EarlyStoppingCorr
├── extract_features.py   # Extract ModernBERT-large [CLS] embeddings → .pt
├── train_moe.py          # Train DEM-MoE
├── train_uwua.py         # Train UWUA baseline
├── train_jury.py         # Train Jury (DCN) baseline
├── train_mbert.py        # Train mBERT regression baseline
├── train_lora.py         # Train LoRA (Llama-3.1-8B) baseline
├── evaluate.py           # Bootstrap CI evaluation
└── requirements.txt
```

---

## Model Architecture

### DEM-MoE (`TopKMoE`)

```
text [CLS] emb  ──┐
annotator emb   ──┼──► gating net ──► top-K softmax ──► weighted sum of K expert MLPs ──► regression head
identity emb    ──┘
```

- **Gating**: linear projection → Gumbel-softmax with annealing temperature *T(t) = T₀ · decay^t*
- **Experts**: independent MLPs (hidden → hidden → 1)
- **Training phases**:
  - Phase A: load-balancing loss L_load only
  - Phase B: add orthogonality L_o (Eq. 6) and variance L_v (Eq. 7)
  - Phase C: encoder fine-tuning with frozen gating
- **Expert bias**: per-expert learnable scalar updated proportional to token imbalance (separate `bias_lr`)

### Baselines

| Model  | Key idea                                                               |
|--------|------------------------------------------------------------------------|
| UWUA   | Annotator (Ea) + label-history (En) embeddings injected into BERT input |
| Jury   | ModernBERT + annotator emb + one-hot survey features → Deep & Cross Network |
| mBERT  | Plain ModernBERT-large fine-tuned with a linear regression head        |
| LoRA   | Llama-3.1-8B-Instruct with LoRA adapters; demographic-conditioned prompt |

---

## Citation

```bibtex
@misc{xu2025modelingannotatordisagreementdemographicaware,
      title={Modeling Annotator Disagreement with Demographic-Aware Experts and Synthetic Perspectives}, 
      author={Yinuo Xu and Veronica Derricks and Allison Earl and David Jurgens},
      year={2025},
      eprint={2508.02853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.02853}, 
}
```
