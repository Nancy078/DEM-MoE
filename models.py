"""
Model definitions for DEM-MoE and baselines.

Classes
-------
MoE model (main contribution):
    BayesianAnnotatorEmbedding  – variational embedding for annotator IDs
    BayesianIdentityEmbedding   – per-attribute variational identity embeddings
    TopKMoE                     – Top-K MoE with Gumbel gating & capacity routing

Baselines:
    AnnotatorAwareRegressor     – UWUA: annotator-aware ModernBERT regressor
    JuryDataset                 – Dataset class for the Jury baseline
    CrossLayer / CrossNetwork   – DCN-style cross network used by JuryRegressor
    JuryRegressor               – Jury: ModernBERT + demographic survey + cross-net
    BERTModel                   – mBERT: plain ModernBERT fine-tuned for regression

Shared utilities:
    EarlyStopping               – stops when validation loss stops decreasing
    EarlyStoppingCorr           – stops when validation correlation stops increasing
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ModernBertModel


# ── MoE model ─────────────────────────────────────────────────────────────────

class BayesianAnnotatorEmbedding(nn.Module):
    """
    Diagonal-Gaussian variational embedding for a single categorical variable
    (annotator IDs or a single demographic attribute).

    Each category gets a learnable mean (mu) and log-variance (log_var).
    At forward time a sample is drawn via the reparameterisation trick and the
    KL divergence against a standard-normal prior is returned for regularisation.
    """

    def __init__(self, num_categories: int, embedding_dim: int):
        super().__init__()
        self.mu      = nn.Embedding(num_categories, embedding_dim)
        self.log_var = nn.Embedding(num_categories, embedding_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.log_var.weight)

    def forward(self, ids: torch.Tensor):
        """
        Args:
            ids: LongTensor of shape (B,) with category indices.
        Returns:
            z       : sampled embedding (B, embedding_dim)
            kl_loss : scalar KL divergence term for the batch
        """
        mu      = self.mu(ids)
        log_var = self.log_var(ids)
        std     = torch.exp(0.5 * log_var)
        z       = mu + torch.randn_like(std) * std   # reparameterisation
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1
        ).mean()
        return z, kl_loss


class BayesianIdentityEmbedding(nn.Module):
    """
    Applies one BayesianAnnotatorEmbedding per demographic attribute and
    concatenates the samples into a single identity vector.

    Args:
        num_categories_list : list of vocabulary sizes, one per attribute
        embedding_dim       : embedding size for each attribute
    """

    def __init__(self, num_categories_list: list[int], embedding_dim: int):
        super().__init__()
        self.embedding_layers = nn.ModuleList([
            BayesianAnnotatorEmbedding(n, embedding_dim)
            for n in num_categories_list
        ])

    def forward(self, identity_ids: torch.Tensor):
        """
        Args:
            identity_ids: LongTensor (B, num_attributes)
        Returns:
            concatenated embedding (B, embedding_dim * num_attributes)
            sum of KL losses across attributes (scalar)
        """
        embeddings, kl_losses = [], []
        for i, layer in enumerate(self.embedding_layers):
            z, kl = layer(identity_ids[:, i])
            embeddings.append(z)
            kl_losses.append(kl)
        return torch.cat(embeddings, dim=1), torch.stack(kl_losses).sum()


class TopKMoE(nn.Module):
    """
    Top-K Mixture-of-Experts for annotator-rating prediction.

    Architecture
    ------------
    Input:  ModernBERT-large [CLS] embedding  (text_embed_size)
          + Bayesian annotator embedding       (hidden_dim)
          + Bayesian identity embedding        (projected to 256)
    Gating: Gumbel-softmax with temperature annealing; top-k selection with
            per-expert load-balancing biases and optional capacity overflow.
    Experts: num_experts independent 2-layer ReLU MLPs.
    Output: scalar regression prediction.

    See train_moe.py for the phased loss-weight training schedule.
    """

    def __init__(
        self,
        input_dim: int,
        text_embed_size: int,
        annotator_size: int,
        num_identity_cats: list[int],
        hidden_dim: int          = 256,
        num_experts: int         = 10,
        identity_embed_size: int = 256,
        dropout_rate: float      = 0.4,
        top_k_experts: int       = 3,
        bias_lr: float           = 0.05,
        T_init: float            = 6.0,
        T_decay: float           = 0.92,
        T_min: float             = 4.0,
        T_warmup_epochs: int     = 25,
        overflow_on: bool        = True,
        capacity_scale: float    = 1.0,
    ):
        super().__init__()

        self.top_k_experts   = top_k_experts
        self.bias_lr         = bias_lr
        self.num_experts     = num_experts
        self.hidden_dim      = hidden_dim
        self.T_init          = T_init
        self.T_decay         = T_decay
        self.T_min           = T_min
        self.T_warmup_epochs = T_warmup_epochs
        self.overflow_on     = overflow_on
        self.capacity_scale  = capacity_scale

        # Per-expert load-balancing bias (updated outside the gradient graph)
        self.register_buffer("expert_biases", torch.zeros(num_experts, dtype=torch.float32))

        self.annotator_embedding = BayesianAnnotatorEmbedding(annotator_size, hidden_dim)
        self.identity_embedding  = BayesianIdentityEmbedding(num_identity_cats, identity_embed_size)

        self.text_layernorm      = nn.LayerNorm(text_embed_size)
        self.annotator_layernorm = nn.LayerNorm(hidden_dim)

        identity_concat_size    = identity_embed_size * len(num_identity_cats)
        projected_identity_size = 256
        self.identity_projection = nn.Linear(identity_concat_size, projected_identity_size)
        self.identity_layernorm  = nn.LayerNorm(projected_identity_size)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            ) for _ in range(num_experts)
        ])

        self.expert_selector = nn.Linear(input_dim, num_experts)
        nn.init.normal_(self.expert_selector.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.expert_selector.bias)

        self.output_layer   = nn.Linear(hidden_dim, 1)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        text_embeds:       torch.Tensor,
        annotator_ids:     torch.Tensor,
        identity_ids:      torch.Tensor,
        epoch:             int | None = None,
        return_expert_ids: bool       = False,
    ):
        """
        Args:
            text_embeds:       (B, text_embed_size) ModernBERT [CLS] embeddings
            annotator_ids:     (B,) encoded annotator indices
            identity_ids:      (B, num_attributes) encoded demographic indices
            epoch:             current training epoch (drives temperature annealing)
            return_expert_ids: if True, return full routing diagnostics

        Returns (compact):
            prediction, kl_ann, kl_id

        Returns (with return_expert_ids=True):
            prediction, kl_ann, kl_id, topk_indices, topk_scores,
            softmax_scores, expert_scores, per_expert_outputs
        """
        text_embeds = self.text_layernorm(text_embeds)

        ann_embeds, kl_ann = self.annotator_embedding(annotator_ids)
        ann_embeds = self.annotator_layernorm(ann_embeds)

        id_embeds, kl_id = self.identity_embedding(identity_ids)
        id_embeds = self.identity_projection(id_embeds)
        id_embeds = self.identity_layernorm(id_embeds)

        x = torch.cat([text_embeds, ann_embeds, id_embeds], dim=1)

        expert_scores = self.expert_selector(x) / 4 + self.expert_biases.unsqueeze(0)

        if self.training and epoch is not None:
            gumbel = -torch.empty_like(expert_scores).exponential_().log()
            if epoch < self.T_warmup_epochs:
                T = self.T_init
            else:
                T = max(self.T_min,
                        self.T_init * (self.T_decay ** (epoch - self.T_warmup_epochs)))
        else:
            T, gumbel = 1.0, 0

        softmax_scores            = torch.softmax((expert_scores + gumbel) / T, dim=-1)
        topk_scores, topk_indices = torch.topk(softmax_scores, self.top_k_experts, dim=-1)

        B           = x.shape[0]
        outputs     = torch.zeros(B, self.hidden_dim, device=x.device)
        per_expert  = x.new_zeros(B, self.num_experts, self.hidden_dim)

        primary_w   = topk_scores[:, 0]
        primary_idx = topk_indices[:, 0]
        capacity    = math.ceil(
            (text_embeds.size(0) / self.num_experts) * self.capacity_scale
        )

        if self.overflow_on:
            counts        = torch.bincount(primary_idx, minlength=self.num_experts)
            overflow_mask = torch.zeros_like(primary_w, dtype=torch.bool)
            for e in range(self.num_experts):
                extra = counts[e] - capacity
                if extra > 0:
                    pos      = torch.nonzero(primary_idx == e, as_tuple=False).flatten()
                    smallest = torch.topk(primary_w[pos], k=extra, largest=False).indices
                    overflow_mask[pos[smallest]] = True
            primary_w[overflow_mask] = 0.0
            topk_scores[:, 0]        = primary_w
            topk_scores = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-9)

        for i in range(B):
            for k in range(self.top_k_experts):
                eidx = topk_indices[i, k].item()
                gw   = topk_scores[i, k]
                h    = self.experts[eidx](x[i])
                per_expert[i, eidx]  = gw.unsqueeze(-1) * h
                outputs[i]          += gw.unsqueeze(-1) * h

        prediction = self.output_layer(self.output_dropout(outputs))

        if return_expert_ids:
            return (prediction, kl_ann, kl_id,
                    topk_indices, topk_scores, softmax_scores, expert_scores, per_expert)
        return prediction, kl_ann, kl_id


# ── UWUA baseline ──────────────────────────────────────────────────────────────

class AnnotatorAwareRegressor(nn.Module):
    """
    UWUA baseline: annotator-aware ModernBERT regressor.

    The model conditions each token's input embedding on two annotator signals
    before passing through ModernBERT:
      - Ea: a learnable annotator embedding
      - En: a running average of prior label embeddings for that annotator
            (leave-one-out during training; global mean for unseen annotators)

    Both signals are injected as soft attention biases on the input embeddings,
    allowing the backbone to attend differently depending on who is annotating.

    Args:
        bert_model:              pre-loaded ModernBertModel instance
        num_annotators:          vocabulary size for annotator IDs
        num_labels:              number of discrete rating levels; ignored when
                                 use_continuous_labels=True
        hidden_size:             dimension of annotator / label embeddings
        dropout_prob:            dropout applied before the regression head
        use_continuous_labels:   if True, encode each rating via a small MLP
                                 (used for datasets with continuous scores such as
                                 PAACT); if False, use a discrete nn.Embedding lookup
    """

    def __init__(
        self,
        bert_model,
        num_annotators: int,
        num_labels: int             = 0,
        hidden_size: int            = 1024,
        dropout_prob: float         = 0.1,
        use_continuous_labels: bool = False,
    ):
        super().__init__()
        self.bert                  = bert_model
        self.hidden_size           = hidden_size
        self.use_continuous_labels = use_continuous_labels

        self.annotator_embeddings = nn.Embedding(num_annotators, hidden_size)

        if use_continuous_labels:
            self.label_mlp = nn.Sequential(
                nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
            )
        else:
            self.label_embeddings = nn.Embedding(num_labels, hidden_size)

        # Running buffers for leave-one-out label embedding (updated each training step)
        self.register_buffer("sumEmb", torch.zeros(num_annotators, hidden_size))
        self.register_buffer("count",  torch.zeros(num_annotators, dtype=torch.long))

        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout   = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def _encode_label(self, annotation_ids: torch.Tensor) -> torch.Tensor:
        if self.use_continuous_labels:
            return self.label_mlp(annotation_ids.unsqueeze(-1).float())
        return self.label_embeddings(annotation_ids)

    def _label_global_mean(self, device: torch.device) -> torch.Tensor:
        if self.use_continuous_labels:
            return torch.zeros(1, self.hidden_size, device=device)
        return self.label_embeddings.weight.mean(dim=0, keepdim=True)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        annotator_ids:  torch.Tensor,
        annotation_ids: torch.Tensor,
        mode: str = "Ea+En",
    ):
        """
        Args:
            input_ids:      (B, T) tokenised text
            attention_mask: (B, T) padding mask
            annotator_ids:  (B,)   encoded annotator indices
            annotation_ids: (B,)   rating values – discrete int indices when
                            use_continuous_labels=False, float scores otherwise
            mode:           which signals to inject – "Ea", "En", or "Ea+En"
        Returns:
            scalar prediction (B,)
        """
        input_embs = self.bert.get_input_embeddings()(input_ids)   # (B, T, H_bert)

        # Mean-pool non-padding tokens → sentence-level summary Es
        mask_exp = attention_mask.unsqueeze(-1)
        Es = (input_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        Ea    = self.annotator_embeddings(annotator_ids)
        sum_i = self.sumEmb[annotator_ids]
        cnt_i = self.count[annotator_ids].unsqueeze(-1).float()
        E_lk  = self._encode_label(annotation_ids)

        if self.training:
            En = (sum_i - E_lk) / (cnt_i - 1).clamp(min=1.0)
            Ea = Ea + torch.randn_like(Ea) * 0.05   # light noise regularisation
        else:
            avg_emb     = sum_i / cnt_i.clamp(min=1.0)
            global_mean = self._label_global_mean(input_ids.device)
            En = torch.where(cnt_i == 0, global_mean, avg_emb)

        a_proj = self.Ws(Es)
        if mode in ("Ea", "Ea+En"):
            alpha_a    = (a_proj * self.Wa(Ea)).sum(-1, keepdim=True)
            input_embs = input_embs + alpha_a.unsqueeze(-1) * Ea.unsqueeze(1)
        if mode in ("En", "Ea+En"):
            alpha_n    = (a_proj * self.Wn(En)).sum(-1, keepdim=True)
            input_embs = input_embs + alpha_n.unsqueeze(-1) * En.unsqueeze(1)

        out = self.bert(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0]    # (B, H_bert)
        return self.regressor(self.dropout(cls)).squeeze(-1)


# ── Jury baseline ─────────────────────────────────────────────────────────────

class JuryDataset(Dataset):
    """
    PyTorch Dataset for the Jury baseline.

    Each item returns tokenised text, one-hot encoded demographic/survey features,
    the regression target, the instance ID, and the annotator ID.

    Args:
        dataframe:             DataFrame with at least columns
                               ['processed', 'scaled_rating', 'instance_id',
                                'encoded_annotator_id'] plus all survey_cols
        survey_cols:           list of one-hot demographic/survey column names
        max_token_len:         maximum token length for the backbone tokenizer
        tokenizer_name:        HuggingFace model name for the tokenizer
    """

    def __init__(
        self,
        dataframe,
        survey_cols: list[str],
        max_token_len: int = 1024,
        tokenizer_name: str = "answerdotai/ModernBERT-large",
    ):
        self.tokenizer    = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataframe    = dataframe
        self.survey_cols  = survey_cols
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        encoding = self.tokenizer.encode_plus(
            row["processed"],
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "survey_info":    torch.tensor(
                                  row[self.survey_cols].astype(float).to_numpy(),
                                  dtype=torch.float32),
            "targets":        torch.tensor(row["scaled_rating"], dtype=torch.float32),
            "instance_id":    row["instance_id"],
            "annotator_id":   torch.tensor(row["encoded_annotator_id"], dtype=torch.long),
        }


class CrossLayer(nn.Module):
    """Single layer of a Deep & Cross Network (DCN)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, 1))
        self.bias   = nn.Parameter(torch.empty(input_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x0: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return xi * torch.matmul(x0.unsqueeze(1), self.weight).squeeze(1) + self.bias


class CrossNetwork(nn.Module):
    """Stack of CrossLayer modules."""

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xi = x0
        for layer in self.cross_layers:
            xi = layer(x0, xi)
        return xi


class JuryRegressor(nn.Module):
    """
    Jury baseline: ModernBERT + annotator embedding + one-hot demographic/survey
    features, fused through a Deep & Cross Network.

    Args:
        annotator_size:   number of unique annotators
        survey_size:      dimensionality of the one-hot survey/demographic vector
        num_targets:      output dimensionality (1 for regression)
        num_cross_layers: depth of the DCN cross network
        dropout_rate:     dropout probability in the deep branch
        backbone_name:    HuggingFace model name for the text backbone
    """

    def __init__(
        self,
        annotator_size: int  = 598,
        survey_size: int     = 35,
        num_targets: int     = 1,
        num_cross_layers: int = 5,
        dropout_rate: float  = 0.2,
        backbone_name: str   = "answerdotai/ModernBERT-large",
    ):
        super().__init__()
        self.survey_size = survey_size

        self.backbone           = ModernBertModel.from_pretrained(backbone_name)
        backbone_hidden         = self.backbone.config.hidden_size  # 1024

        self.survey_ff          = nn.Linear(survey_size, 128)
        self.annotator_embedding = nn.Embedding(annotator_size, 128)

        combined_dim = backbone_hidden + 128 + 128   # text + annotator + survey
        self.cross_net = CrossNetwork(combined_dim, num_cross_layers)
        self.deep = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(combined_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),          nn.LayerNorm(256), nn.GELU(),
        )
        self.regressor = nn.Linear(256 + combined_dim, num_targets)

    @torch._dynamo.disable
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        survey_info:    torch.Tensor,
        annotator_ids:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, T)
            attention_mask: (B, T)
            survey_info:    (B, survey_size) one-hot demographic features
            annotator_ids:  (B,)
        Returns:
            predictions:    (B, num_targets)
        """
        content = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]                              # (B, 1024)

        ann_emb  = self.annotator_embedding(annotator_ids)    # (B, 128)
        surv_emb = self.survey_ff(
            survey_info.view(-1, self.survey_size)             # (B, 128)
        )

        combined    = torch.cat([content, ann_emb, surv_emb], dim=1)
        cross_out   = self.cross_net(combined)
        deep_out    = self.deep(combined)
        merged      = torch.cat([deep_out, cross_out], dim=1)
        return self.regressor(merged)


# ── mBERT baseline ────────────────────────────────────────────────────────────

class BERTModel(nn.Module):
    """
    mBERT baseline: ModernBERT-large fine-tuned for regression.

    A single linear head is placed on top of the [CLS] token representation.
    No annotator information is used; this is the text-only lower bound.

    Args:
        model_name: HuggingFace model identifier for the backbone
    """

    def __init__(self, model_name: str = "answerdotai/ModernBERT-large"):
        super().__init__()
        self.bert   = ModernBertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids:      (B, T)
            attention_mask: (B, T)
        Returns:
            predictions:    (B, 1)
        """
        cls = self.bert(input_ids, attention_mask).last_hidden_state[:, 0]
        return self.linear(cls)


# ── Shared training utilities ─────────────────────────────────────────────────

class EarlyStopping:
    """Stops training when validation loss stops decreasing."""

    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class EarlyStoppingCorr:
    """Stops training when validation correlation stops increasing."""

    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience    = patience
        self.delta       = delta
        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False

    def __call__(self, val_score: float):
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
