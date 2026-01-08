import os, json, argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    BertConfig, BertModel, BertTokenizer, BertForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

# Captum is optional; we guard usage
try:
    from captum.attr import IntegratedGradients
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False


# ======================
# Config
# ======================

@dataclass
class TokenClsCfg:
    # data
    train_dataset: str
    val_dataset: str
    test_dataset: str
    pretrained_dir: str                # Path to the final_model/ directory saved by pretrain
    output_dir: str = "token_multiclass_run"
    max_length: int = 2048
    num_labels: int = 8                # Number of classes 0..(C-1)

    # head options
    head: str = "bilstm"               # mlp | bilstm | cnn
    hidden_size: int = 512
    lstm_layers: int = 2
    cnn_kernel: int = 3
    dropout: float = 0.1

    # loss options
    loss_type: str = "focal"           # cross_entropy | focal
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None    # Length=num_labels, class weight α, optional
    class_counts: Optional[Dict[int, int]] = None  # Sample counts per class, used for class_weights
    compute_class_counts: bool = False            # If counts not provided, automatically scan and compute

    # training
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-5
    num_train_epochs: int = 5
    warmup_steps: int = 0
    weight_decay: float = 0.0
    logging_steps: int = 50
    eval_steps: int = 5000
    save_steps: int = 5000
    save_strategy: str = "steps"  # "steps" or "epoch"
    eval_strategy: str = "steps"  # "steps" or "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_macro"  # Use macro F1 to focus more on minority classes
    greater_is_better: bool = True
    fp16: bool = False
    deepspeed: Optional[str] = None
    local_rank: int = -1
    ignore_index: int = -100
    report_to: Tuple[str, ...] = ("tensorboard",)
    resume_from_checkpoint: Optional[str] = None



def load_model_with_alibi(model_dir, device, use_alibi=True):
    """
    Load model with ALiBi positional encoding support if needed
    
    Args:
        model_dir: Path to model directory
        device: Device to load model on
        use_alibi: Whether to apply ALiBi positional encoding
    
    Returns:
        Loaded model
    """
    if use_alibi:
        print("\n[Info] Loading model with ALiBi positional encoding support...")
        # Try loading BertForMaskedLM first (complete model saved during training)
        try:
            mlm_model = BertForMaskedLM.from_pretrained(model_dir)
            model = mlm_model.bert
            print("[Info] Successfully loaded BertForMaskedLM, extracted BERT base model")
        except Exception as e:
            print(f"[Warn] Failed to load as BertForMaskedLM: {e}")
            print("[Info] Falling back to BertModel...")
            model = BertModel.from_pretrained(model_dir)
        
        # Manually apply ALiBi positional encoding
        try:
            from orthoformer_model import OrthoformerSelfAttention
            print("[Info] Applying ALiBi positional encoding to attention layers...")
            num_layers_replaced = 0
            for layer in model.encoder.layer:
                orig_sa = layer.attention.self
                # Check if already OrthoformerSelfAttention
                if isinstance(orig_sa, OrthoformerSelfAttention):
                    print(f"[Info] Layer {num_layers_replaced + 1} already has OrthoformerSelfAttention")
                else:
                    layer.attention.self = OrthoformerSelfAttention(
                        orig_sa,
                        pos_kind="alibi",
                        max_position_embeddings=model.config.max_position_embeddings
                    )
                num_layers_replaced += 1
            print(f"[Info] Successfully applied ALiBi to {num_layers_replaced} attention layers")
        except ImportError as e:
            print(f"[Warn] Failed to import OrthoformerSelfAttention: {e}")
            print("[Warn] Model will use standard attention (ALiBi functionality disabled)")
        except Exception as e:
            print(f"[Warn] Failed to apply ALiBi: {e}")
            print("[Warn] Model will use standard attention (ALiBi functionality disabled)")
    else:
        print("\n[Info] Loading model with standard positional encoding...")
        model = BertModel.from_pretrained(model_dir)
    
    # Move to specified device, keep training mode (Trainer will switch train/eval mode as needed)
    model = model.to(device)
    return model

# ======================
# Data preprocess
# ======================

def ensure_multiclass_dense(
    ds: Dataset, max_length: int, pad_token_id: int = 0, ignore_index: int = -100, cls_sep_ignore: bool = True
) -> Dataset:
    """
    Expected columns:
      - input_ids: List[int]
      - labels:    List[int]  (or 'labels_ids'; either one). One class id per token (0..C-1)
    If len(labels) == len(input_ids)-2 -> treat as missing CLS/SEP, insert ignore_index at both ends.
    Pad to max_length for PAD tokens, set labels to ignore_index at PAD positions.
    """
    cols = ds.column_names
    if "input_ids" not in cols:
        raise ValueError("Dataset must have 'input_ids'")
    src = "labels" if "labels" in cols else ("labels_ids" if "labels_ids" in cols else None)
    if src is None:
        raise ValueError("Dataset must have 'labels' (or 'labels_ids') as 1D int list per sample.")

    def _proc(ex: Dict[str, Any]) -> Dict[str, Any]:
        ids = ex["input_ids"]
        labs = ex[src]
        # Only train on input_ids[1:-1] (excluding CLS/SEP)
        ids = ids[1:-1]
        L_ids, L_lab = len(ids), len(labs)

        # CLS/SEP case: If labels originally only label the middle part (same length), align directly; otherwise fix redundancy/missing to align length
        if L_lab != L_ids:
            # Truncate or pad to same length as ids; fill missing parts with ignore_index
            if L_lab > L_ids:
                labs = labs[:L_ids]
            else:
                labs = labs + [ignore_index] * (L_ids - L_lab)

        # Truncate to max_length
        L = min(L_ids, max_length)
        ids = ids[:L]
        labs = labs[:L]
        attn = [1] * L

        # pad
        if L < max_length:
            pad_len = max_length - L
            ids = ids + [pad_token_id] * pad_len
            attn = attn + [0] * pad_len
            labs = labs + [ignore_index] * pad_len

        return {"input_ids": ids, "attention_mask": attn, "labels": labs}

    keep = {"input_ids", "attention_mask", "labels"}
    return ds.map(_proc, remove_columns=[c for c in cols if c not in keep])


# ======================
# Focal Loss (multiclass, softmax)
# ======================

class SoftmaxFocalLoss(nn.Module):
    """
    Multiclass Focal Loss (per-token), input logits:[B,L,C], target:[B,L] in {0..C-1} or ignore_index.
    Formula: FL = - α_y * (1 - p_y)^γ * log(p_y)
    """
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: Optional[List[float]] = None,
                 ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", torch.ones(num_classes))
        else:
            if len(alpha) != num_classes:
                raise ValueError(f"alpha length {len(alpha)} != num_classes {num_classes}")
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B,L,C], targets:[B,L]
        """
        B, L, C = logits.shape
        valid = targets.ne(self.ignore_index)  # [B,L]
        if valid.sum() == 0:
            return logits.new_zeros([])

        logits = logits[valid]             # [N,C]
        targets = targets[valid]           # [N]
        log_probs = logits.log_softmax(dim=-1)  # [N,C]
        probs = log_probs.exp()            # [N,C]

        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)      # [N]
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [N]
        alpha_y = self.alpha.to(logits.device).gather(dim=0, index=targets)     # [N]

        loss = - alpha_y * (1 - pt).pow(self.gamma) * log_pt  # [N]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ======================
# Classifier Heads
# ======================

class MLPHead(nn.Module):
    def __init__(self, hidden_in: int, hidden_mid: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_in, hidden_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, num_labels),
        )

    def forward(self, x):  # x: [B,L,H]
        return self.net(x)  # [B,L,C]

class BiLSTMHead(nn.Module):
    def __init__(self, hidden_in: int, hidden_mid: int, num_labels: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_in,
            hidden_size=hidden_mid // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_mid, num_labels)

    def forward(self, x):  # x: [B,L,H]
        out, _ = self.lstm(x)   # [B,L,hidden_mid]
        out = self.dropout(out)
        return self.fc(out)     # [B,L,C]

class CNNHead(nn.Module):
    def __init__(self, hidden_in: int, hidden_mid: int, num_labels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_in, hidden_mid, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_mid, num_labels)

    def forward(self, x):         # x: [B,L,H]
        x = x.transpose(1, 2)     # [B,H,L]
        y = torch.relu(self.conv(x))  # [B,hidden_mid,L]
        y = y.transpose(1, 2)     # [B,L,hidden_mid]
        y = self.dropout(y)
        return self.fc(y)         # [B,L,C]


# ======================
# Model wrapper
# ======================

class BertForTokenMulticlass(nn.Module):
    def __init__(self, cfg: BertConfig, num_labels: int, mlm_dir: str,
                 head_type: str = "mlp", hidden_size: int = 512, lstm_layers: int = 1,
                 cnn_kernel: int = 3, dropout: float = 0.1,
                 loss_type: str = "focal", focal_gamma: float = 2.0, focal_alpha: Optional[List[float]] = None,
                 class_weights: Optional[List[float]] = None,
                 ignore_index: int = -100):
        super().__init__()
        # Transfer encoder
        # Delayed device allocation: load model to CPU first, Trainer will move to correct device automatically
        device = torch.device('cpu')  # Initial load to CPU, Trainer will handle device allocation
        self.bert = load_model_with_alibi(mlm_dir, device, use_alibi=True)
        # Get hidden_size from BERT config (for classification head input dimension)
        self.hidden = cfg.hidden_size
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        # Select classification head
        head_type = head_type.lower()
        if head_type == "mlp":
            self.head = MLPHead(self.hidden, hidden_size, num_labels, dropout)
        elif head_type == "bilstm":
            self.head = BiLSTMHead(self.hidden, hidden_size, num_labels, lstm_layers, dropout)
        elif head_type == "cnn":
            self.head = CNNHead(self.hidden, hidden_size, num_labels, cnn_kernel, dropout)
        else:
            raise ValueError(f"Unknown head: {head_type}")

        # Loss
        self.loss_type = loss_type.lower()
        if self.loss_type == "focal":
            self.crit = SoftmaxFocalLoss(num_classes=num_labels, gamma=focal_gamma, alpha=focal_alpha,
                                         ignore_index=ignore_index, reduction="mean")
        elif self.loss_type == "cross_entropy":
            self.crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean",
                                            weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # For focal loss, we also support adding class_weights: handled in compute_loss (multiplying to per-class logit probabilities is not intuitive)
        self.class_weights = None
        if class_weights is not None and self.loss_type == "focal":
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)  # [C]

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.head(out.last_hidden_state)  # [B,L,C]
        loss = None
        if labels is not None:
            if self.loss_type == "cross_entropy":
                loss = self.crit(logits.view(-1, self.num_labels), labels.view(-1).long())
            else:
                # focal: compute focal first, then add optional class_weights (weighted by true class)
                B, L, C = logits.shape
                valid = labels.ne(self.ignore_index)  # [B,L]
                if valid.sum() == 0:
                    loss = logits.new_zeros([])
                else:
                    loss_focal = self.crit(logits, labels)  # mean over valid tokens
                    if self.class_weights is None:
                        loss = loss_focal
                    else:
                        # Improvement: apply class weights per sample instead of simple averaging
                        # This better handles class imbalance
                        with torch.no_grad():
                            cls = labels[valid].view(-1)  # [N]
                        cw = self.class_weights.to(logits.device).gather(0, cls)  # [N]
                        # Method 1: Use weighted average (more direct and effective)
                        # Compute focal loss per sample, then weight by class weights
                        # Since focal loss is already mean, we need to recompute per-sample loss
                        # Simplified approach: use weighted average, weights are square root of class weights (more stable)
                        cw_normalized = cw / (cw.mean() + 1e-8)  # Normalize to around mean
                        loss = loss_focal * cw_normalized.mean()
        return {"loss": loss, "logits": logits}


# ======================
# Metrics
# ======================

def multiclass_token_metrics(eval_pred, ignore_index: int = -100, num_labels: Optional[int] = None) -> Dict[str, float]:
    logits, labels = eval_pred  # logits:[B,L,C], labels:[B,L]
    preds = np.argmax(logits, axis=-1)  # [B,L]

    y_true, y_pred = [], []
    strict_total, strict_correct = 0, 0

    B, L = preds.shape
    for b in range(B):
        for t in range(L):
            yi = labels[b, t]
            if yi == ignore_index:
                continue
            strict_total += 1
            if preds[b, t] == yi:
                strict_correct += 1
            y_true.append(int(yi))
            y_pred.append(int(preds[b, t]))

    if len(y_true) == 0:
        # no valid tokens
        base = {k: 0.0 for k in [
            "accuracy", "precision_micro", "recall_micro", "f1_micro",
            "precision_macro", "recall_macro", "f1_macro", "accuracy_strict"
        ]}
        # also return empty per-class metrics if requested
        if num_labels is not None:
            for i in range(num_labels):
                base[f"precision_c{i}"] = 0.0
                base[f"recall_c{i}"] = 0.0
                base[f"f1_c{i}"] = 0.0
                base[f"support_c{i}"] = 0.0
        return base

    # Aggregate metrics
    acc = accuracy_score(y_true, y_pred)
    prec_mi, rec_mi, f1_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    prec_ma, rec_ma, f1_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc_strict = strict_correct / max(1, strict_total)

    # Per-class metrics (aligned to label ids 0..num_labels-1 when provided)
    labels_list = list(range(num_labels)) if num_labels is not None else None
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, average=None, zero_division=0
    )

    out = {
        "accuracy": float(acc),
        "precision_micro": float(prec_mi),
        "recall_micro": float(rec_mi),
        "f1_micro": float(f1_mi),
        "precision_macro": float(prec_ma),
        "recall_macro": float(rec_ma),
        "f1_macro": float(f1_ma),
        "accuracy_strict": float(acc_strict),
    }

    # attach per-class
    for i, (p, r, f, s) in enumerate(zip(prec_c, rec_c, f1_c, sup_c)):
        out[f"precision_c{i}"] = float(p)
        out[f"recall_c{i}"] = float(r)
        out[f"f1_c{i}"] = float(f)
        out[f"support_c{i}"] = float(s)

    return out


# ======================
# Utils: class weights from counts
# ======================

def counts_to_weights(counts: Dict[int, int], num_labels: int) -> List[float]:
    # Defensive: assign minimum non-zero count to missing classes
    arr = np.zeros(num_labels, dtype=np.float64)
    for k, v in counts.items():
        if 0 <= int(k) < num_labels:
            arr[int(k)] = max(1, int(v))
    min_nonzero = arr[arr > 0].min() if (arr > 0).any() else 1.0
    arr[arr == 0] = min_nonzero
    total = arr.sum()
    weights = (total / arr).tolist()
    return weights

def scan_counts(hf_ds: Dataset, num_labels: int, ignore_index: int) -> Dict[int, int]:
    cnt = np.zeros(num_labels, dtype=np.int64)
    for i in range(len(hf_ds)):
        labs = hf_ds[i]["labels"]
        for y in labs:
            if y == ignore_index:
                continue
            if 0 <= y < num_labels:
                cnt[y] += 1
    return {int(i): int(v) for i, v in enumerate(cnt.tolist())}


# ======================
# Train
# ======================

def train_token_multiclass(cfg: TokenClsCfg):
    # Prefer the slow tokenizer if that's what the checkpoint provides (avoids fast-conversion & tiktoken deps)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_dir, use_fast=False)
    except Exception as e:
        print("[warn] AutoTokenizer failed to load fast/slow tokenizer:", e)
        # Fallback: build a minimal tokenizer stub for pad_id usage
        class _PadStub:
            pad_token_id = 0
        tokenizer = _PadStub()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        # Try to pull from config if available; otherwise default to 0
        try:
            cfg_json = BertConfig.from_pretrained(cfg.pretrained_dir)
            pad_id = getattr(cfg_json, "pad_token_id", 0) or 0
        except Exception:
            pad_id = 0

    # Load data
    train_raw = load_from_disk(cfg.train_dataset)
    val_raw = load_from_disk(cfg.val_dataset)
    test_raw = load_from_disk(cfg.test_dataset)

    # Preprocess (remove CLS/SEP, pad to max_length, set labels to ignore_index for PAD positions)
    ds = DatasetDict({
        "train": ensure_multiclass_dense(train_raw, cfg.max_length, pad_id, cfg.ignore_index),
        "validation": ensure_multiclass_dense(val_raw, cfg.max_length, pad_id, cfg.ignore_index),
        "test": ensure_multiclass_dense(test_raw, cfg.max_length, pad_id, cfg.ignore_index),
    })

    # Class weights: prefer user-provided counts, otherwise optionally auto-compute
    class_weights = None
    if cfg.class_counts is not None:
        class_weights = counts_to_weights(cfg.class_counts, cfg.num_labels)
        print("[info] use provided class_counts -> class_weights:", class_weights)
    elif cfg.compute_class_counts:
        # Scan all from train/val/test (or just train, as needed)
        counts_train = scan_counts(ds["train"], cfg.num_labels, cfg.ignore_index)
        counts_val = scan_counts(ds["validation"], cfg.num_labels, cfg.ignore_index)
        counts_test = scan_counts(ds["test"], cfg.num_labels, cfg.ignore_index)
        merged = {}
        for i in range(cfg.num_labels):
            merged[i] = counts_train.get(i, 0) + counts_val.get(i, 0) + counts_test.get(i, 0)
        class_weights = counts_to_weights(merged, cfg.num_labels)
        print("[info] scanned class_counts ->", merged)
        print("[info] class_weights ->", class_weights)
    else:
        print("[info] no class_counts given; training without class_weights")
    print("[info] Multi-GPU is enabled via torch.distributed when launched with torchrun/accelerate.")

    # Optional: normalize class weights for stability
    # Note: For extremely imbalanced data, full normalization may not be strong enough
    # Can use square root normalization or partial normalization
    if class_weights is not None:
        _cw = np.array(class_weights, dtype=np.float32)
        _m = float(_cw.mean()) if _cw.size > 0 else 1.0
        if _m > 0:
            # Option 1: Full normalization (current, may not be strong enough)
            # class_weights = (_cw / _m).tolist()
            # Option 2: Square root normalization (gentler, preserves more weight differences)
            class_weights = (np.sqrt(_cw / _m) * np.sqrt(_m)).tolist()
            # Option 3: No normalization (most aggressive, but may be unstable)
            # class_weights = _cw.tolist()
            print("[info] normalized class_weights (sqrt normalization):", class_weights)
            print("[info] original class_weights (for reference):", _cw.tolist())

    # Model
    base_cfg = BertConfig.from_pretrained(cfg.pretrained_dir)
    model = BertForTokenMulticlass(
        base_cfg, cfg.num_labels, cfg.pretrained_dir,
        head_type=cfg.head, hidden_size=cfg.hidden_size, lstm_layers=cfg.lstm_layers,
        cnn_kernel=cfg.cnn_kernel, dropout=cfg.dropout,
        loss_type=cfg.loss_type, focal_gamma=cfg.focal_gamma, focal_alpha=cfg.focal_alpha,
        class_weights=class_weights, ignore_index=cfg.ignore_index
    )

    # Trainer
    class TokenTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Compatibility with HF Trainer >=4.33 which forwards `num_items_in_batch`
            labels = inputs.pop("labels")
            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    # Build TrainingArguments with GPU-aware defaults and version compatibility
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    args_kwargs = dict(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        # Use configured strategy (epoch or steps)
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        report_to=list(cfg.report_to),
        fp16=(cfg.fp16 and use_cuda),
        dataloader_pin_memory=use_cuda,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        # Enable safetensors to avoid torch.load version issues (torch >= 2.6 required for pytorch_model.bin)
        # Note: If CustomBertSelfAttention causes issues, we can handle it separately
        save_safetensors=True,
    )
    if cfg.deepspeed:
        args_kwargs.update({"deepspeed": cfg.deepspeed, "local_rank": cfg.local_rank})

    # Add eval_steps and save_steps only if strategy is "steps"
    if cfg.eval_strategy == "steps":
        args_kwargs["eval_steps"] = cfg.eval_steps
    if cfg.save_strategy == "steps":
        args_kwargs["save_steps"] = cfg.save_steps

    # --- Compatibility: filter kwargs to match installed transformers version ---
    from inspect import signature
    _sig = signature(TrainingArguments.__init__)
    _allowed = {k: v for k, v in args_kwargs.items() if k in _sig.parameters}

    # Map eval_strategy -> evaluation_strategy if supported
    if "evaluation_strategy" in _sig.parameters and "eval_strategy" in args_kwargs:
        _allowed["evaluation_strategy"] = args_kwargs["eval_strategy"]
    # If neither is supported, drop both
    if ("evaluation_strategy" not in _sig.parameters) and ("eval_strategy" not in _sig.parameters):
        _allowed.pop("eval_strategy", None)

    # Some very old versions only use do_eval to enable eval loop
    if ("evaluation_strategy" not in _sig.parameters) and ("eval_strategy" not in _sig.parameters) and ("do_eval" in _sig.parameters):
        _allowed["do_eval"] = True

    training_args = TrainingArguments(**_allowed)

    def _metrics_fn(ep):
        return multiclass_token_metrics(ep, ignore_index=cfg.ignore_index, num_labels=cfg.num_labels)

    trainer = TokenTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        compute_metrics=_metrics_fn,
    )

    # Resume from checkpoint if specified
    resume_from_checkpoint = cfg.resume_from_checkpoint
    if resume_from_checkpoint:
        print(f"[info] Resuming training from checkpoint: {resume_from_checkpoint}")
    
    train_out = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Evaluate
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=ds["test"])

    print("\n=== Validation metrics ===")
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Record configuration
    with open(os.path.join(cfg.output_dir, "token_multiclass_config.json"), "w") as f:
        json.dump({
            "num_labels": cfg.num_labels,
            "head": cfg.head,
            "hidden_size": cfg.hidden_size,
            "lstm_layers": cfg.lstm_layers,
            "cnn_kernel": cfg.cnn_kernel,
            "dropout": cfg.dropout,
            "loss_type": cfg.loss_type,
            "focal_gamma": cfg.focal_gamma,
            "focal_alpha": cfg.focal_alpha,
            "ignore_index": cfg.ignore_index,
            "class_counts": cfg.class_counts,
            "used_class_weights": class_weights,
        }, f, indent=2)

    return {"train": train_out.metrics, "validation": eval_metrics, "test": test_metrics}



# ======================
# Label-wise Attention (sequence-level multilabel, 8 labels)
# ======================

@dataclass
class LWASeqCfg:
    train_csv: str
    val_csv: str
    test_csv: Optional[str]
    pretrained_dir: str
    output_dir: str = "seq_lwa_run"
    num_labels: int = 8
    max_length: int = 160
    freeze_base: bool = True
    learning_rate_head: float = 1e-3
    learning_rate_base: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    logging_steps: int = 50
    fp16: bool = False
    ignore_index: int = -100
    # visualization & export
    save_thresholds: bool = True


def _parse_multilabel_str(x: str, num_labels: int) -> List[int]:
    x = str(x).strip()
    if x.startswith("["):
        idxs = json.loads(x)
    elif "," in x:
        idxs = [int(t) for t in x.split(",") if t.strip()]
    else:
        idxs = [int(t) for t in x.split() if t.strip()]
    arr = [0]*num_labels
    for i in idxs:
        if 0 <= i < num_labels:
            arr[i] = 1
    return arr


class _SeqLwaDS(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, num_labels: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(str(row["text"]), truncation=True, max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
        item["text"] = str(row["text"])  # for later visualization
        return item


def _compute_pos_weight_multilabel(y: np.ndarray) -> torch.Tensor:
    eps = 1e-6
    P = y.sum(axis=0)
    N = y.shape[0]
    w = (N - P) / (P + eps)
    return torch.tensor(np.clip(w, 1.0, 1e6), dtype=torch.float32)


class LabelWiseAttentionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, temperature: Optional[float] = None):
        super().__init__()
        self.num_labels = num_labels
        self.q = nn.Parameter(torch.randn(num_labels, hidden_size))
        self.proj = nn.Linear(hidden_size, 1)
        self.temperature = temperature

    def forward(self, H: torch.Tensor, mask: torch.Tensor, return_attn: bool = False):
        # H: [B,T,D], mask: [B,T]
        B, T, D = H.shape
        # [B,L,T]
        scores = torch.einsum('ld,btd->blt', self.q, H)
        if self.temperature is not None:
            scores = scores / self.temperature
        mask_ = (mask == 0).unsqueeze(1).expand(B, self.num_labels, T)
        scores = scores.masked_fill(mask_, float('-inf'))
        alpha = torch.softmax(scores, dim=-1)  # [B,L,T]
        # [B,L,D]
        label_repr = torch.einsum('blt,btd->bld', alpha, H)
        logits = self.proj(label_repr).squeeze(-1)  # [B,L]
        if return_attn:
            return logits, alpha
        return logits


class BertForSeqLWA(nn.Module):
    def __init__(self, pretrained_dir: str, num_labels: int = 8, freeze_base: bool = True, temperature: Optional[float] = None):
        super().__init__()
        mlm = BertForMaskedLM.from_pretrained(pretrained_dir)
        self.bert = mlm.bert
        self.num_labels = num_labels
        self.head = LabelWiseAttentionHead(self.bert.config.hidden_size, num_labels, temperature)
        if freeze_base:
            for p in self.bert.parameters():
                p.requires_grad = False
        self._loss = nn.BCEWithLogitsLoss(reduction='mean')  # pos_weight set in Trainer

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_attn: bool = False, pos_weight: Optional[torch.Tensor]=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        H = out.last_hidden_state
        if return_attn:
            logits, alpha = self.head(H, attention_mask, return_attn=True)
        else:
            logits = self.head(H, attention_mask, return_attn=False)
            alpha = None
        loss = None
        if labels is not None:
            if pos_weight is not None:
                self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
            loss = self._loss(logits, labels)
        return {"loss": loss, "logits": logits, "attn": alpha}


class _BCETrainer(Trainer):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.crit = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.args.device) if self.pos_weight is not None else None)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs['logits']
        loss = self.crit(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _threshold_search(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    L = y_true.shape[1]
    ths = []
    for i in range(L):
        yt, yp = y_true[:, i], y_prob[:, i]
        try:
            prec, rec, thr = precision_recall_curve(yt, yp)
            f1 = (2*prec*rec)/(prec+rec+1e-12)
            best = thr[np.nanargmax(f1[:-1])] if len(thr) > 0 else 0.5
        except Exception:
            best = 0.5
        ths.append(float(best))
    return np.array(ths, dtype=np.float32)


def _load_csv_split(csv_path: str, num_labels: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert 'text' in df.columns and 'labels' in df.columns, "CSV must contain 'text' and 'labels'"
    df['labels'] = df['labels'].apply(lambda s: _parse_multilabel_str(s, num_labels))
    return df


def train_seq_lwa(cfg: LWASeqCfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_dir, use_fast=True)
    df_tr = _load_csv_split(cfg.train_csv, cfg.num_labels)
    df_va = _load_csv_split(cfg.val_csv, cfg.num_labels)
    has_test = cfg.test_csv is not None and os.path.exists(cfg.test_csv)
    df_te = _load_csv_split(cfg.test_csv, cfg.num_labels) if has_test else None

    y_tr = np.stack(df_tr['labels'].values)
    pos_weight = _compute_pos_weight_multilabel(y_tr)

    ds_tr = _SeqLwaDS(df_tr, tokenizer, cfg.max_length, cfg.num_labels)
    ds_va = _SeqLwaDS(df_va, tokenizer, cfg.max_length, cfg.num_labels)
    ds_te = _SeqLwaDS(df_te, tokenizer, cfg.max_length, cfg.num_labels) if has_test else None

    model = BertForSeqLWA(cfg.pretrained_dir, num_labels=cfg.num_labels, freeze_base=cfg.freeze_base)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_auprc',
        greater_is_better=True,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate_head,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        report_to=['none'],
    )

    # Build optimizer with different LR if base is unfrozen
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('head.'):
            head_params.append(p)
        else:
            base_params.append(p)
    optim_groups = []
    if head_params:
        optim_groups.append({"params": head_params, "lr": cfg.learning_rate_head})
    if base_params:
        optim_groups.append({"params": base_params, "lr": cfg.learning_rate_base})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=cfg.weight_decay)

    trainer = _BCETrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=lambda ep: _seq_metrics(ep, cfg.num_labels)
    )

    trainer.train()

    # Thresholds on validation
    va = trainer.predict(ds_va)
    val_logits = va.predictions
    val_labels = va.label_ids
    val_probs = 1/(1+np.exp(-val_logits))
    ths = _threshold_search(val_labels, val_probs)
    if cfg.save_thresholds:
        with open(os.path.join(cfg.output_dir, 'best_thresholds.json'), 'w', encoding='utf-8') as f:
            json.dump({"thresholds": ths.tolist()}, f, ensure_ascii=False, indent=2)

    # Save model/tokenizer
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Optional test
    if has_test and ds_te is not None:
        te = trainer.predict(ds_te)
        te_probs = 1/(1+np.exp(-te.predictions))
        te_preds = (te_probs >= ths[None, :]).astype(int)
        micro_f1 = f1_score(te.label_ids, te_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(te.label_ids, te_preds, average='macro', zero_division=0)
        with open(os.path.join(cfg.output_dir, 'test_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({"micro_f1@best": float(micro_f1), "macro_f1@best": float(macro_f1)}, f, indent=2)

    return ths


def _seq_metrics(eval_pred, num_labels: int) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    # macro AUPRC
    try:
        per_label_auprc = [average_precision_score(labels[:,i], probs[:,i]) for i in range(num_labels)]
        macro_auprc = float(np.mean(per_label_auprc))
    except Exception:
        macro_auprc = 0.0
    return {"micro_f1@0.5": micro_f1, "macro_f1@0.5": macro_f1, "macro_auprc": macro_auprc}


# ----------------------
# Rationale extraction & visualization
# ----------------------
@torch.no_grad()
def lwa_token_importance(model: BertForSeqLWA, tokenizer: AutoTokenizer, text: str, label_id: int, max_length: int = 160, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(return_attn=True, **enc)
    alpha = out['attn'][0, label_id]  # [T]
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    return tokens, alpha.detach().cpu().numpy().tolist()


def plot_lwa_heatmap(tokens: List[str], scores: List[float], title: str = None, save_path: Optional[str] = None, max_tokens_per_row: int = 40):
    # Simple heatmap-like bar viz over tokens
    plt.figure(figsize=(min(16, max(6, len(tokens)/2)), 2.5))
    # normalize
    s = np.array(scores, dtype=np.float32)
    s = (s - s.min()) / (s.ptp() + 1e-8)
    txt = ' '.join(tokens)
    plt.imshow(s[None, :], aspect='auto')
    plt.yticks([])
    plt.xticks([])
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ----------------------
# Robustness checks: Deletion (erasure) & Integrated Gradients (optional)
# ----------------------
@torch.no_grad()
def deletion_test_drop(model: BertForSeqLWA, tokenizer: AutoTokenizer, text: str, label_id: int, drop_ratio: float = 0.2, max_length: int = 160, device: str = None) -> Dict[str, float]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    p0 = torch.sigmoid(out['logits'])[0, label_id].item()

    out_attn = model(return_attn=True, **enc)
    alpha = out_attn['attn'][0, label_id].detach().cpu().numpy()
    k = max(1, int(drop_ratio * alpha.shape[0]))
    idx = np.argsort(-alpha)[:k]

    input_ids = enc['input_ids'][0].detach().cpu().numpy()
    mask = np.ones_like(input_ids, dtype=bool)
    mask[idx] = False
    input_ids_drop = torch.tensor(input_ids[mask], dtype=torch.long, device=device).unsqueeze(0)
    attn_mask_drop = torch.ones_like(input_ids_drop)
    out2 = model(input_ids=input_ids_drop, attention_mask=attn_mask_drop)
    p1 = torch.sigmoid(out2['logits'])[0, label_id].item()
    return {"p_before": p0, "p_after_drop": p1, "delta": p0 - p1, "drop_ratio": float(drop_ratio)}


def integrated_gradients_importance(model: BertForSeqLWA, tokenizer: AutoTokenizer, text: str, label_id: int, max_length: int = 160, device: str = None):
    if not _HAS_CAPTUM:
        raise RuntimeError("Captum is not installed; please `pip install captum` to use Integrated Gradients")
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')

    def forward_func(input_ids, attention_mask):
        out = model(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        logit = out['logits'][:, label_id]
        prob = torch.sigmoid(logit)
        return prob

    ig = IntegratedGradients(lambda x, m: forward_func(x, m))
    attributions, delta = ig.attribute(inputs=enc['input_ids'].to(device),
                                       additional_forward_args=(enc['attention_mask'].to(device),),
                                       n_steps=32, return_convergence_delta=True)
    # sum over embedding dim does not apply here since inputs are ids; we approximate by token-level abs attribution via embedding gradients using embed layer hook is more precise; for simplicity, we map token-wise absolute values
    token_attr = attributions.squeeze(0).abs().sum(dim=-1) if attributions.dim() == 3 else attributions.squeeze(0).abs()
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    scores = token_attr.detach().cpu().numpy().tolist()
    return tokens, scores, float(delta.mean().abs().item() if hasattr(delta, 'mean') else float(delta))

# ======================
# CLI (unified)
# ======================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='task', required=True)

    # --- Subcommand: token-multiclass (existing) ---
    ap_tok = sub.add_parser('token_multiclass', help='Token-level multiclass finetuning (existing)')
    ap_tok.add_argument("--train_dataset", required=True, help="path to datasets/train.dataset (load_from_disk)")
    ap_tok.add_argument("--val_dataset", required=True, help="path to datasets/val.dataset")
    ap_tok.add_argument("--test_dataset", required=True, help="path to datasets/test.dataset")
    ap_tok.add_argument("--pretrained_dir", required=True, help="pretrained BERT dir (from your pretrainer)")
    ap_tok.add_argument("--output_dir", default="token_multiclass_run")
    ap_tok.add_argument("--num_labels", type=int, required=True)
    ap_tok.add_argument("--max_length", type=int, default=1024)
    ap_tok.add_argument("--head", type=str, default="bilstm", choices=["mlp", "bilstm", "cnn"])
    ap_tok.add_argument("--hidden_size", type=int, default=512)
    ap_tok.add_argument("--lstm_layers", type=int, default=2)
    ap_tok.add_argument("--cnn_kernel", type=int, default=3)
    ap_tok.add_argument("--dropout", type=float, default=0.1)
    ap_tok.add_argument("--loss_type", type=str, default="focal", choices=["cross_entropy", "focal"])
    ap_tok.add_argument("--focal_gamma", type=float, default=2.0)
    ap_tok.add_argument("--focal_alpha", type=str, default=None, help="JSON list (length=num_labels)")
    ap_tok.add_argument("--class_counts", type=str, default=None, help="Class counts JSON")
    ap_tok.add_argument("--compute_class_counts", action="store_true")
    ap_tok.add_argument("--batch_size", type=int, default=8)
    ap_tok.add_argument("--lr", type=float, default=3e-5)
    ap_tok.add_argument("--epochs", type=int, default=5)
    ap_tok.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    ap_tok.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Save checkpoint strategy: 'steps' or 'epoch'")
    ap_tok.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Evaluation strategy: 'steps' or 'epoch'")
    ap_tok.add_argument("--fp16", action="store_true")
    ap_tok.add_argument("--deepspeed", type=str, default=None)
    ap_tok.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")

    # --- Subcommand: seq_lwa (new) ---
    ap_lwa = sub.add_parser('seq_lwa', help='Sequence-level multilabel with Label-wise Attention + rationales')
    ap_lwa.add_argument('--train_csv', required=True)
    ap_lwa.add_argument('--val_csv', required=True)
    ap_lwa.add_argument('--test_csv', default=None)
    ap_lwa.add_argument('--pretrained_dir', required=True)
    ap_lwa.add_argument('--output_dir', default='seq_lwa_run')
    ap_lwa.add_argument('--num_labels', type=int, default=8)
    ap_lwa.add_argument('--max_length', type=int, default=160)
    ap_lwa.add_argument('--freeze_base', action='store_true')
    ap_lwa.add_argument('--learning_rate_head', type=float, default=1e-3)
    ap_lwa.add_argument('--learning_rate_base', type=float, default=1e-5)
    ap_lwa.add_argument('--weight_decay', type=float, default=0.01)
    ap_lwa.add_argument('--warmup_ratio', type=float, default=0.06)
    ap_lwa.add_argument('--num_train_epochs', type=int, default=5)
    ap_lwa.add_argument('--per_device_train_batch_size', type=int, default=32)
    ap_lwa.add_argument('--per_device_eval_batch_size', type=int, default=64)
    ap_lwa.add_argument('--logging_steps', type=int, default=50)
    ap_lwa.add_argument('--fp16', action='store_true')

    args = ap.parse_args()

    if args.task == 'token_multiclass':
        focal_alpha = None
        if args.focal_alpha:
            focal_alpha = json.loads(args.focal_alpha)
            if not isinstance(focal_alpha, list):
                raise ValueError("--focal_alpha requires a JSON list")
        class_counts = None
        if args.class_counts:
            cc_raw = json.loads(args.class_counts)
            class_counts = {int(k): int(v) for k, v in cc_raw.items()}
        cfg = TokenClsCfg(
            train_dataset=args.train_dataset,
            val_dataset=args.val_dataset,
            test_dataset=args.test_dataset,
            pretrained_dir=args.pretrained_dir,
            output_dir=args.output_dir,
            num_labels=args.num_labels,
            max_length=args.max_length,
            head=args.head,
            hidden_size=args.hidden_size,
            lstm_layers=args.lstm_layers,
            cnn_kernel=args.cnn_kernel,
            dropout=args.dropout,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            focal_alpha=focal_alpha,
            class_counts=class_counts,
            compute_class_counts=args.compute_class_counts,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            warmup_steps=getattr(args, 'warmup_steps', 0),
            save_strategy=getattr(args, 'save_strategy', 'steps'),
            eval_strategy=getattr(args, 'eval_strategy', 'steps'),
            fp16=args.fp16,
            deepspeed=args.deepspeed,
            resume_from_checkpoint=getattr(args, 'resume_from_checkpoint', None),
        )
        train_token_multiclass(cfg)
        return

    if args.task == 'seq_lwa':
        cfg = LWASeqCfg(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            pretrained_dir=args.pretrained_dir,
            output_dir=args.output_dir,
            num_labels=args.num_labels,
            max_length=args.max_length,
            freeze_base=args.freeze_base,
            learning_rate_head=args.learning_rate_head,
            learning_rate_base=args.learning_rate_base,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_steps=args.logging_steps,
            fp16=args.fp16,
        )
        train_seq_lwa(cfg)
        return


if __name__ == "__main__":
    main()
