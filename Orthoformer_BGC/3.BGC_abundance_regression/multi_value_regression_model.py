#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add project root directory to Python path so foundation_model can be imported as a module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from foundation_model.orthoformer_model import OrthoformerSelfAttention

# ======================
# Config
# ======================

@dataclass
class MultiValueRegressionCfg:
    # data
    train_dataset: str
    val_dataset: str
    test_dataset: str
    pretrained_dir: str  # pretrained BERT model directory
    output_dir: str = "multi_value_regression_run"
    max_length: int = 2048

    # Regression targets
    num_targets: int = 8  # number of regression targets

    # Model architecture
    head: str = "mlp"  # mlp | bilstm | cnn
    hidden_size: int = 512
    lstm_layers: int = 2
    cnn_kernel: int = 3
    dropout: float = 0.1

    # Loss configuration
    loss_type: str = "mse"  # mse | mae | huber
    huber_delta: float = 1.0

    # Training
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 3e-5
    num_train_epochs: int = 5
    warmup_steps: int = 0
    weight_decay: float = 0.0
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_mse"
    greater_is_better: bool = False  # MSE is better when lower
    fp16: bool = False
    deepspeed: Optional[str] = None
    local_rank: int = -1
    report_to: Tuple[str, ...] = ("tensorboard",)


# ======================
# Data preprocessing
# ======================

def ensure_regression_dense(ds: Dataset, max_length: int, pad_token_id: int = 0) -> Dataset:
    """Preprocess dataset for multi-value regression"""
    cols = ds.column_names
    if "input_ids" not in cols or "targets" not in cols:
        raise ValueError("Dataset must have 'input_ids' and 'targets' columns")

    def _proc(ex: Dict[str, Any]) -> Dict[str, Any]:
        ids = ex["input_ids"]
        targets = ex["targets"]
        
        # Remove BERT CLS/SEP tokens if present
        if len(ids) > 2 and ids[0] == 101 and ids[-1] == 102:
            ids = ids[1:-1]
        
        # Truncate and pad
        L = min(len(ids), max_length)
        ids = ids[:L] + [pad_token_id] * (max_length - L)
        attn = [1] * L + [0] * (max_length - L)
        
        # Normalize targets to list of floats
        if isinstance(targets, (int, float)):
            targets = [float(targets)]
        elif isinstance(targets, list):
            targets = [float(t) for t in targets]
        else:
            raise ValueError(f"Invalid targets format: {type(targets)}")
        
        return {"input_ids": ids, "attention_mask": attn, "targets": targets}

    keep = {"input_ids", "attention_mask", "targets"}
    return ds.map(_proc, remove_columns=[c for c in cols if c not in keep])


# ======================
# Regression Heads
# ======================

class MLPRegressionHead(nn.Module):
    """MLP regression head with global average pooling"""
    def __init__(self, hidden_in: int, hidden_mid: int, num_targets: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_in, hidden_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, hidden_mid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid // 2, num_targets),
        )

    def forward(self, x):  # [B,L,H] -> [B,num_targets]
        return self.net(x.mean(dim=1))


class BiLSTMRegressionHead(nn.Module):
    """BiLSTM regression head with global average pooling"""
    def __init__(self, hidden_in: int, hidden_mid: int, num_targets: int, num_layers: int = 1, dropout: float = 0.1):
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
        self.fc = nn.Sequential(
            nn.Linear(hidden_mid, hidden_mid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid // 2, num_targets)
        )

    def forward(self, x):  # [B,L,H] -> [B,num_targets]
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out.mean(dim=1)))


class CNNRegressionHead(nn.Module):
    """CNN regression head with global average pooling"""
    def __init__(self, hidden_in: int, hidden_mid: int, num_targets: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_in, hidden_mid, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_mid, hidden_mid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid // 2, num_targets)
        )

    def forward(self, x):  # [B,L,H] -> [B,num_targets]
        x = torch.relu(self.conv(x.transpose(1, 2)))  # [B,hidden_mid,L]
        return self.fc(self.dropout(x.mean(dim=2)))  # [B,num_targets]


# ======================
# Multi-Value Regression Model
# ======================

class BertForMultiValueRegression(nn.Module):
    def __init__(self, cfg: BertConfig, num_targets: int, mlm_dir: str,
                 head_type: str = "mlp", hidden_size: int = 512, lstm_layers: int = 1,
                 cnn_kernel: int = 3, dropout: float = 0.1,
                 loss_type: str = "mse", huber_delta: float = 1.0):
        super().__init__()

        # Load pretrained BERT encoder
        mlm = BertForMaskedLM.from_pretrained(mlm_dir)
        self.bert = mlm.bert
        # Replace attention layers with ALiBi-supporting version
        for layer in self.bert.encoder.layer:
            layer.attention.self = OrthoformerSelfAttention(
                layer.attention.self,
                pos_kind="alibi",
                max_position_embeddings=self.bert.config.max_position_embeddings
            )
        self.hidden = cfg.hidden_size
        self.num_targets = num_targets

        # Regression head
        head_type = head_type.lower()
        if head_type == "mlp":
            self.regression_head = MLPRegressionHead(self.hidden, hidden_size, num_targets, dropout)
        elif head_type == "bilstm":
            self.regression_head = BiLSTMRegressionHead(self.hidden, hidden_size, num_targets, lstm_layers, dropout)
        elif head_type == "cnn":
            self.regression_head = CNNRegressionHead(self.hidden, hidden_size, num_targets, cnn_kernel, dropout)
        else:
            raise ValueError(f"Unknown head: {head_type}")

        # Loss functions
        self.loss_type = loss_type.lower()
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == "huber":
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, input_ids, attention_mask=None, targets=None):
        # Get BERT outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        predictions = self.regression_head(bert_outputs.last_hidden_state)  # [B,num_targets]
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, dtype=torch.float32, device=predictions.device)
            
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            
            if targets.shape[-1] != self.num_targets:
                if targets.shape[-1] == 1:
                    targets = targets.expand(-1, self.num_targets)
                else:
                    raise ValueError(f"Targets shape {targets.shape} doesn't match num_targets {self.num_targets}")
            
            loss = self.loss_fn(predictions, targets)
        
        return {"loss": loss, "predictions": predictions}


def regression_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for multi-value regression"""
    predictions, targets = eval_pred
    
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Ensure 2D arrays
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    # Compute overall metrics
    metrics = {
        "mse": float(mean_squared_error(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
    }
    
    # Compute per-target metrics
    for i in range(predictions.shape[1]):
        metrics.update({
            f"target_{i}_mse": float(mean_squared_error(targets[:, i], predictions[:, i])),
            f"target_{i}_mae": float(mean_absolute_error(targets[:, i], predictions[:, i])),
            f"target_{i}_r2": float(r2_score(targets[:, i], predictions[:, i])),
        })
    
    return metrics


# ======================
# Utils
# ======================

def normalize_targets(dataset: Dataset, target_col: str = "targets", stats: Optional[Dict] = None) -> Tuple[Dataset, Dict[str, float]]:
    """Normalize regression targets to have zero mean and unit variance"""
    if stats is None:
        # Compute statistics from training data
        targets = []
        for i in range(len(dataset)):
            target = dataset[i][target_col]
            targets.append([float(target)] if isinstance(target, (int, float)) else [float(t) for t in target])
        
        targets = np.array(targets)
        mean = np.mean(targets, axis=0)
        std = np.std(targets, axis=0)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    
    # Normalize targets using provided statistics
    def normalize_example(example, idx):
        target = example[target_col]
        target_arr = np.array([float(target)] if isinstance(target, (int, float)) else [float(t) for t in target])
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        example[target_col] = ((target_arr - mean) / std).tolist()
        return example
    
    return dataset.map(normalize_example, with_indices=True), stats


def denormalize_predictions(predictions: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Denormalize predictions using the original statistics"""
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    return predictions * std + mean
