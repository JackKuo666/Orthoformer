"""
Orthoformer Model Implementation
Custom BERT-based model with support for ALiBi and RoPE positional encodings.
"""

import os
import json
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainerCallback
)
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import Dataset, DatasetDict, load_from_disk, ClassLabel, Features, Value
import pickle
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
import math



def _build_alibi_slopes(num_heads: int) -> torch.Tensor:
    import math
    m = 2 ** math.floor(math.log2(num_heads))
    slopes = torch.pow(2, -torch.arange(0, m, dtype=torch.float32) / m)
    if m < num_heads:
        extra = torch.pow(2, -torch.arange(1, 2*(num_heads - m)+1, 2, dtype=torch.float32) / m)
        slopes = torch.cat([slopes, extra], dim=0)
    return slopes


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class OrthoformerSelfAttention(nn.Module):
    def __init__(self, orig: BertSelfAttention, pos_kind: str, max_position_embeddings: int):
        super().__init__()
        self.orig = orig
        self.pos_kind = pos_kind
        self.num_attention_heads = orig.num_attention_heads
        self.attention_head_size = orig.attention_head_size
        self.all_head_size = orig.all_head_size
        self.query = orig.query
        self.key = orig.key
        self.value = orig.value
        self.dropout = orig.dropout
        self.is_decoder = getattr(orig, 'is_decoder', False)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.register_buffer("alibi_slopes", None, persistent=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _maybe_build_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (self.rope_cos is not None) and (self.rope_cos.size(0) >= seq_len):
            return
        dim = self.attention_head_size
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.rope_cos = emb.cos().to(dtype)
        self.rope_sin = emb.sin().to(dtype)

    def _build_alibi_bias(self, L: int, H: int, device, dtype):
        if self.alibi_slopes is None or self.alibi_slopes.numel() != H:
            self.alibi_slopes = _build_alibi_slopes(H).to(device=device, dtype=dtype)
        pos = torch.arange(L, device=device)
        dist = (pos[None, :] - pos[:, None]).abs().to(dtype)
        bias = self.alibi_slopes.view(1, H, 1, 1) * (-dist.view(1, 1, L, L))
        return bias

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.pos_kind == "rope":
            L = hidden_states.size(1)
            self._maybe_build_rope(L, hidden_states.device, hidden_states.dtype)
            query_layer, key_layer = _apply_rotary_pos_emb(query_layer, key_layer, self.rope_cos[:L, :], self.rope_sin[:L, :])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.pos_kind == "alibi":
            B, H, L, _ = attention_scores.shape
            attention_scores = attention_scores + self._build_alibi_bias(L, H, attention_scores.device, attention_scores.dtype)

        if attention_mask is not None:
            # Normalize attention_mask to additive mask shape [B,1,1,L] with values 0 or -inf
            am = attention_mask
            with torch.no_grad():
                if am.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                    am = am.to(dtype=attention_scores.dtype)
                    am = (1.0 - am) * (-1e4)
                else:
                    # Already float: assume may already be additive mask; if range is {0,1}, convert to {0,-1e4}
                    maxv = float(am.max().item()) if am.numel() > 0 else 0.0
                    minv = float(am.min().item()) if am.numel() > 0 else 0.0
                    if minv >= 0.0 and maxv <= 1.0:
                        am = (1.0 - am) * (-1e4)
                    am = am.to(dtype=attention_scores.dtype)
                # Expand to [B,1,1,L]
                while am.dim() < 4:
                    am = am.unsqueeze(1)
                if am.size(-1) != attention_scores.size(-1):
                    am = am[..., :attention_scores.size(-1)]
            # Use a detached copy of the mask
            am = am.detach().clone()
            attention_scores = attention_scores + am

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        return outputs

