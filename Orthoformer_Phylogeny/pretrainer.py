"""
GeneBERT训练器模块
基于基因表达数据的BERT无监督训练器
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


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 数据相关
    dataset_path: str
    token_dictionary_file: str
    output_dir: str = "gene_bert_output"
    model_name: str = "gene-bert"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 1024
    
    # 训练相关
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
    warmup_ratio: Optional[float] = None
    lr_scheduler_type: str = "linear"  # linear | cosine | cosine_with_restarts | polynomial | constant | constant_with_warmup
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # MLM 掩码概率（用于 DataCollator）
    mlm_probability: float = 0.15
    # MLM 掩码风格：token（逐token）| span（连续片段）
    mlm_mask_style: str = "token"
    # Span masking 平均长度（上限）
    span_length: int = 5
    # 位置编码：abs | rope | alibi（当前实现abs，rope/alibi为预留）
    pos_encoding: str = "abs"
    # 评估/保存/日志策略
    evaluation_strategy: str = "steps"  # 'no' | 'steps' | 'epoch'
    save_strategy: str = "steps"        # 'no' | 'steps' | 'epoch'
    report_to: List[str] = None
    
    # 模型相关
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = hidden_size * 4
    
    # 优化相关
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    
    # 分布式训练
    deepspeed: Optional[str] = None
    local_rank: int = -1
    use_accelerate: bool = False
    
    # 分类相关
    enable_classification: bool = False
    num_classes: int = 2
    classification_labels: Optional[np.ndarray] = None
    classification_weight: float = 0.0
    detailed_classification_metrics: bool = True  # 是否计算详细的分类指标
    classification_weight_warmup_steps: int = 0  # λ warmup步数，0表示不使用warmup
    
    # 稀有类处理相关
    rare_class_mode: str = "other"  # 'other' | 'filter' | 'keep'
    min_count: int = 100  # 合并/过滤阈值，默认100（基于实际数据分布优化）
    
    # 数据采样相关（用于快速测试）
    data_sample_ratio: float = 1.0  # 数据采样比例，1.0表示使用全部数据，0.1表示使用10%数据
    # 训练控制与随机种子
    gradient_accumulation_steps: int = 1
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: Optional[dict] = None
    seed: int = 42
    data_seed: int = 42
    
    # 分类头池化策略
    classification_pool: str = "mask_mean"  # 可选: 'cls' | 'mask_mean' | 'attn'
    
    # Head/Tail分析配置
    head_min_count: int = 50  # head类的最小样本数阈值
    use_coverage_head: bool = False  # 是否使用覆盖率法划分head/tail
    head_coverage: float = 0.9  # 覆盖率阈值（0.9表示覆盖90%样本的类为head）
    top_k_accuracy: int = 5  # top-k准确率的k值
    # 分类损失配置
    loss_kind: Optional[str] = None  # 'ce' | 'weighted_ce' | 'focal' | None(默认CE)
    class_weight_beta: float = 0.5
    focal_gamma: float = 2.0
    use_class_weights: bool = False
    classification_weight_warmup_ratio: Optional[float] = None  # 优先按比例warmup

    # v4 采样与 v5 对比学习配置（新增）
    sampling_mode: Optional[str] = None  # 'stratified' | 'temperature' | 'mix' | None
    quota_per_class: Optional[int] = None
    sampling_beta: Optional[float] = None
    mix_ratio: Optional[float] = None
    simcse_alpha: float = 0.0
    simcse_tau: float = 0.1
    simcse_pool: str = "mask_mean"  # 'mask_mean' | 'attn'


class TokenizerManager:
    """Tokenizer管理器"""
    
    def __init__(self, token_dictionary_file: str, output_dir: Path, max_length: int):
        self.token_dictionary_file = token_dictionary_file
        self.output_dir = output_dir
        self.max_length = max_length
        self.token_dictionary = self._load_token_dictionary()
        self.tokenizer = self._create_tokenizer()
    
    def _load_token_dictionary(self) -> Dict[str, int]:
        """加载token字典"""
        if self.token_dictionary_file.endswith(".csv"):
            with open(self.token_dictionary_file, "r", encoding="utf-8") as f:
                return {line.split(",")[0]: int(line.split(",")[1]) for line in f.readlines()}
        elif self.token_dictionary_file.endswith(".pkl"):
            with open(self.token_dictionary_file, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的token字典文件格式: {self.token_dictionary_file}")
    
    def _create_tokenizer(self) -> BertTokenizer:
        """创建tokenizer"""
        # 检查是否在主进程中，避免多进程并发写入
        if hasattr(os, 'environ') and 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            if local_rank != 0:
                # 非主进程等待主进程创建文件
                import time
                while not (self.output_dir / "vocab.txt").exists():
                    time.sleep(1)
                return BertTokenizer.from_pretrained(
                    str(self.output_dir),
                    do_lower_case=False,
                    use_fast=False
                )
        
        # 主进程创建文件
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用文件锁避免并发写入
        lock_file = self.output_dir / "tokenizer.lock"
        if lock_file.exists():
            # 如果锁文件存在，等待其他进程完成
            import time
            while lock_file.exists():
                time.sleep(0.1)
            # 检查文件是否已经创建
            if (self.output_dir / "vocab.txt").exists() and (self.output_dir / "tokenizer_config.json").exists():
                return BertTokenizer.from_pretrained(
                    str(self.output_dir),
                    do_lower_case=False,
                    use_fast=False
                )
        
        # 创建锁文件
        lock_file.touch()
        
        try:
            # 创建词汇表文件
            vocab_file = self.output_dir / "vocab.txt"
            
            # 使用临时文件和原子操作避免并发写入问题
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.output_dir, encoding="utf-8") as temp_file:
                for token, token_id in sorted(self.token_dictionary.items(), key=lambda x: x[1]):
                    temp_file.write(f"{token}\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())  # 确保数据写入磁盘
                temp_path = temp_file.name
            
            # 原子性地移动文件
            try:
                shutil.move(temp_path, str(vocab_file))
            except Exception as e:
                # 如果移动失败，清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
            
            # 创建tokenizer配置
            tokenizer_config = {
                "model_max_length": self.max_length,
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                "cls_token": "<cls>",
                "sep_token": "<sep>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "padding_side": "right",
                "model_input_names": ["input_ids", "attention_mask"]
            }
            
            config_file = self.output_dir / "tokenizer_config.json"
            
            # 使用临时文件和原子操作避免并发写入问题
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.output_dir) as temp_file:
                json.dump(tokenizer_config, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # 确保数据写入磁盘
                temp_path = temp_file.name
            
            # 原子性地移动文件
            try:
                shutil.move(temp_path, str(config_file))
            except Exception as e:
                # 如果移动失败，清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
                
        finally:
            # 清理锁文件
            if lock_file.exists():
                lock_file.unlink()
        
        return BertTokenizer.from_pretrained(
            str(self.output_dir),
            do_lower_case=False,
            use_fast=False
        )


class ClassificationHead(nn.Module):
    """
    可选池化策略的分类头：
      - pool='cls'       : 取第0位 [CLS]
      - pool='mask_mean' : 基于 attention_mask 的均值池化
      - pool='attn'      : 可学习注意力池化（token加权平均）
    """
    
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1,
                 pool: str = "mask_mean"):
        super().__init__()
        assert pool in {"cls", "mask_mean", "attn"}
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        if pool == "attn":
            self.attn_vec = nn.Parameter(torch.randn(hidden_size))
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        hidden_states: [B, L, H]
        attention_mask: [B, L]，mask=1表示有效token；mask缺失时 attn/mask_mean 将退化为简单均值
        """
        B, L, H = hidden_states.shape
        if self.pool == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pool == "mask_mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
                summed = (hidden_states * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                pooled = summed / denom
        else:  # 'attn'
            a = self.attn_vec / (self.attn_vec.norm() + 1e-6)
            scores = torch.einsum("blh,h->bl", hidden_states, a)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = torch.einsum("bl,blh->bh", weights, hidden_states)

        pooled = self.norm(pooled)
        feats = self.proj(self.dropout(pooled))
        logits = self.classifier(feats)
        return logits





class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with optional class weights.

    Args:
        gamma: Focusing parameter that down-weights easy examples.
        weight: Optional class weights tensor of shape [num_classes].
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_per_example = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(logits, target)
        pt = torch.exp(-ce_per_example)
        focal = (1.0 - pt) ** self.gamma * ce_per_example
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


def compute_class_weights_from_dataset(dataset: Dataset, num_classes: int, beta: float = 0.5) -> torch.Tensor:
    """Compute smoothed inverse-frequency class weights from a HuggingFace Dataset.

    Uses column 'classification_labels'. Weights are normalized to sum to num_classes.
    """
    if 'classification_labels' not in dataset.column_names:
        raise ValueError("Dataset must contain 'classification_labels' column to compute class weights")
    labels = np.asarray(dataset['classification_labels'], dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.power(counts.astype(np.float64), beta)
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def make_loss_fn(
    dataset: Dataset,
    num_classes: int,
    device: torch.device,
    kind: str = 'weighted_ce',
    beta: float = 0.5,
    gamma: float = 2.0,
    use_class_weights: bool = True,
) -> nn.Module:
    """Factory to create CE / Weighted CE / Focal loss from dataset label distribution.

    kind: 'ce' | 'weighted_ce' | 'focal'
    beta: smoothing exponent for inverse-frequency weights
    gamma: focal focusing parameter
    """
    if kind == 'ce':
        return nn.CrossEntropyLoss()

    weight_tensor = None
    if use_class_weights:
        weight_tensor = compute_class_weights_from_dataset(dataset, num_classes=num_classes, beta=beta).to(device)
    if kind == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=weight_tensor)
    if kind == 'focal':
        return FocalLoss(gamma=gamma, weight=weight_tensor)

    raise ValueError(f"Unknown loss kind: {kind}")


class GeneBERTWithClassification(nn.Module):
    """结合BERT和分类头的模型"""
    
    def __init__(self, bert_model: BertForMaskedLM, classification_head: ClassificationHead, 
                 num_classes: int, label_to_id: Optional[Dict[str, int]] = None,
                 loss_fn: Optional[nn.Module] = None):
        super().__init__()
        self.bert = bert_model
        self.classification_head = classification_head
        self.num_classes = num_classes
        self.label_to_id = label_to_id or {}
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                classification_labels: Optional[torch.Tensor] = None,
                return_classification: bool = False) -> Dict[str, torch.Tensor]:
        # BERT前向传播
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        hidden_states = bert_outputs.hidden_states[-1]
        classification_logits = self.classification_head(hidden_states, attention_mask)
        assert classification_logits.dim() == 2 and classification_logits.size(1) == self.num_classes
        
        # 计算分类损失
        classification_loss = None
        if classification_labels is not None:
            # 确保labels的数据类型和取值范围正确
            assert classification_labels.dtype == torch.long, "classification_labels 必须是 Long"
            assert (0 <= classification_labels).all() and (classification_labels < self.num_classes).all(), \
                   "classification_labels 越界"
            
            # 直接使用logits和labels，无需view操作
            classification_loss = self.loss_fn(classification_logits, classification_labels)
        
        if return_classification:
            # 推理模式：返回输出字典（不构造 total_loss，避免与 Trainer.compute_loss 重复）
            return {
                'loss': bert_outputs.loss,
                'mlm_loss': bert_outputs.loss,
                'classification_logits': classification_logits,
                'classification_loss': classification_loss,
                'classification_predictions': torch.argmax(classification_logits, dim=-1),
                'hidden_states': hidden_states
            }
        else:
            # 训练模式：返回包含分类logits的输出，供训练完成后计算分类指标使用
            outputs = {
                'loss': bert_outputs.loss,
                'classification_logits': classification_logits,
                'classification_predictions': torch.argmax(classification_logits, dim=-1),
                'hidden_states': hidden_states
            }
            
            # 如果有分类损失，也包含在输出中
            if classification_loss is not None:
                outputs['classification_loss'] = classification_loss
            
            return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the underlying BERT model."""
        if hasattr(self, 'bert') and hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            print("Warning: BERT model does not support gradient checkpointing")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the underlying BERT model."""
        if hasattr(self, 'bert') and hasattr(self.bert, 'gradient_checkpointing_disable'):
            self.bert.gradient_checkpointing_disable()
        else:
            print("Warning: BERT model does not support gradient checkpointing")

    
    
    def save_pretrained(self, save_directory: str):
        """保存模型，包括分类头和标签映射"""
        # 保存BERT部分
        self.bert.save_pretrained(save_directory)
        
        # 保存分类头
        classification_head_path = os.path.join(save_directory, "classification_head")
        os.makedirs(classification_head_path, exist_ok=True)
        # 使用safetensors格式保存，更安全更快
        save_file(self.classification_head.state_dict(), 
                 os.path.join(classification_head_path, "model.safetensors"))
        
        # 保存分类头配置
        classification_config = {
            "hidden_size": self.classification_head.classifier.in_features,
            "num_classes": self.num_classes,
            "dropout": self.classification_head.dropout.p
        }
        with open(os.path.join(classification_head_path, "config.json"), "w") as f:
            json.dump(classification_config, f, indent=2)
        
        # 保存标签映射
        if self.label_to_id:
            label_mapping = {
                "label_to_id": self.label_to_id,
                "id_to_label": {v: k for k, v in self.label_to_id.items()},
                "num_classes": self.num_classes
            }
            with open(os.path.join(classification_head_path, "label_mapping.json"), "w") as f:
                json.dump(label_mapping, f, indent=2)
            print(f"Saved label mapping with {len(self.label_to_id)} classes to {classification_head_path}/label_mapping.json")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, num_classes: int):
        """从预训练模型加载，包括分类头和标签映射"""
        # 加载BERT模型
        bert_model = BertForMaskedLM.from_pretrained(pretrained_model_path)
        
        # 加载分类头
        classification_head_path = os.path.join(pretrained_model_path, "classification_head")
        label_to_id = {}
        
        if os.path.exists(classification_head_path):
            config_path = os.path.join(classification_head_path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            
            classification_head = ClassificationHead(
                hidden_size=config["hidden_size"],
                num_classes=num_classes,
                dropout=config["dropout"]
            )
            
            # 支持 safetensors 与 pytorch bin 两种格式
            st_path = os.path.join(classification_head_path, "model.safetensors")
            bin_path = os.path.join(classification_head_path, "pytorch_model.bin")
            if os.path.exists(st_path):
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(st_path)
                classification_head.load_state_dict(state_dict)
            elif os.path.exists(bin_path):
                classification_head.load_state_dict(torch.load(bin_path, map_location="cpu"))
            else:
                print("Warning: No classification head weights found; using randomly initialized head")
            
            # 加载标签映射
            label_mapping_path = os.path.join(classification_head_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, "r") as f:
                    label_mapping = json.load(f)
                label_to_id = label_mapping["label_to_id"]
                print(f"Loaded label mapping with {len(label_to_id)} classes from {label_mapping_path}")
            else:
                print("Warning: No label mapping file found")
        else:
            # 如果没有保存的分类头，创建新的
            classification_head = ClassificationHead(
                hidden_size=bert_model.config.hidden_size,
                num_classes=num_classes
            )
            print("Warning: No classification head found, created new one")
        
        return cls(bert_model, classification_head, num_classes, label_to_id)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: TrainingConfig, tokenizer_manager: TokenizerManager):
        self.config = config
        self.tokenizer_manager = tokenizer_manager
        self.model = None
    
    def create_model_config(self) -> BertConfig:
        """创建BERT模型配置"""
        return BertConfig(
            vocab_size=len(self.tokenizer_manager.token_dictionary),
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_act="gelu",
            max_length=self.config.max_length,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=self.config.max_length,
            type_vocab_size=1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=self.tokenizer_manager.token_dictionary.get("<pad>", 0),
            mask_token_id=self.tokenizer_manager.token_dictionary.get("<mask>", 1),
            cls_token_id=self.tokenizer_manager.token_dictionary.get("<cls>", 2),
            eos_token_id=self.tokenizer_manager.token_dictionary.get("<eos>", 3),
            sep_token_id=self.tokenizer_manager.token_dictionary.get("<sep>", 4),
            unk_token_id=self.tokenizer_manager.token_dictionary.get("<unk>", 5),
            model_type="bert",
            tie_word_embeddings=False
        )
    
    def create_model(self, label_to_id: Optional[Dict[str, int]] = None) -> nn.Module:
        """创建模型"""
        config = self.create_model_config()
        bert_model = BertForMaskedLM(config)
        
        if self.config.enable_classification:
            classification_head = ClassificationHead(
                hidden_size=config.hidden_size,
                num_classes=self.config.num_classes,
                pool=getattr(self.config, 'classification_pool', 'mask_mean')
            )
            self.model = GeneBERTWithClassification(
                bert_model=bert_model,
                classification_head=classification_head,
                num_classes=self.config.num_classes,
                label_to_id=label_to_id
            )
            print(f"Created model with classification head (num_classes={self.config.num_classes})")
            if label_to_id:
                print(f"Model includes label mapping with {len(label_to_id)} classes")
            print(f"Classification pooling strategy: {getattr(self.config, 'classification_pool', 'mask_mean')}")

        else:
            self.model = bert_model
            print("Created standard BERT model")
        
        # 根据 pos_encoding 注入 RoPE/ALiBi（仅BERT路径）
        pos_kind = getattr(self.config, 'pos_encoding', 'abs')
        if pos_kind in {"rope", "alibi"}:
            try:
                self._apply_positional_encoding_overrides(self.model, pos_kind)
                # 禁用绝对位置嵌入的梯度与初值影响
                try:
                    emb = self.model.bert.embeddings.position_embeddings
                    with torch.no_grad():
                        emb.weight.zero_()
                    emb.weight.requires_grad = False
                except Exception:
                    pass
                print(f"Applied positional encoding: {pos_kind}")
            except Exception as e:
                print(f"[WARN] Failed to apply {pos_kind}: {e}")

        return self.model

    def _apply_positional_encoding_overrides(self, model: nn.Module, pos_kind: str):
        # 遍历encoder层，替换自注意力为自定义实现
        target = model.bert if hasattr(model, 'bert') else model
        for layer in target.encoder.layer:
            orig_sa: BertSelfAttention = layer.attention.self
            layer.attention.self = CustomBertSelfAttention(orig_sa, pos_kind, max_position_embeddings=self.config.max_length)


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


class CustomBertSelfAttention(nn.Module):
    def __init__(self, orig: BertSelfAttention, pos_kind: str, max_position_embeddings: int):
        super().__init__()
        # Don't save self.orig to avoid duplicate parameter references in state_dict
        # self.orig = orig  # Removed to fix safetensors shared memory error
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
                past_key_values: Optional[Tuple[torch.Tensor]] = None, cache_position: Optional[torch.Tensor] = None,
                output_attentions: bool = False):
        # Hugging Face occasionally renames the cache argument; normalize here for compatibility
        if past_key_values is not None and past_key_value is None:
            past_key_value = past_key_values
        # cache_position is unused for vanilla BERT, but accept it to stay API-compatible
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
            # 规范化 attention_mask 为加性掩码形状 [B,1,1,L]，值为 0 或 -inf
            am = attention_mask
            with torch.no_grad():
                if am.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                    am = am.to(dtype=attention_scores.dtype)
                    am = (1.0 - am) * (-1e4)
                else:
                    # 已是浮点：假设可能已为加性掩码；若范围是 {0,1}，转换为 {0,-1e4}
                    maxv = float(am.max().item()) if am.numel() > 0 else 0.0
                    minv = float(am.min().item()) if am.numel() > 0 else 0.0
                    if minv >= 0.0 and maxv <= 1.0:
                        am = (1.0 - am) * (-1e4)
                    am = am.to(dtype=attention_scores.dtype)
                # 扩展到 [B,1,1,L]
                while am.dim() < 4:
                    am = am.unsqueeze(1)
                if am.size(-1) != attention_scores.size(-1):
                    am = am[..., :attention_scores.size(-1)]
            # 使用与原图分离的掩码副本
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


class DataManager:
    """数据管理器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def split_dataset(self) -> Tuple[DatasetDict, Dict[str, int]]:
        """
        拆分数据集，返回 (DatasetDict, label_to_id)
        支持稀有类处理策略：
          - self.config.rare_class_mode in {'other', 'filter', 'keep'}
          - self.config.min_count (缺省 2)
        其余配置：
          - train_ratio, val_ratio, test_ratio 之和应为 1
          - enable_classification: 是否启用分类任务
        """
        print("Loading dataset...")
        raw_ds = load_from_disk(self.config.dataset_path)
        df = raw_ds.to_pandas()

        # ---- v4 采样策略（占位实现，按需精化）----
        sampling_mode = getattr(self.config, 'sampling_mode', None)
        if sampling_mode in {"stratified", "temperature", "mix"}:
            print(f"[Sampling] mode={sampling_mode}")
            if sampling_mode == "stratified":
                quota = int(getattr(self.config, 'quota_per_class', 0) or 0)
                if quota > 0 and "labels" in df.columns:
                    df = (df.groupby("labels", group_keys=False)
                            .apply(lambda g: g.sample(n=min(len(g), quota), random_state=self.config.data_seed)))
                    print(f"[Sampling:stratified] quota={quota}, new_size={len(df):,}")
            elif sampling_mode == "temperature":
                beta = float(getattr(self.config, 'sampling_beta', 0.5) or 0.5)
                if "labels" in df.columns:
                    counts = df['labels'].value_counts()
                    probs = (1.0 / np.power(counts.reindex(df['labels']).values, beta))
                    probs = probs / probs.sum()
                    df = df.sample(n=len(df), replace=False, weights=probs, random_state=self.config.data_seed)
                    print(f"[Sampling:temperature] beta={beta}, reshuffled with weights")
            elif sampling_mode == "mix":
                r = float(getattr(self.config, 'mix_ratio', 0.7) or 0.7)
                if "labels" in df.columns:
                    # 全量 r 部分 + stratified (quota = median per-class)
                    counts = df['labels'].value_counts()
                    quota = int(np.median(counts.values))
                    df_bal = (df.groupby("labels", group_keys=False)
                                .apply(lambda g: g.sample(n=min(len(g), quota), random_state=self.config.data_seed)))
                    n_full = int(len(df) * r)
                    n_bal = len(df) - n_full
                    df = pd.concat([
                        df.sample(n=n_full, random_state=self.config.data_seed),
                        df_bal.sample(n=min(n_bal, len(df_bal)), random_state=self.config.data_seed)
                    ]).sample(frac=1.0, random_state=self.config.data_seed).reset_index(drop=True)
                    print(f"[Sampling:mix] r={r}, quota≈{quota}, new_size={len(df):,}")

        # ---- 数据采样（用于快速测试）----
        sample_ratio = float(getattr(self.config, "data_sample_ratio", 1.0))
        if sample_ratio < 1.0:
            original_size = len(df)
            # 检查是否有标签列
            has_labels_for_sampling = "labels" in df.columns
            # 使用分层采样确保每个类别都按比例采样
            if has_labels_for_sampling:
                from sklearn.model_selection import train_test_split
                # 检查是否有单样本类别
                label_counts = df['labels'].value_counts()
                single_sample_classes = label_counts[label_counts == 1]
                
                if len(single_sample_classes) > 0:
                    print(f"[数据采样] 发现 {len(single_sample_classes)} 个单样本类别，使用混合采样策略")
                    # 混合策略：对多样本类别使用分层采样，对单样本类别使用随机采样
                    multi_sample_mask = df['labels'].isin(label_counts[label_counts >= 2].index)
                    single_sample_mask = ~multi_sample_mask
                    
                    df_multi = df[multi_sample_mask]
                    df_single = df[single_sample_mask]
                    
                    if len(df_multi) > 0:
                        # 对多样本类别进行分层采样
                        df_multi_sampled, _ = train_test_split(
                            df_multi, 
                            train_size=sample_ratio, 
                            stratify=df_multi['labels'], 
                            random_state=42
                        )
                    else:
                        df_multi_sampled = df_multi
                    
                    if len(df_single) > 0:
                        # 对单样本类别进行随机采样
                        df_single_sampled = df_single.sample(frac=sample_ratio, random_state=42)
                    else:
                        df_single_sampled = df_single
                    
                    # 合并结果
                    df = pd.concat([df_multi_sampled, df_single_sampled]).reset_index(drop=True)
                    print(f"[数据采样] 混合采样: {original_size:,} → {len(df):,} 样本")
                else:
                    # 所有类别都有多个样本，可以直接分层采样
                    df, _ = train_test_split(
                        df, 
                        train_size=sample_ratio, 
                        stratify=df['labels'], 
                        random_state=42
                    )
                    print(f"[数据采样] 分层采样: {original_size:,} → {len(df):,} 样本")
            else:
                # 随机采样
                df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
                print(f"[数据采样] 随机采样: {original_size:,} → {len(df):,} 样本")
        else:
            print(f"[数据采样] 使用全部数据: {len(df):,} 样本")

        # ---- 基本检查 ----
        has_labels = "labels" in df.columns
        if not has_labels and self.config.enable_classification:
            print("Warning: enable_classification=True 但数据集中没有 'labels' 列；将按非分类任务处理。")
            self.config.enable_classification = False

        # ---- 配置读取 ----
        rare_mode = getattr(self.config, "rare_class_mode", "other")   # 'other' | 'filter' | 'keep'
        min_count = int(getattr(self.config, "min_count", 2))          # 合并/过滤阈值，默认2（满足sklearn分层要求）
        train_ratio = float(self.config.train_ratio)
        val_ratio   = float(self.config.val_ratio)
        test_ratio  = float(self.config.test_ratio)

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train/val/test 比例之和必须为 1"

        # ---- 输出基本数据信息 ----
        print("---- Dataset summary ----")
        print(f"  Total samples: {len(df)}")
        if has_labels:
            lbl_counts = df['labels'].value_counts()
            print(f"  Unique classes: {df['labels'].nunique()}")
            print(f"  Most common class: {lbl_counts.index[0]} ({lbl_counts.iloc[0]} samples)")
            print(f"  Classes with only 1 sample: {(lbl_counts == 1).sum()}")
            print(f"  Classes with ≤5 samples: {(lbl_counts <= 5).sum()}")
            
            # 显示 Top 10 标签统计
            print(f"\n  Top 10 labels by frequency:")
            top_10 = lbl_counts.head(10)
            total_samples = len(df)
            for i, (label, count) in enumerate(top_10.items(), 1):
                percentage = (count / total_samples) * 100
                print(f"    {i:2d}. {label}: {count:,} samples ({percentage:.1f}%)")
            
            # 基于数据分布给出 min_count 建议
            if len(top_10) >= 10:
                # 基于第10个类别的样本数给出建议
                min_count_suggestion = max(50, min(200, top_10.iloc[9] // 10))
                print(f"\n  💡 Suggested min_count: {min_count_suggestion} (based on data distribution)")
            else:
                print(f"\n  💡 Suggested min_count: 100 (default for balanced training)")

        # ---- 稀有类处理（仅在分类任务中生效）----
        if self.config.enable_classification:
            if rare_mode == "other":
                # 低频类合并到 'Other'
                counts = df['labels'].value_counts()
                head = set(counts[counts >= min_count].index)
                df['labels_mapped'] = df['labels'].where(df['labels'].isin(head), other='Other')
                print(f"[rare_mode=other] 使用 min_count={min_count}，"
                      f"head classes={len(head)}，低频类合并到 'Other'")
            elif rare_mode == "filter":
                # 低频类直接过滤
                counts = df['labels'].value_counts()
                keep = set(counts[counts >= min_count].index)
                before = len(df)
                df = df[df['labels'].isin(keep)].copy()
                df['labels_mapped'] = df['labels']
                print(f"[rare_mode=filter] 使用 min_count={min_count}，"
                      f"过滤样本数={before - len(df)}，保留类={len(keep)}")
            elif rare_mode == "keep":
                # 保留所有类（不建议评估）
                df['labels_mapped'] = df['labels']
                print(f"[rare_mode=keep] 不做合并/过滤，可能导致 Train/Val/Test 类集合不一致（不建议用于严谨评估）")
            else:
                raise ValueError(f"Unknown rare_class_mode: {rare_mode}")
        else:
            # 非分类任务无需映射
            df['labels_mapped'] = df['labels'] if has_labels else None

        # ---- 构建 label_to_id & num_classes（仅分类任务）----
        label_to_id: Dict[str, int] = {}
        if self.config.enable_classification:
            classes = sorted(df['labels_mapped'].unique())
            label_to_id = {c: i for i, c in enumerate(classes)}
            df['classification_labels'] = df['labels_mapped'].map(label_to_id).astype('int64')
            self.config.num_classes = len(classes)
            print(f"num_classes = {self.config.num_classes}；前10个类别编码示例：{dict(list(label_to_id.items())[:10])}")

        # ---- 分层拆分 ----
        def stratified_split(labels_np):
            """两段式分层：Train/Temp -> Val/Test，处理单样本类别"""
            from sklearn.model_selection import StratifiedShuffleSplit
            
            # 检查是否有单样本类别
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            single_sample_mask = counts == 1
            multi_sample_mask = counts >= 2
            
            if np.any(single_sample_mask):
                print(f"[分层拆分] 发现 {np.sum(single_sample_mask)} 个单样本类别，使用混合拆分策略")
                
                # 混合策略：对多样本类别使用分层拆分，对单样本类别使用随机拆分
                multi_sample_labels = unique_labels[multi_sample_mask]
                single_sample_labels = unique_labels[single_sample_mask]
                
                # 分离多样本和单样本的索引
                multi_idx = np.where(np.isin(labels_np, multi_sample_labels))[0]
                single_idx = np.where(np.isin(labels_np, single_sample_labels))[0]
                
                train_idx_list = []
                val_idx_list = []
                test_idx_list = []
                
                if len(multi_idx) > 0:
                    # 对多样本类别进行分层拆分
                    multi_labels = labels_np[multi_idx]
                    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_ratio), random_state=42)
                    multi_train_idx, multi_temp_idx = next(sss1.split(multi_idx, multi_labels))
                    
                    # Temp 再分 Val/Test（若 temp 中某类仅出现1次，则回退为随机拆分以避免sklearn报错）
                    temp_labels = multi_labels[multi_temp_idx]
                    test_size_rel = test_ratio / (val_ratio + test_ratio)
                    unique_temp, counts_temp = np.unique(temp_labels, return_counts=True)
                    if np.any(counts_temp < 2):
                        rng = np.random.default_rng(42)
                        perm = rng.permutation(len(multi_temp_idx))
                        n_test_rel = int(round(len(multi_temp_idx) * test_size_rel))
                        multi_test_idx_rel = perm[:n_test_rel]
                        multi_val_idx_rel = perm[n_test_rel:]
                    else:
                        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size_rel, random_state=42)
                        multi_val_idx_rel, multi_test_idx_rel = next(sss2.split(np.arange(len(multi_temp_idx)), temp_labels))
                    
                    train_idx_list.extend(multi_idx[multi_train_idx])
                    val_idx_list.extend(multi_idx[multi_temp_idx[multi_val_idx_rel]])
                    test_idx_list.extend(multi_idx[multi_temp_idx[multi_test_idx_rel]])
                
                if len(single_idx) > 0:
                    # 对单样本类别进行随机拆分
                    rng = np.random.default_rng(42)
                    single_perm = rng.permutation(len(single_idx))
                    n_train_single = int(len(single_idx) * train_ratio)
                    n_val_single = int(len(single_idx) * val_ratio)
                    
                    train_idx_list.extend(single_idx[single_perm[:n_train_single]])
                    val_idx_list.extend(single_idx[single_perm[n_train_single:n_train_single+n_val_single]])
                    test_idx_list.extend(single_idx[single_perm[n_train_single+n_val_single:]])
                
                return np.array(train_idx_list), np.array(val_idx_list), np.array(test_idx_list)
            else:
                # 所有类别都有多个样本，可以直接分层拆分
                sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_ratio), random_state=42)
                idx_all = np.arange(len(labels_np))
                train_idx, temp_idx = next(sss1.split(idx_all, labels_np))

                temp_labels = labels_np[temp_idx]
                # Temp 再分 Val/Test（若 temp 中某类仅出现1次，则回退为随机拆分）
                test_size_rel = test_ratio / (val_ratio + test_ratio)
                unique_temp, counts_temp = np.unique(temp_labels, return_counts=True)
                if np.any(counts_temp < 2):
                    rng = np.random.default_rng(42)
                    perm = rng.permutation(len(temp_idx))
                    n_test_rel = int(round(len(temp_idx) * test_size_rel))
                    test_idx_rel = perm[:n_test_rel]
                    val_idx_rel = perm[n_test_rel:]
                else:
                    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size_rel, random_state=42)
                    val_idx_rel, test_idx_rel = next(sss2.split(np.arange(len(temp_idx)), temp_labels))

                val_idx  = temp_idx[val_idx_rel]
                test_idx = temp_idx[test_idx_rel]
                return train_idx, val_idx, test_idx

        if self.config.enable_classification:
            # 用重映射标签做分层
            y = df['classification_labels'].to_numpy()
            train_idx, val_idx, test_idx = stratified_split(y)
        else:
            # 非分类任务，随机拆分
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(df))
            n_train = int(len(df) * train_ratio)
            n_val   = int(len(df) * val_ratio)
            train_idx = perm[:n_train]
            val_idx   = perm[n_train:n_train+n_val]
            test_idx  = perm[n_train+n_val:]

        # ---- 构建各 split 的 DataFrame ----
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        # ---- 打印分布（修复 Unique classes 统计）----
        def print_split_stats(name, sdf: pd.DataFrame):
            print(f"  {name.capitalize()}:")
            print(f"    Samples: {len(sdf)}")
            if self.config.enable_classification:
                print(f"    Unique classes: {sdf['labels_mapped'].nunique()}")
                vc = sdf['labels_mapped'].value_counts()
                print(f"    Classes with only 1 sample: {(vc == 1).sum()}")
                print(f"    Classes with ≤5 samples: {(vc <= 5).sum()}")

        print("Class distribution after split:")
        print_split_stats("train", train_df)
        print_split_stats("validation", val_df)
        print_split_stats("test", test_df)

        # ---- 转回 DatasetDict ----
        # 自动推断 features；若启用分类，尽量把标签列做成 ClassLabel
        def to_hf_dataset(sdf: pd.DataFrame) -> Dataset:
            sdf_out = sdf.copy()
            # 只在启用分类时保留 'classification_labels'；同时保留原始 'labels' 方便调试
            if self.config.enable_classification:
                # 构建 features（可选：若希望严格定义列类型）
                class_names = [None] * self.config.num_classes
                for k, v in label_to_id.items():
                    class_names[v] = k
                features = None
                # 如果想强制定义 features，可启用以下块（否则 datasets 会自动推断）
                # features = Features({
                #     **{col: Value("string") for col in sdf_out.columns if col not in ("classification_labels",)},
                #     "classification_labels": ClassLabel(num_classes=self.config.num_classes, names=class_names),
                # })
                ds = Dataset.from_pandas(sdf_out, preserve_index=False) if features is None \
                     else Dataset.from_pandas(sdf_out, preserve_index=False, features=features)
                return ds
            else:
                return Dataset.from_pandas(sdf_out, preserve_index=False)

        dataset_dict = DatasetDict({
            "train": to_hf_dataset(train_df),
            "validation": to_hf_dataset(val_df),
            "test": to_hf_dataset(test_df)
        })

        print("Dataset split complete:")
        print(f"  Train: {len(dataset_dict['train'])} samples")
        print(f"  Validation: {len(dataset_dict['validation'])} samples")
        print(f"  Test: {len(dataset_dict['test'])} samples")
        
        # 额外提示：若 rare_mode='keep'，评估时请明确说明 Val/Test 缺失的类不会影响 eval_loss，
        # 可能导致早停依据偏差；建议优先使用 'other' 或 'filter'。
        return dataset_dict, label_to_id
    
    def preprocess_function(self, examples: Dict[str, Any], tokenizer_manager: TokenizerManager) -> Dict[str, Any]:
        """预处理函数"""
        if "token_type_ids" in examples:
            print("发现 token_type_ids，跳过该样本")
            return {}

        input_ids = examples["input_ids"]
        vocab_size = len(tokenizer_manager.token_dictionary)
        
        # 检查input_ids是否超出vocab范围
        for seq in input_ids:
            max_id = max(seq)
            min_id = min(seq)
            if max_id >= vocab_size or min_id < 0:
                print(f"异常input_ids: {seq}")
                return {}

        # 处理序列长度
        processed_input_ids = []
        for seq in input_ids:
            if len(seq) > self.config.max_length:
                seq = seq[:self.config.max_length]
            elif len(seq) < self.config.max_length:
                seq = seq + [tokenizer_manager.token_dictionary.get("<pad>", 0)] * (self.config.max_length - len(seq))
            processed_input_ids.append(seq)
        
        # 创建attention mask
        attention_mask = []
        for seq in processed_input_ids:
            mask = [1 if token != tokenizer_manager.token_dictionary.get("<pad>", 0) else 0 for token in seq]
            attention_mask.append(mask)
        
        result = {
            "input_ids": processed_input_ids,
            "attention_mask": attention_mask
        }
        
        if self.config.enable_classification and "classification_labels" in examples:
            # 保留所有标签，包括样本不足的类别
            result["classification_labels"] = examples["classification_labels"]
        elif self.config.enable_classification and "labels" in examples:
            # 如果数据集中有原始标签，转换为数字ID
            # 这里假设标签已经在split_dataset中被处理过了
            result["classification_labels"] = examples["labels"]
        
        return result


class MetricsCalculator:
    """指标计算器 - 支持批处理、head/tail分析、top-k指标"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        # Head/Tail划分配置
        self.head_min_count = getattr(config, 'head_min_count', 50)  # 频次阈值
        self.use_coverage_head = getattr(config, 'use_coverage_head', False)  # 是否使用覆盖率法
        self.head_coverage = getattr(config, 'head_coverage', 0.9)  # 覆盖率阈值
        self.top_k = getattr(config, 'top_k_accuracy', 5)  # top-k准确率
    
    def compute_classification_metrics(self, model: nn.Module, tokenized_datasets: DatasetDict, 
                                     trainer: Trainer) -> Dict[str, Dict[str, float]]:
        """计算分类指标，包括micro-F1、macro-F1、head/tail分析和top-k指标"""
        results = {'validation': {}, 'test': {}}
        
        for split_name in ['validation', 'test']:
            print(f"Computing final classification metrics for {split_name} set...")
            
            dataset = tokenized_datasets[split_name]
            if 'classification_labels' not in dataset.column_names:
                print(f"Warning: No classification labels found in {split_name} dataset")
                continue
            
            # 使用DataLoader进行批处理，确保与训练时一致
            from torch.utils.data import DataLoader
            from transformers import default_data_collator
            
            eval_batch_size = getattr(trainer.args, 'per_device_eval_batch_size', self.config.batch_size)
            data_collator = getattr(trainer, 'data_collator', default_data_collator)
            
            dataloader = DataLoader(
                dataset, 
                batch_size=eval_batch_size,
                collate_fn=data_collator,
                shuffle=False
            )
            
            # 收集预测结果
            predictions = []
            logits_list = []
            model.eval()
            
            with torch.no_grad():
                for batch in dataloader:
                    # 确保tensor在正确设备上
                    batch = {k: v.to(trainer.args.device) for k, v in batch.items() 
                            if isinstance(v, torch.Tensor)}
                    
                    outputs = model(
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        return_classification=True
                    )
                    
                    batch_predictions = outputs['classification_predictions'].cpu()
                    batch_logits = outputs['classification_logits'].cpu()
                    
                    predictions.append(batch_predictions)
                    logits_list.append(batch_logits)
            
            # 合并所有预测结果
            predictions = torch.cat(predictions).numpy()[:len(dataset)]
            logits = torch.cat(logits_list).numpy()[:len(dataset)]
            true_labels = np.asarray(dataset['classification_labels'], dtype=int)
            
            # 数据质量检查
            self._check_data_quality(true_labels, predictions)
            
            # 计算基础指标
            accuracy = accuracy_score(true_labels, predictions)
            
            if self.config.detailed_classification_metrics:
                try:
                    # 计算多种F1指标
                    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                        true_labels, predictions, average='weighted', zero_division=0
                    )
                    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                        true_labels, predictions, average='micro', zero_division=0
                    )
                    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                        true_labels, predictions, average='macro', zero_division=0
                    )
                    
                    # Head/Tail分析
                    head_classes, head_mask, tail_mask = self._compute_head_tail_split(true_labels)
                    
                    # 计算head/tail指标
                    head_metrics = self._compute_block_metrics(true_labels, predictions, head_mask)
                    tail_metrics = self._compute_block_metrics(true_labels, predictions, tail_mask)
                    
                    # 计算top-k准确率
                    top_k_metrics = self._compute_top_k_metrics(logits, true_labels, head_mask, tail_mask)
                    
                except Exception as e:
                    print(f"Warning: Could not compute detailed metrics: {e}")
                    precision_weighted, recall_weighted, f1_weighted = 0.0, 0.0, 0.0
                    precision_micro, recall_micro, f1_micro = 0.0, 0.0, 0.0
                    precision_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0
                    head_metrics, tail_metrics, top_k_metrics = {}, {}, {}
            else:
                print(f"Detailed metrics disabled, using accuracy only")
                precision_weighted, recall_weighted, f1_weighted = None, None, None
                precision_micro, recall_micro, f1_micro = None, None, None
                precision_macro, recall_macro, f1_macro = None, None, None
                head_metrics, tail_metrics, top_k_metrics = {}, {}, {}
            
            # 组装结果
            if self.config.detailed_classification_metrics and precision_weighted is not None:
                results[split_name].update({
                    'classification_accuracy': accuracy,
                    'classification_precision_weighted': precision_weighted,
                    'classification_recall_weighted': recall_weighted,
                    'classification_f1_weighted': f1_weighted,
                    'classification_precision_micro': precision_micro,
                    'classification_recall_micro': recall_micro,
                    'classification_f1_micro': f1_micro,
                    'classification_precision_macro': precision_macro,
                    'classification_recall_macro': recall_macro,
                    'classification_f1_macro': f1_macro,
                })
                
                # 添加head/tail指标
                if head_metrics:
                    results[split_name].update({
                        'classification_head_accuracy': head_metrics['accuracy'],
                        'classification_head_precision': head_metrics['precision'],
                        'classification_head_recall': head_metrics['recall'],
                        'classification_head_f1': head_metrics['f1'],
                        'classification_head_support': head_metrics['support'],
                    })
                
                if tail_metrics:
                    results[split_name].update({
                        'classification_tail_accuracy': tail_metrics['accuracy'],
                        'classification_tail_precision': tail_metrics['precision'],
                        'classification_tail_recall': tail_metrics['recall'],
                        'classification_tail_f1': tail_metrics['f1'],
                        'classification_tail_support': tail_metrics['support'],
                    })
                
                # 添加top-k指标
                if top_k_metrics:
                    results[split_name].update(top_k_metrics)
                
                # 打印详细结果
                self._print_detailed_results(split_name, accuracy, f1_micro, f1_macro, f1_weighted,
                                          head_metrics, tail_metrics, top_k_metrics, head_classes)
            else:
                results[split_name].update({'classification_accuracy': accuracy})
                print(f"  {split_name} classification metrics:")
                print(f"     Accuracy: {accuracy:.4f}")
        
        return results
    
    def _check_data_quality(self, true_labels: np.ndarray, predictions: np.ndarray):
        """数据质量检查"""
        unique_classes = len(np.unique(true_labels))
        total_samples = len(true_labels)
        
        # 检查标签范围
        assert (0 <= true_labels).all() and (true_labels < self.config.num_classes).all(), \
               f"Labels out of range [0, {self.config.num_classes-1}]"
        
        # 统计单样本类别
        from collections import Counter
        label_counts = Counter(true_labels)
        single_sample_classes = sum(1 for count in label_counts.values() if count == 1)
        min_count_classes = sum(1 for count in label_counts.values() if count < self.head_min_count)
        
        print(f"  Classification info:")
        print(f"     Total samples: {total_samples}")
        print(f"     Unique classes: {unique_classes}")
        print(f"     Single-sample classes: {single_sample_classes} ({single_sample_classes/unique_classes*100:.1f}%)")
        print(f"     Classes < {self.head_min_count} samples: {min_count_classes} ({min_count_classes/unique_classes*100:.1f}%)")
        
        # 长尾分布检查
        if single_sample_classes / unique_classes > 0.3:
            print(f"Warning: High proportion of single-sample classes ({single_sample_classes/unique_classes*100:.1f}%)")
        if min_count_classes / unique_classes > 0.8:
            print(f"Warning: Most classes have < {self.head_min_count} samples ({min_count_classes/unique_classes*100:.1f}%)")
    
    def _compute_head_tail_split(self, true_labels: np.ndarray):
        """计算head/tail划分"""
        from collections import Counter
        
        cnt = Counter(true_labels)
        items = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        freqs = np.array([c for _, c in items])
        labels_sorted = np.array([l for l, _ in items])
        
        if self.use_coverage_head:
            # 覆盖率法：覆盖到指定比例的样本
            cum = freqs.cumsum() / freqs.sum()
            k = int(np.searchsorted(cum, self.head_coverage)) + 1
            head_classes = set(labels_sorted[:k])
        else:
            # 频次阈值法
            head_classes = set(labels_sorted[freqs >= self.head_min_count])
        
        head_mask = np.isin(true_labels, list(head_classes))
        tail_mask = ~head_mask
        
        return head_classes, head_mask, tail_mask
    
    def _compute_block_metrics(self, true_labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray):
        """计算指定mask区域的指标"""
        if mask.sum() == 0:
            return dict(precision=0.0, recall=0.0, f1=0.0, accuracy=0.0, support=0)
        
        masked_labels = true_labels[mask]
        masked_predictions = predictions[mask]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            masked_labels, masked_predictions, average='macro', zero_division=0
        )
        accuracy = accuracy_score(masked_labels, masked_predictions)
        
        return dict(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            support=int(mask.sum())
        )
    
    def _compute_top_k_metrics(self, logits: np.ndarray, true_labels: np.ndarray, 
                              head_mask: np.ndarray, tail_mask: np.ndarray):
        """计算top-k准确率"""
        logits_tensor = torch.tensor(logits)
        true_labels_tensor = torch.tensor(true_labels)
        
        # 总体top-k
        top_k_indices = logits_tensor.topk(self.top_k, dim=-1).indices
        top_k_hits = (top_k_indices == true_labels_tensor.unsqueeze(-1)).any(dim=-1).float().mean()
        
        metrics = {f'classification_top_{self.top_k}_accuracy': top_k_hits.item()}
        
        # Head top-k
        if head_mask.sum() > 0:
            head_logits = logits_tensor[head_mask]
            head_labels = true_labels_tensor[head_mask]
            head_top_k_indices = head_logits.topk(self.top_k, dim=-1).indices
            head_top_k_hits = (head_top_k_indices == head_labels.unsqueeze(-1)).any(dim=-1).float().mean()
            metrics[f'classification_head_top_{self.top_k}_accuracy'] = head_top_k_hits.item()
        
        # Tail top-k
        if tail_mask.sum() > 0:
            tail_logits = logits_tensor[tail_mask]
            tail_labels = true_labels_tensor[tail_mask]
            tail_top_k_indices = tail_logits.topk(self.top_k, dim=-1).indices
            tail_top_k_hits = (tail_top_k_indices == tail_labels.unsqueeze(-1)).any(dim=-1).float().mean()
            metrics[f'classification_tail_top_{self.top_k}_accuracy'] = tail_top_k_hits.item()
        
        return metrics
    
    def _print_detailed_results(self, split_name: str, accuracy: float, f1_micro: float, 
                               f1_macro: float, f1_weighted: float, head_metrics: dict, 
                               tail_metrics: dict, top_k_metrics: dict, head_classes: set):
        """打印详细结果"""
        print(f"  {split_name} classification metrics:")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     Micro-F1: {f1_micro:.4f}")
        print(f"     Macro-F1: {f1_macro:.4f}")
        print(f"     Weighted-F1: {f1_weighted:.4f}")
        
        if top_k_metrics:
            print(f"     Top-{self.top_k} Accuracy: {top_k_metrics[f'classification_top_{self.top_k}_accuracy']:.4f}")
        
        if head_metrics:
            print(f"     Head Classes ({len(head_classes)} classes, {head_metrics['support']} samples):")
            print(f"       Accuracy: {head_metrics['accuracy']:.4f}, F1: {head_metrics['f1']:.4f}")
            if f'classification_head_top_{self.top_k}_accuracy' in top_k_metrics:
                print(f"       Top-{self.top_k}: {top_k_metrics[f'classification_head_top_{self.top_k}_accuracy']:.4f}")
        
        if tail_metrics:
            print(f"     Tail Classes ({self.config.num_classes - len(head_classes)} classes, {tail_metrics['support']} samples):")
            print(f"       Accuracy: {tail_metrics['accuracy']:.4f}, F1: {tail_metrics['f1']:.4f}")
            if f'classification_tail_top_{self.top_k}_accuracy' in top_k_metrics:
                print(f"       Top-{self.top_k}: {top_k_metrics[f'classification_tail_top_{self.top_k}_accuracy']:.4f}")


class CustomDataCollator(DataCollatorForLanguageModeling):
    """自定义数据整理器，支持分类标签；支持 token/span 两种动态掩码"""
    
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15,
                 mlm_mask_style: str = "token", span_length: int = 5):
        super().__init__(tokenizer, mlm, mlm_probability)
        assert mlm_mask_style in {"token", "span"}
        self.mlm_mask_style = mlm_mask_style
        self.span_length = max(1, int(span_length))
    
    def __call__(self, features):
        # 动态掩码：按需使用父类（token级掩码）或自定义span掩码
        if self.mlm and self.mlm_mask_style == "span":
            batch = self._torch_call_span_masking(features)
        else:
            batch = super().__call__(features)
        
        # 添加分类标签（如果存在）
        if "classification_labels" in features[0]:
            batch["classification_labels"] = torch.tensor([
                f["classification_labels"] for f in features
            ])
        
        return batch

    def _torch_call_span_masking(self, features):
        # 参照父类实现，改用span级别随机掩码（每batch动态重采样）
        import torch
        from torch.nn.utils.rnn import pad_sequence
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_input = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = batch_input.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=labels.device)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 避免 in-place 操作：使用 masked_fill 的非 in-place 版本
        probability_matrix = probability_matrix.masked_fill(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), 0.0)

        # 生成span mask
        batch_size, seq_len = labels.shape
        span_mask = torch.zeros_like(labels, dtype=torch.bool)
        for b in range(batch_size):
            can_mask = (probability_matrix[b] > 0)
            target_tokens = int(self.mlm_probability * can_mask.sum().item())
            remaining = target_tokens
            attempts = 0
            while remaining > 0 and attempts < seq_len * 2:
                attempts += 1
                max_len = min(self.span_length, remaining) if remaining > 0 else self.span_length
                span_len = max(1, int(torch.randint(1, max_len + 1, (1,)).item()))
                start = int(torch.randint(0, max(1, seq_len - span_len + 1), (1,)).item())
                segment = torch.arange(start, min(seq_len, start + span_len), device=labels.device)
                valid = can_mask[segment] & (~span_mask[b, segment])
                if valid.any():
                    # 避免 in-place 操作：创建新的 mask 而不是就地修改
                    new_mask = span_mask[b].clone()
                    new_mask[segment] = new_mask[segment] | valid
                    span_mask[b] = new_mask
                    remaining -= int(valid.sum().item())

        masked_indices = span_mask & (probability_matrix > 0)
        # 避免 in-place 操作：使用 where 创建新的 labels tensor
        labels = torch.where(~masked_indices, torch.tensor(-100, device=labels.device), labels)

        # 80% [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        # 避免 in-place 操作：使用 where 创建新的 batch_input tensor
        batch_input = torch.where(indices_replaced, torch.tensor(self.tokenizer.mask_token_id, device=batch_input.device), batch_input)
        
        # 10% random
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        # 避免 in-place 操作：使用 where 创建新的 batch_input tensor
        batch_input = torch.where(indices_random, random_words, batch_input)
        # 10% keep original: 留空即可

        attention_mask = (batch_input != self.tokenizer.pad_token_id).long()
        return {"input_ids": batch_input, "labels": labels, "attention_mask": attention_mask}


class LossLoggingCallback(TrainerCallback):
    """
    记录 train 阶段的 mlm/class/total（来自 compute_loss 的 model.last_losses），
    以及 eval/test 阶段的整体 loss（来自 logs['eval_loss'] / logs['test_loss']）。
    支持分布式只在主进程写日志。
    """

    def __init__(self, get_lambda_fn=None, log_dir: str = None, flush_every: int = 100):
        """
        get_lambda_fn: 可选函数，用于获取当前 λ（classification_weight），例如：
            lambda: trainer.args.classification_weight 或 从 model/config 读取；
            若为 None，则不单独记录 λ。
        log_dir: TensorBoard 日志目录；若 None，用 Trainer 默认的日志器即可（但这里我们显式用 SummaryWriter）。
        flush_every: 每多少次写入后 flush 一次。
        """
        self.get_lambda_fn = get_lambda_fn
        self.writer = None
        self.log_dir = log_dir
        self.flush_every = flush_every
        self._writes = 0

    # ---------- 生命周期 ----------
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return control
        if self.log_dir:
            self.writer = SummaryWriter(self.log_dir)
            print(f"[TB] Initialized SummaryWriter at {self.log_dir}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return control
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            print("[TB] Closed SummaryWriter")
        return control

    # ---------- 训练/评估统一日志入口 ----------
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 只在主进程写
        if not state.is_local_process_zero or self.writer is None or logs is None:
            return control

        step = state.global_step

        # 尝试访问底层模型（DistributedDataParallel 包装了模型）
        actual_model = model
        if hasattr(model, 'module'):
            actual_model = model.module

        # 1) 训练阶段：从 model.last_losses 读取三路 loss（由 compute_loss 写入）
        #    在训练和评估阶段都记录，但评估阶段可能没有 model.last_losses
        is_eval_log = any(k.startswith("eval_") for k in logs.keys())
        is_test_log = any(k.startswith("test_") for k in logs.keys())

        if hasattr(actual_model, "last_losses") and actual_model.last_losses is not None:
            last = actual_model.last_losses
            if last.get("mlm_loss") is not None:
                self._add_scalar("Loss/train_mlm", last["mlm_loss"].item(), step)
            if last.get("classification_loss") is not None:
                self._add_scalar("Loss/train_classification", last["classification_loss"].item(), step)
                # 记录 weighted 分类损失（使用当前 λ）
                if self.get_lambda_fn is not None:
                    curr_lambda = float(self.get_lambda_fn())
                    self._add_scalar("Loss/train_weighted_classification", curr_lambda * last["classification_loss"].item(), step)
            if last.get("total_loss") is not None:
                self._add_scalar("Loss/train_total", last["total_loss"].item(), step)

        # 2) 评估/测试阶段：从 logs 里取整体 loss
        if "eval_loss" in logs:
            self._add_scalar("Loss/eval_total", float(logs["eval_loss"]), step)
        if "test_loss" in logs:
            self._add_scalar("Loss/test_total", float(logs["test_loss"]), step)

        # 3) 记录当前 λ：优先从 model.last_losses 读（真实 warmup 后 λ），否则退回 get_lambda_fn
        try:
            if hasattr(model, "last_losses") and model.last_losses is not None and "curr_lambda" in model.last_losses:
                curr_lambda = float(model.last_losses["curr_lambda"].item() if hasattr(model.last_losses["curr_lambda"], 'item') else model.last_losses["curr_lambda"])
                self._add_scalar("Hyperparams/classification_weight", curr_lambda, step)
            elif self.get_lambda_fn is not None:
                curr_lambda = float(self.get_lambda_fn())
                self._add_scalar("Hyperparams/classification_weight", curr_lambda, step)
        except Exception:
            pass

        return control

    # ---------- 小工具 ----------
    def _add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self._writes += 1
        if self._writes % self.flush_every == 0:
            self.writer.flush()


class CustomTrainer(Trainer):
    """自定义训练器，支持分类损失"""
    
    def __init__(self, *args, classification_weight=0.0, simcse_alpha: float = 0.0, simcse_tau: float = 0.1, simcse_pool: str = "mask_mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_weight = classification_weight
        self.simcse_alpha = float(simcse_alpha)
        self.simcse_tau = float(simcse_tau)
        assert simcse_pool in {"mask_mean", "attn"}
        self.simcse_pool = simcse_pool
        # one-time warnings
        self._simcse_attn_warned = False
        self._simcse_hidden_states_warned = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算损失，包括分类损失，并在输出中提供 mlm/class/total 三类loss以便记录"""
        # 提取分类标签（完全避免修改 inputs）
        classification_labels = inputs.get("classification_labels", None)
        # 创建新的 inputs 字典，排除 classification_labels
        inputs_clean = {k: v for k, v in inputs.items() if k != "classification_labels"}
        
        # 前向传播 - 传递分类标签给模型
        if classification_labels is not None:
            outputs = model(**inputs_clean, classification_labels=classification_labels)
        else:
            outputs = model(**inputs_clean)
        
        # 基础损失（MLM损失）
        mlm_loss = outputs["loss"]
        classification_loss = outputs.get("classification_loss", None)
        
        # 计算当前λ（支持warmup），并用于总损失
        curr_lambda = self._get_current_lambda() if (self.classification_weight > 0 and classification_loss is not None) else 0.0
        total_loss = mlm_loss + curr_lambda * classification_loss if classification_loss is not None else mlm_loss

        # v5: SimCSE 对比学习（序列级），α>0 时启用：再次前向生成另一"视图"，计算 InfoNCE
        if self.simcse_alpha and self.simcse_alpha > 0:
            # 使用基于 dropout 的第二视图，避免第二次前向引入任何共享张量的就地修改风险
            hidden_state1 = outputs.get('hidden_states', None)
            if isinstance(hidden_state1, (list, tuple)) and len(hidden_state1) > 0:
                hidden_state1 = hidden_state1[-1]
            if hidden_state1 is not None:
                # 复制第一视图的隐藏态并施加随机 dropout 作为第二视图（不参与梯度）
                dropout_layer = nn.Dropout(p=0.1)
                with torch.no_grad():
                    hidden_state2 = dropout_layer(hidden_state1.detach().clone())
                # 使用 inputs_clean 中的 attention_mask
                attn_mask = inputs_clean.get('attention_mask', None)
                def pool(h):
                    if self.simcse_pool == "mask_mean":
                        if attn_mask is None:
                            return h.mean(dim=1)
                        # 使用与原图分离的 mask，避免任何下游操作影响原始张量版本
                        mask = attn_mask.detach().clone()
                        if isinstance(mask, (list, tuple)):
                            mask = mask[0]
                        mask = mask.to(dtype=h.dtype, device=h.device).unsqueeze(-1)
                        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                    # simcse_pool == 'attn'
                    # 若存在可学习注意力向量，则使用 masked softmax 加权池化
                    attn_vec = None
                    try:
                        actual_model = model.module if hasattr(model, 'module') else model
                        if hasattr(actual_model, 'classification_head') and hasattr(actual_model.classification_head, 'attn_vec'):
                            attn_vec = actual_model.classification_head.attn_vec
                    except Exception:
                        attn_vec = None
                    if attn_vec is not None:
                        a = attn_vec / (attn_vec.norm() + 1e-6)
                        # scores: [B, L]
                        scores = torch.einsum("blh,h->bl", h, a)
                        if attn_mask is not None:
                            am = attn_mask.detach().clone()
                            if isinstance(am, (list, tuple)):
                                am = am[0]
                            scores = scores.masked_fill(am == 0, float("-inf"))
                        weights = torch.softmax(scores, dim=1)
                        return torch.einsum("bl,blh->bh", weights, h)
                    # 否则回退为 mask 均值
                    if (self.simcse_pool == 'attn') and (not self._simcse_attn_warned):
                        try:
                            print("[SimCSE] simcse_pool=attn 但未找到 classification_head.attn_vec；回退为 mask-mean 池化")
                        except Exception:
                            pass
                        self._simcse_attn_warned = True
                    if attn_mask is None:
                        return h.mean(dim=1)
                    mask = attn_mask.detach().clone()
                    if isinstance(mask, (list, tuple)):
                        mask = mask[0]
                    mask = mask.to(dtype=h.dtype, device=h.device).unsqueeze(-1)
                    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                z1 = nn.functional.normalize(pool(hidden_state1), dim=-1)
                # 仅通过 z1 反向传播，对比视图 z2 不参与梯度
                z2 = nn.functional.normalize(pool(hidden_state2).detach(), dim=-1)
                logits_c = torch.matmul(z1, z2.t()) / max(1e-6, self.simcse_tau)
                labels_c = torch.arange(logits_c.size(0), device=logits_c.device)
                nce = nn.CrossEntropyLoss()(logits_c, labels_c)
                total_loss = total_loss + self.simcse_alpha * nce
            else:
                if not self._simcse_hidden_states_warned:
                    try:
                        print("[SimCSE] 未获得 hidden_states，已跳过对比损失（这不影响常规MLM/分类训练）")
                    except Exception:
                        pass
                    self._simcse_hidden_states_warned = True
        
        # 保存输出供回调函数使用（detach 以便记录）
        outputs["mlm_loss"] = mlm_loss.detach()
        if classification_loss is not None:
            outputs["classification_loss"] = classification_loss.detach()
        outputs["total_loss"] = total_loss.detach()
        outputs["curr_lambda"] = torch.tensor(curr_lambda)
        
        # 在所有阶段都设置 last_losses，以便 LossLoggingCallback 可以访问
        # 需要设置到实际的模型上，而不是 DistributedDataParallel 包装器
        actual_model = model
        if hasattr(model, 'module'):
            actual_model = model.module
        
        actual_model.last_losses = {
            "mlm_loss": outputs["mlm_loss"],
            "classification_loss": outputs.get("classification_loss"),
            "total_loss": outputs["total_loss"],
            "curr_lambda": outputs["curr_lambda"],
        }
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _infer_total_steps(self):
        """推断总训练步数"""
        # 优先使用用户指定的 max_steps
        if getattr(self.args, "max_steps", 0) and self.args.max_steps > 0:
            return int(self.args.max_steps)

        # 否则用 epoch * dataloader size 估算
        try:
            dl_len = len(self.get_train_dataloader())
            grad_accum = getattr(self.args, "gradient_accumulation_steps", 1)
            total_steps = (dl_len // grad_accum) * int(self.args.num_train_epochs)
            return max(1, total_steps)
        except Exception:
            # 如果无法获取dataloader，使用当前步数估算
            return max(1, int(self.state.global_step + 1))

    def _get_current_lambda(self):
        """获取当前的λ值，支持warmup"""
        # 1) 比例优先：如果传入比例，则使用总步数 * ratio 进行warmup
        warmup_ratio = getattr(self.args, 'classification_weight_warmup_ratio', None)
        if warmup_ratio is not None:
            total_steps = self._infer_total_steps()
            warmup_steps = max(1, int(total_steps * float(warmup_ratio)))
        else:
            # 2) 固定步数
            if not hasattr(self.args, 'classification_weight_warmup_steps') or self.args.classification_weight_warmup_steps == 0:
                return self.classification_weight
            warmup_steps = int(self.args.classification_weight_warmup_steps)
        
        current_step = self.state.global_step
        
        if current_step < warmup_steps:
            # 线性warmup
            return self.classification_weight * (current_step / warmup_steps)
        else:
            return self.classification_weight


class GeneBERTTrainer:
    """GeneBERT训练器主类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        if self.config.report_to is None:
            self.config.report_to = ["tensorboard"]
        # 为梯度检查点提供安全默认：使用非重入，以避免 DDP "ready twice" 问题
        if getattr(self.config, 'gradient_checkpointing', False) and getattr(self.config, 'gradient_checkpointing_kwargs', None) is None:
            self.config.gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个管理器
        self.tokenizer_manager = TokenizerManager(
            config.token_dictionary_file, 
            self.output_dir, 
            config.max_length
        )
        self.model_manager = ModelManager(config, self.tokenizer_manager)
        self.data_manager = DataManager(config)
        self.metrics_calculator = MetricsCalculator(config)
        
        # 训练相关对象
        self.model = None
        self.trainer = None
    
    def train(self) -> Tuple[Trainer, Dict[str, Dict[str, float]]]:
        """训练模型"""
        print("Starting GeneBERT training...")
        # 可选启用 PyTorch 异常检测以定位 in-place 修改来源
        try:
            if os.environ.get("TORCH_ANOMALY", "0") == "1":
                torch.autograd.set_detect_anomaly(True)
                print("[Debug] torch.autograd anomaly detection enabled (TORCH_ANOMALY=1)")
        except Exception:
            pass
        
        # 拆分数据集
        dataset_dict, label_to_id = self.data_manager.split_dataset()
        
        # 预处理数据集
        print("Preprocessing datasets...")
        tokenized_datasets = dataset_dict.map(
            lambda examples: self.data_manager.preprocess_function(examples, self.tokenizer_manager),
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        # 创建模型
        print("Creating model...")
        self.model = self.model_manager.create_model(label_to_id)
        # 确保输出 hidden_states（无分类/有分类都开启），供 SimCSE 使用
        try:
            if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'config'):
                self.model.bert.config.output_hidden_states = True
            elif hasattr(self.model, 'config'):
                self.model.config.output_hidden_states = True
        except Exception:
            pass
        
        # 创建数据整理器
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer_manager.tokenizer,
            mlm=True,
            mlm_probability=getattr(self.config, 'mlm_probability', 0.15),
            mlm_mask_style=getattr(self.config, 'mlm_mask_style', 'token'),
            span_length=getattr(self.config, 'span_length', 5)
        )
        
        # 设置训练参数（避免将 None 传给 warmup_ratio）
        training_args_kwargs = dict(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            lr_scheduler_type=getattr(self.config, 'lr_scheduler_type', 'linear'),
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            max_grad_norm=getattr(self.config, 'max_grad_norm', 1.0),
            eval_strategy=getattr(self.config, 'evaluation_strategy', 'steps'),
            save_strategy=getattr(self.config, 'save_strategy', 'steps'),
            logging_dir=str(self.output_dir / "tf_logs"),
            report_to=getattr(self.config, 'report_to', ["tensorboard"]),
            dataloader_pin_memory=False,
            push_to_hub=False,
            ddp_find_unused_parameters=False,
            save_safetensors=False,
            save_on_each_node=True,
            fp16=True if (getattr(self.config, 'fp16', False)) else (True if self.config.deepspeed else False),
            bf16=True if getattr(self.config, 'bf16', False) else False,
            gradient_accumulation_steps=getattr(self.config, 'gradient_accumulation_steps', 1),
            gradient_checkpointing=getattr(self.config, 'gradient_checkpointing', False),
            gradient_checkpointing_kwargs=getattr(self.config, 'gradient_checkpointing_kwargs', None),
            remove_unused_columns=False,
            **({"deepspeed": self.config.deepspeed, "local_rank": self.config.local_rank} if self.config.deepspeed else {})
        )
        if getattr(self.config, 'warmup_ratio', None) is not None:
            training_args_kwargs["warmup_ratio"] = float(self.config.warmup_ratio)
        training_args = TrainingArguments(**training_args_kwargs)
        
        # 创建回调函数列表
        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        
        # 如果启用分类，添加loss记录回调
        if self.config.enable_classification and self.config.classification_weight > 0:
            # 创建lambda获取函数，支持动态权重和warmup
            def get_lambda():
                if hasattr(self.trainer, '_get_current_lambda'):
                    return self.trainer._get_current_lambda()
                return self.config.classification_weight
            
            loss_callback = LossLoggingCallback(
                get_lambda_fn=get_lambda,
                log_dir=str(self.output_dir / "tf_logs"),
                flush_every=50  # 每50次写入flush一次
            )
            callbacks.append(loss_callback)
            print(f"Added LossLoggingCallback with classification_weight={self.config.classification_weight}")
        else:
            print(f"Not adding LossLoggingCallback: enable_classification={self.config.enable_classification}, classification_weight={self.config.classification_weight}")
        
        # 如启用分类且指定了loss_kind，则构建并注入自定义损失函数
        if self.config.enable_classification and getattr(self.config, 'loss_kind', None):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loss_fn = make_loss_fn(
                    dataset=tokenized_datasets["train"],
                    num_classes=self.config.num_classes,
                    device=device,
                    kind=self.config.loss_kind,
                    beta=getattr(self.config, 'class_weight_beta', 0.5),
                    gamma=getattr(self.config, 'focal_gamma', 2.0),
                    use_class_weights=getattr(self.config, 'use_class_weights', True),
                )
                if hasattr(self.model, 'loss_fn'):
                    self.model.loss_fn = loss_fn
                    print(f"Injected custom classification loss: kind={self.config.loss_kind}")
            except Exception as e:
                print(f"Warning: Failed to build custom loss_fn ({self.config.loss_kind}): {e}")

        # 创建训练器
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            callbacks=callbacks,
            classification_weight=self.config.classification_weight,
            simcse_alpha=getattr(self.config, 'simcse_alpha', 0.0),
            simcse_tau=getattr(self.config, 'simcse_tau', 0.1),
            simcse_pool=getattr(self.config, 'simcse_pool', 'mask_mean')
        )
        # 若使用按比例warmup，预估总步数并打印换算warmup步数
        if getattr(self.config, 'classification_weight_warmup_ratio', None) is not None:
            try:
                total_steps = self.trainer._infer_total_steps()
                warmup_steps_est = int(total_steps * float(self.config.classification_weight_warmup_ratio))
                print(f"Lambda warmup (ratio={self.config.classification_weight_warmup_ratio}): total_steps≈{total_steps}, warmup_steps≈{warmup_steps_est}")
            except Exception as e:
                print(f"Warning: could not estimate total steps for warmup ratio: {e}")
        
        # 开始训练
        print("Starting training...")
        train_result = self.trainer.train()
        
        # 保存最终模型
        print("Saving final model...")
        final_model_path = self.output_dir / "final_model"
        
        try:
            # 使用自定义的保存方法
            if self.config.enable_classification:
                self.model.save_pretrained(str(final_model_path))
            else:
                self.model.save_pretrained(str(final_model_path), safe_serialization=True)
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
            # 手动保存模型，使用safetensors格式
            save_file(self.model.state_dict(), str(final_model_path / "model.safetensors"))
            if hasattr(self.model, 'config'):
                self.model.config.save_pretrained(str(final_model_path))
        
        # 保存tokenizer
        self.tokenizer_manager.tokenizer.save_pretrained(str(final_model_path))
        
        # 保存 generation_config.json 以消除警告
        generation_config = {
            "max_length": self.config.max_length,
            "do_sample": False,
            "num_beams": 1,
            "early_stopping": False,
            "pad_token_id": self.tokenizer_manager.token_dictionary.get("<pad>", 0),
            "eos_token_id": self.tokenizer_manager.token_dictionary.get("<eos>", 3),
        }
        generation_config_file = final_model_path / "generation_config.json"
        with open(generation_config_file, "w") as f:
            json.dump(generation_config, f, indent=2)
        
        # 计算参数量
        print("Calculating model parameters...")
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 计算核心模型参数量（不包含MLM头）：直接从实际模型中减去 MLM 头参数
        # 兼容两种模型包装：GeneBERTWithClassification(包含 .bert: BertForMaskedLM) 或 直接 BertForMaskedLM
        bert_mlm_module = self.model.bert if hasattr(self.model, 'bert') else self.model
        if hasattr(bert_mlm_module, 'cls') and hasattr(bert_mlm_module.cls, 'predictions'):
            tie_word_embeddings = bool(getattr(bert_mlm_module.config, 'tie_word_embeddings', True))
            mlm_head_params = 0
            for name, param in bert_mlm_module.cls.named_parameters():
                # decoder.weight 在 tie_word_embeddings=True 时与词嵌入共享，不应重复计入
                if name == 'predictions.decoder.weight' and tie_word_embeddings:
                    continue
                mlm_head_params += param.numel()
        else:
            # 如果没有MLM头（极少数情况），则认为额外为0
            mlm_head_params = 0

        core_model_params = total_params - mlm_head_params
        
        print(f"Training Model Params (with MLM head): {total_params:,} ({total_params/1_000_000:.2f}M)")
        print(f"Core Model Params (without MLM head): {core_model_params:,} ({core_model_params/1_000_000:.2f}M)")
        print(f"MLM Head Params: {total_params - core_model_params:,} ({(total_params - core_model_params)/1_000_000:.2f}M)")
        
        # 保存完整的模型配置信息
        model_config_info = {
            "model_type": "GeneBERTWithClassification" if self.config.enable_classification else "BertForMaskedLM",
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
            "max_length": self.config.max_length,
            "vocab_size": len(self.tokenizer_manager.token_dictionary),
            "classification_config": {
                "enabled": self.config.enable_classification,
                "num_classes": self.config.num_classes if self.config.enable_classification else None
            },
            "parameter_counts": {
                "training_total_params": total_params,
                "core_model_params": core_model_params,
                "mlm_head_params": total_params - core_model_params
            }
        }
        
        config_file = final_model_path / "model_config.json"
        with open(config_file, "w") as f:
            json.dump(model_config_info, f, indent=2)

        print(f"Training completed! Final model saved to: {final_model_path}")
        if self.config.enable_classification:
            print(f"Classification head is saved in {final_model_path}/classification_head/")
        
        
        # 保存训练结果
        print("\n" + "="*60)
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # 评估最终模型
        print("\n" + "="*60)
        print("Evaluating final model...")
        eval_results = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_results)
        self.trainer.save_metrics("eval", eval_results)
        
        # 计算MLM困惑度
        if 'eval_loss' in eval_results:
            eval_perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
            eval_results['eval_perplexity'] = eval_perplexity
            print(f"Validation MLM Loss: {eval_results['eval_loss']:.4f}")
            print(f"Validation MLM Perplexity: {eval_perplexity:.2f}")
        
        # 在测试集上评估
        print("\n" + "="*60)
        print("Evaluating on test set...")
        test_results = self.trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        self.trainer.log_metrics("test", test_results)
        self.trainer.save_metrics("test", test_results)
        
        # 计算测试集MLM困惑度
        if 'test_loss' in test_results:
            test_perplexity = torch.exp(torch.tensor(test_results['test_loss'])).item()
            test_results['test_perplexity'] = test_perplexity
            print(f"Test MLM Loss: {test_results['test_loss']:.4f}")
            print(f"Test MLM Perplexity: {test_perplexity:.2f}")
        
        # 如果启用分类，计算分类指标
        if self.config.enable_classification:
            print("\n" + "="*60)
            print("Computing classification metrics...")
            classification_metrics = self.metrics_calculator.compute_classification_metrics(
                self.model, tokenized_datasets, self.trainer
            )
            eval_results.update(classification_metrics['validation'])
            test_results.update(classification_metrics['test'])
        
        
        
        all_results = {
            "training_total_params": total_params,
            "core_model_params": core_model_params,
            "mlm_head_params": total_params - core_model_params,
            "train": train_result.metrics,
            "validation": eval_results,
            "test": test_results,
            "num_classes": self.config.num_classes if self.config.enable_classification else None,
            "enable_classification": self.config.enable_classification
        }
        
        return self.trainer, all_results
    
    def save_model_info(self):
        """保存模型信息"""
        model_info = {
            "model_name": self.config.model_name,
            "vocab_size": len(self.tokenizer_manager.token_dictionary),
            "max_length": self.config.max_length,
            "training_params": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "warmup_steps": self.config.warmup_steps
            },
            "dataset_info": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio
            },
            "classification_info": {
                "enabled": self.config.enable_classification,
                "num_classes": self.config.num_classes if self.config.enable_classification else None,
                "classification_weight": self.config.classification_weight
            }
        }
        
        info_file = self.output_dir / "model_info.json"
        with open(info_file, "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to: {info_file}")


def train_gene_bert_model(
    dataset_path: str,
    token_dictionary_file: str,
    output_dir: str = "gene_bert_output",
    **kwargs
) -> Tuple[Trainer, Dict[str, Dict[str, float]]]:
    """
    训练GeneBERT模型的便捷函数
    
    Args:
        dataset_path: 数据集路径
        token_dictionary_file: token字典文件路径
        output_dir: 输出目录
        **kwargs: 其他训练参数
    
    Returns:
        Tuple[Trainer, Dict]: 训练器实例和结果字典
    """
    # 创建配置
    config = TrainingConfig(
        dataset_path=dataset_path,
        token_dictionary_file=token_dictionary_file,
        output_dir=output_dir,
        **kwargs
    )
    
    # 创建训练器
    trainer = GeneBERTTrainer(config)
    
    # 训练模型
    trainer_instance, all_results = trainer.train()
    
    # 保存模型信息
    trainer.save_model_info()
    
    return trainer_instance, all_results
