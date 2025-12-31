#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from inspect import signature

import torch
from datasets import DatasetDict, load_from_disk
from transformers import BertConfig, AutoTokenizer, Trainer, TrainingArguments

from multi_value_regression_model import (
    MultiValueRegressionCfg, BertForMultiValueRegression, ensure_regression_dense,
    regression_metrics, normalize_targets
)


def _load_tokenizer_and_pad_id(pretrained_dir: str):
    """Load tokenizer and extract pad_token_id with fallback"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, use_fast=False)
        pad_id = getattr(tokenizer, "pad_token_id", None)
    except Exception as e:
        print(f"[warn] AutoTokenizer failed: {e}")
        tokenizer = type('_PadStub', (), {'pad_token_id': 0})()
        pad_id = None
    
    if pad_id is None:
        try:
            cfg = BertConfig.from_pretrained(pretrained_dir)
            pad_id = getattr(cfg, "pad_token_id", 0) or 0
        except Exception:
            pad_id = 0
    
    return tokenizer, pad_id


def _create_training_args(cfg: MultiValueRegressionCfg):
    """Create TrainingArguments with version compatibility"""
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    args_kwargs = {
        "output_dir": cfg.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "logging_steps": cfg.logging_steps,
        "eval_steps": cfg.eval_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": cfg.load_best_model_at_end,
        "metric_for_best_model": cfg.metric_for_best_model,
        "greater_is_better": cfg.greater_is_better,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "report_to": list(cfg.report_to),
        "fp16": cfg.fp16 and use_cuda,
        "dataloader_pin_memory": use_cuda,
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
    }
    
    if cfg.deepspeed:
        args_kwargs.update({"deepspeed": cfg.deepspeed, "local_rank": cfg.local_rank})
    
    # Version compatibility
    sig = signature(TrainingArguments.__init__)
    allowed = {k: v for k, v in args_kwargs.items() if k in sig.parameters}
    
    if "evaluation_strategy" in sig.parameters and "eval_strategy" in args_kwargs:
        allowed["evaluation_strategy"] = args_kwargs["eval_strategy"]
    elif "eval_strategy" not in sig.parameters:
        allowed.pop("eval_strategy", None)
        if "do_eval" in sig.parameters:
            allowed["do_eval"] = True
    
    return TrainingArguments(**allowed)


def _print_metrics(title: str, metrics: dict):
    """Print metrics in a formatted way"""
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        fmt = f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
        print(fmt)


def _save_config(cfg: MultiValueRegressionCfg, target_stats: dict, output_dir: str):
    """Save model configuration and target statistics"""
    config_data = {
        "num_targets": cfg.num_targets,
        "head": cfg.head,
        "hidden_size": cfg.hidden_size,
        "lstm_layers": cfg.lstm_layers,
        "cnn_kernel": cfg.cnn_kernel,
        "dropout": cfg.dropout,
        "loss_type": cfg.loss_type,
        "huber_delta": cfg.huber_delta,
        "pretrained_dir": cfg.pretrained_dir,
        "target_stats": target_stats,
        "max_length": cfg.max_length,
    }
    with open(os.path.join(output_dir, "regression_config.json"), "w") as f:
        json.dump(config_data, f, indent=2)


class RegressionTrainer(Trainer):
    """Custom Trainer for regression tasks"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs.pop("targets")
        outputs = model(**inputs, targets=targets)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        targets = inputs.get("targets")
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = outputs.get("predictions")
        loss = outputs.get("loss")
        return (loss, predictions, targets) if (predictions is not None and targets is not None) else (loss, None, None)


def train_multivalue_regression(cfg: MultiValueRegressionCfg):
    """Train multi-value regression model"""
    # Load tokenizer and pad_id
    tokenizer, pad_id = _load_tokenizer_and_pad_id(cfg.pretrained_dir)
    
    # Load and preprocess datasets
    ds = DatasetDict({
        "train": ensure_regression_dense(load_from_disk(cfg.train_dataset), cfg.max_length, pad_id),
        "validation": ensure_regression_dense(load_from_disk(cfg.val_dataset), cfg.max_length, pad_id),
        "test": ensure_regression_dense(load_from_disk(cfg.test_dataset), cfg.max_length, pad_id),
    })
    
    # Normalize targets
    print("[info] Normalizing targets...")
    ds["train"], target_stats = normalize_targets(ds["train"])
    ds["validation"], _ = normalize_targets(ds["validation"], stats=target_stats)
    ds["test"], _ = normalize_targets(ds["test"], stats=target_stats)
    
    # Initialize model
    model = BertForMultiValueRegression(
        BertConfig.from_pretrained(cfg.pretrained_dir), cfg.num_targets, cfg.pretrained_dir,
        head_type=cfg.head, hidden_size=cfg.hidden_size, lstm_layers=cfg.lstm_layers,
        cnn_kernel=cfg.cnn_kernel, dropout=cfg.dropout,
        loss_type=cfg.loss_type, huber_delta=cfg.huber_delta
    )
    
    # Initialize trainer
    trainer = RegressionTrainer(
        model=model,
        args=_create_training_args(cfg),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        compute_metrics=regression_metrics,
    )
    
    # Train
    print(f"[info] Starting multi-value regression training with {cfg.num_targets} targets")
    train_out = trainer.train()
    
    # Save model
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(cfg.output_dir)
    
    # Evaluate
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=ds["test"])
    
    _print_metrics("Validation metrics", eval_metrics)
    _print_metrics("Test metrics", test_metrics)
    
    # Save configuration
    _save_config(cfg, target_stats, cfg.output_dir)
    
    return {"train": train_out.metrics, "validation": eval_metrics, "test": test_metrics}


def main():
    ap = argparse.ArgumentParser(description="Multi-value regression training")
    
    # Data
    ap.add_argument("--train_dataset", required=True, help="path to train dataset")
    ap.add_argument("--val_dataset", required=True, help="path to validation dataset")
    ap.add_argument("--test_dataset", required=True, help="path to test dataset")
    ap.add_argument("--pretrained_dir", required=True, help="pretrained BERT directory")
    ap.add_argument("--output_dir", default="multivalue_regression_run", help="output directory")
    
    # Model
    ap.add_argument("--num_targets", type=int, default=8, help="number of regression targets")
    ap.add_argument("--max_length", type=int, default=1024, help="maximum sequence length")
    ap.add_argument("--head", type=str, default="mlp", choices=["mlp", "bilstm", "cnn"])
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--cnn_kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    
    # Training
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--deepspeed", type=str, default=None)
    
    args = ap.parse_args()
    
    cfg = MultiValueRegressionCfg(
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        test_dataset=args.test_dataset,
        pretrained_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_targets=args.num_targets,
        head=args.head,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        cnn_kernel=args.cnn_kernel,
        dropout=args.dropout,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        eval_steps=args.eval_steps,
    )
    
    train_multivalue_regression(cfg)


if __name__ == "__main__":
    main()
