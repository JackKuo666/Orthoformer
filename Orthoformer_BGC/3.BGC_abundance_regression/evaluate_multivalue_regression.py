#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, BertConfig

from multi_value_regression_model import (
    BertForMultiValueRegression, regression_metrics,
    denormalize_predictions
)


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        "input_ids": [item["input_ids"] for item in batch],
        "attention_mask": [item["attention_mask"] for item in batch],
        "targets": [item["targets"] for item in batch],
        "sample_name": [item["sample_name"] for item in batch]
    }


def ensure_regression_dense(
        ds: Dataset, max_length: int, pad_token_id: int = 0
) -> Dataset:
    """
    Preprocess dataset for multi-value regression.

    Expected columns:
      - input_ids: List[int]
      - targets: List[float] (regression targets)
    """
    cols = ds.column_names
    if "input_ids" not in cols:
        raise ValueError("Dataset must have 'input_ids'")
    if "targets" not in cols:
        raise ValueError("Dataset must have 'targets' for regression")

    def _proc(ex: Dict[str, Any]) -> Dict[str, Any]:
        sample_name = ex.get("sample_name", "")
        ids = ex["input_ids"]
        targets = ex["targets"]

        # Remove CLS/SEP tokens if present
        if len(ids) > 2 and ids[0] == 101 and ids[-1] == 102:  # BERT CLS/SEP tokens
            ids = ids[1:-1]

        # Truncate to max_length
        L = min(len(ids), max_length)
        ids = ids[:L]
        attn = [1] * L

        # Pad sequences
        if L < max_length:
            pad_len = max_length - L
            ids = ids + [pad_token_id] * pad_len
            attn = attn + [0] * pad_len

        # Ensure targets is a list of floats
        if isinstance(targets, (int, float)):
            targets = [float(targets)]
        elif isinstance(targets, list):
            targets = [float(t) for t in targets]
        else:
            raise ValueError(f"Invalid targets format: {type(targets)}")

        return {
            "sample_name": ex["sample_name"],
            "input_ids": ids,
            "attention_mask": attn,
            "targets": targets
        }

    keep = {"sample_name", "input_ids", "attention_mask", "targets"}
    return ds.map(_proc, remove_columns=[c for c in cols if c not in keep])


def load_model_and_tokenizer(model_dir: str, pretrained_dir: str, config_path: str = None, device: torch.device = None):
    """Load trained model and tokenizer"""
    if config_path is None:
        config_path = os.path.join(model_dir, "regression_config.json")

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load tokenizer from pretrained directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, use_fast=False)
    except Exception as e:
        print(f"[warn] Failed to load tokenizer: {e}")
        tokenizer = None

    # Initialize model with pretrained BERT architecture
    base_cfg = BertConfig.from_pretrained(pretrained_dir)
    model = BertForMultiValueRegression(
        base_cfg,
        config["num_targets"],
        pretrained_dir,
        head_type=config["head"],
        hidden_size=config["hidden_size"],
        lstm_layers=config["lstm_layers"],
        cnn_kernel=config["cnn_kernel"],
        dropout=config["dropout"],
        loss_type=config["loss_type"],
        huber_delta=config.get("huber_delta", 1.0)
    )

    # Load model weights
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(model_path):
        # Load state dict from pytorch_model.bin
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Try loading from model.safetensors
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
            except ImportError:
                # If safetensors is not available, try regular torch.load
                state_dict = torch.load(safetensors_path, map_location=device)
                model.load_state_dict(state_dict)
        else:
            # If neither exists, check what files are available
            available_files = os.listdir(model_dir)
            print(f"Available files in {model_dir}: {available_files}")
            raise FileNotFoundError(
                f"No model weights found in {model_dir}. Expected pytorch_model.bin or model.safetensors")

    model.eval()

    return model, tokenizer, config


def evaluate_model(model, tokenizer, dataset, config, denormalize: bool = True, 
                   device: torch.device = None, batch_size: int = 32, 
                   use_distributed: bool = False):
    """Evaluate model on dataset"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # Wrap model with DDP if using distributed evaluation
    if use_distributed and dist.is_initialized():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    # Preprocess dataset
    pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer else 0
    processed_dataset = ensure_regression_dense(dataset, config["max_length"], pad_id)

    # Create DataLoader with DistributedSampler if using distributed evaluation
    if use_distributed and dist.is_initialized():
        sampler = DistributedSampler(processed_dataset, shuffle=False)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )

    # Collect predictions and targets
    predictions = []
    targets = []
    sample_names = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Prepare inputs
            input_ids = torch.stack([torch.tensor(ids) for ids in batch["input_ids"]]).to(device)
            attention_mask = torch.stack([torch.tensor(mask) for mask in batch["attention_mask"]]).to(device)
            batch_targets = batch["targets"]
            batch_sample_names = batch["sample_name"]

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                batch_predictions = outputs["predictions"].cpu().numpy()
            else:
                batch_predictions = outputs.cpu().numpy()

            # Denormalize predictions if needed
            if denormalize and "target_stats" in config:
                batch_predictions = denormalize_predictions(
                    batch_predictions, config["target_stats"]
                )

            predictions.append(batch_predictions)
            targets.extend(batch_targets)
            sample_names.extend(batch_sample_names)

    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    targets = np.array(targets)

    # Gather results from all processes if using distributed evaluation
    if use_distributed and dist.is_initialized():
        # Gather predictions, targets, and sample_names from all processes
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Gather predictions
        gathered_predictions = [None] * world_size
        dist.all_gather_object(gathered_predictions, predictions)
        predictions = np.concatenate(gathered_predictions, axis=0)
        
        # Gather targets
        gathered_targets = [None] * world_size
        dist.all_gather_object(gathered_targets, targets)
        targets = np.concatenate(gathered_targets, axis=0)
        
        # Gather sample names
        gathered_sample_names = [None] * world_size
        dist.all_gather_object(gathered_sample_names, sample_names)
        sample_names = [name for sublist in gathered_sample_names for name in sublist]
        
        # Only rank 0 should compute metrics and return results
        if rank != 0:
            return None, None, None, None

    # Compute metrics
    metrics = regression_metrics((predictions, targets))

    return metrics, predictions, targets, sample_names


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Print metrics in a formatted way"""
    print(f"\n=== {title} ===")

    # Overall metrics
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    # Per-target metrics
    num_targets = len([k for k in metrics.keys() if k.startswith("target_") and k.endswith("_mse")])
    print("\nPer-target metrics:")
    for i in range(num_targets):
        print(f"Target {i}:")
        print(f"  MSE: {metrics[f'target_{i}_mse']:.4f}")
        print(f"  MAE: {metrics[f'target_{i}_mae']:.4f}")
        print(f"  R²: {metrics[f'target_{i}_r2']:.4f}")


def analyze_predictions(predictions: np.ndarray, targets: np.ndarray, num_targets: int):
    """Analyze prediction quality"""
    print("\n=== Prediction Analysis ===")

    for i in range(num_targets):
        pred_i = predictions[:, i]
        target_i = targets[:, i]

        # Correlation
        correlation = np.corrcoef(pred_i, target_i)[0, 1]

        # Error statistics
        errors = pred_i - target_i
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        print(f"Target {i}:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Mean Error: {mean_error:.4f}")
        print(f"  Std Error: {std_error:.4f}")
        print(f"  Min Error: {np.min(errors):.4f}")
        print(f"  Max Error: {np.max(errors):.4f}")


def save_predictions(predictions: np.ndarray, targets: np.ndarray, output_path: str):
    """Save predictions and targets to file"""
    results = {
        "predictions": predictions.tolist(),
        "targets": targets.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Predictions saved to {output_path}")


def save_predictions_parquet(sample_names, predictions: np.ndarray, targets: np.ndarray, output_path: str,
                             num_targets: int):
    """Save predictions and targets to parquet file"""
    # Create DataFrame with predictions and targets
    data = {}

    data["sample_name"] = sample_names
    target_columns= ["mixed", "nrps", "nrps related peptides", "other", "pks",
                     "ripps", "saccharides & derivatives", "terpenes"]
    # Add prediction columns
    for i, target_name in enumerate(target_columns):
        pred_i = np.where(predictions[:, i] < 0, 0, predictions[:, i])
        data[f'pred_{target_name}'] = pred_i

    # Add target columns
    for i, target_name in enumerate(target_columns):
        data[f'true_{target_name}'] = targets[:, i]

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to parquet
    df.to_parquet(output_path, index=False)

    print(f"Predictions saved to parquet file: {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-value regression model")

    # Required arguments
    parser.add_argument("--model_dir", required=True, help="path to trained model directory")
    parser.add_argument("--pretrained_dir", required=True, help="path to pretrained BERT directory")
    parser.add_argument("--test_dataset", required=True, help="path to test dataset")

    # Optional arguments
    parser.add_argument("--output_dir", default="evaluation_results", help="output directory for results")
    parser.add_argument("--denormalize", action="store_true", default=True, help="denormalize predictions and targets")
    parser.add_argument("--save_predictions", action="store_true", help="save predictions to JSON file")
    parser.add_argument("--save_parquet", action="store_true", help="save predictions to parquet file")
    parser.add_argument("--config_path", help="path to model config file (default: model_dir/regression_config.json)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for evaluation")
    
    args = parser.parse_args()

    # Initialize distributed training if torchrun is used
    use_distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        use_distributed = True
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        if rank == 0:
            print(f"Using distributed evaluation with {world_size} GPUs")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using single GPU/CPU evaluation on {device}")

    # Create output directory (only rank 0 needs to do this)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    if rank == 0:
        print("Loading model and tokenizer...")
    model, tokenizer, config = load_model_and_tokenizer(
        args.model_dir, args.pretrained_dir, args.config_path, device=device
    )

    # Load test dataset
    if rank == 0:
        print("Loading test dataset...")
    test_dataset = load_from_disk(args.test_dataset)

    # Evaluate model
    if rank == 0:
        print("Evaluating model...")
    metrics, predictions, targets, sample_names = evaluate_model(
        model, tokenizer, test_dataset, config, 
        denormalize=args.denormalize,
        device=device,
        batch_size=args.batch_size,
        use_distributed=use_distributed
    )

    # Only rank 0 should print and save results
    if rank == 0 and metrics is not None:
        # Print metrics
        print_metrics(metrics, "Test Set Metrics")

        # Analyze predictions
        analyze_predictions(predictions, targets, config["num_targets"])

        # Save results
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nResults saved to {results_path}")

        # Save predictions if requested
        if args.save_predictions:
            predictions_path = os.path.join(args.output_dir, "predictions.json")
            save_predictions(predictions, targets, predictions_path)

        # Save predictions as parquet if requested
        if args.save_parquet:
            parquet_path = os.path.join(args.output_dir, "predictions.parquet")
            save_predictions_parquet(sample_names, predictions, targets, parquet_path, config["num_targets"])
    
    # Cleanup distributed training
    if use_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
