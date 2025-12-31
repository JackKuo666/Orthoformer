# Multi-Value Regression Model

This directory contains scripts for training and evaluating a multi-value regression model based on BERT with custom attention mechanisms (ALiBi). The model predicts multiple continuous values from sequence data.

## Overview

The model consists of:
- **BERT Encoder**: Pretrained BERT model with OrthoformerSelfAttention (ALiBi positional encoding)
- **Regression Head**: One of three architectures (MLP, BiLSTM, or CNN) for final predictions
- **Multi-Target Support**: Predicts multiple continuous values simultaneously

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- NumPy
- scikit-learn
- pandas (for parquet output)

## Dataset Format

The dataset should be saved using HuggingFace `datasets` and must contain the following columns:

- `input_ids`: List[int] - Tokenized input sequence
- `targets`: List[float] - Regression targets (can be a single float or list of floats)
- `sample_name` (optional): str - Sample identifier

Example dataset structure:
```python
{
    "input_ids": [101, 2023, 2003, ..., 102],  # BERT token IDs
    "targets": [0.5, 1.2, 0.8, 0.3, 0.1, 0.0, 0.4, 0.2],  # 8 target values
    "sample_name": "sample_001"
}
```

## Training

### Basic Usage

```bash
python train_multivalue_regression.py \
    --train_dataset /path/to/train/dataset \
    --val_dataset /path/to/val/dataset \
    --test_dataset /path/to/test/dataset \
    --pretrained_dir /path/to/pretrained/bert \
    --output_dir ./output_model \
    --num_targets 8 \
    --max_length 2048
```

### Full Training Example

```bash
python train_multivalue_regression.py \
    --train_dataset ./datasets/train \
    --val_dataset ./datasets/val \
    --test_dataset ./datasets/test \
    --pretrained_dir ./pretrained_model \
    --output_dir ./models/multivalue_regression \
    --num_targets 8 \
    --max_length 2048 \
    --head mlp \
    --hidden_size 512 \
    --dropout 0.1 \
    --loss_type mse \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --lr 3e-5 \
    --epochs 30 \
    --eval_steps 500 \
    --fp16
```

### Training Arguments

#### Data Arguments
- `--train_dataset`: Path to training dataset (required)
- `--val_dataset`: Path to validation dataset (required)
- `--test_dataset`: Path to test dataset (required)
- `--pretrained_dir`: Path to pretrained BERT model directory (required)
- `--output_dir`: Output directory for trained model (default: `multivalue_regression_run`)

#### Model Configuration
- `--num_targets`: Number of regression targets (default: 8)
- `--max_length`: Maximum sequence length (default: 1024)
- `--head`: Regression head type - `mlp`, `bilstm`, or `cnn` (default: `mlp`)
- `--hidden_size`: Hidden size for regression head (default: 512)
- `--lstm_layers`: Number of LSTM layers (for BiLSTM head, default: 2)
- `--cnn_kernel`: CNN kernel size (for CNN head, default: 3)
- `--dropout`: Dropout rate (default: 0.1)

#### Loss Configuration
- `--loss_type`: Loss function - `mse`, `mae`, or `huber` (default: `mse`)
- `--huber_delta`: Delta parameter for Huber loss (default: 1.0)

#### Training Configuration
- `--train_batch_size`: Training batch size (default: 32)
- `--eval_batch_size`: Evaluation batch size (default: 32)
- `--lr`: Learning rate (default: 3e-5)
- `--epochs`: Number of training epochs (default: 5)
- `--eval_steps`: Evaluation frequency in steps (default: 500)
- `--fp16`: Enable mixed precision training (flag)
- `--deepspeed`: Path to DeepSpeed config file (optional)

### Model Architecture Options

#### MLP Head (default)
Simple multi-layer perceptron with global average pooling:
```bash
--head mlp --hidden_size 512
```

#### BiLSTM Head
Bidirectional LSTM with global average pooling:
```bash
--head bilstm --hidden_size 512 --lstm_layers 2
```

#### CNN Head
1D Convolutional network with global average pooling:
```bash
--head cnn --hidden_size 512 --cnn_kernel 3
```

### Output

After training, the model directory will contain:
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `regression_config.json`: Model configuration and target statistics
- `tokenizer files`: Tokenizer configuration
- Training logs and checkpoints

## Evaluation

### Basic Usage

```bash
python evaluate_multivalue_regression.py \
    --model_dir ./models/multivalue_regression \
    --pretrained_dir ./pretrained_bert \
    --test_dataset ./datasets/test \
    --output_dir ./evaluation_results
```

### Full Evaluation Example

```bash
python evaluate_multivalue_regression.py \
    --model_dir ./models/multivalue_regression \
    --pretrained_dir ./pretrained_bert \
    --test_dataset ./datasets/test \
    --output_dir ./evaluation_results \
    --batch_size 32 \
    --denormalize \
    --save_predictions \
    --save_parquet
```

### Evaluation Arguments

- `--model_dir`: Path to trained model directory (required)
- `--pretrained_dir`: Path to pretrained BERT directory (required)
- `--test_dataset`: Path to test dataset (required)
- `--output_dir`: Output directory for results (default: `evaluation_results`)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--denormalize`: Denormalize predictions using training statistics (default: True)
- `--save_predictions`: Save predictions to JSON file (flag)
- `--save_parquet`: Save predictions to parquet file (flag)
- `--config_path`: Path to model config file (optional, defaults to `model_dir/regression_config.json`)

### Distributed Evaluation

For multi-GPU evaluation:
```bash
torchrun --nproc_per_node=4 evaluate_multivalue_regression.py \
    --model_dir ./models/multivalue_regression \
    --pretrained_dir ./pretrained_bert \
    --test_dataset ./datasets/test \
    --output_dir ./evaluation_results
```

### Evaluation Output

The evaluation script generates:
- **Metrics**: Overall and per-target MSE, MAE, and R² scores
- **Prediction Analysis**: Correlation, error statistics for each target
- `evaluation_results.json`: Detailed metrics
- `predictions.json`: Predictions and targets (if `--save_predictions` is used)
- `predictions.parquet`: Predictions in parquet format (if `--save_parquet` is used)

## Model Configuration File

The `regression_config.json` file contains:
```json
{
  "num_targets": 8,
  "head": "mlp",
  "hidden_size": 512,
  "lstm_layers": 2,
  "cnn_kernel": 3,
  "dropout": 0.1,
  "loss_type": "mse",
  "huber_delta": 1.0,
  "pretrained_dir": "./pretrained_bert",
  "target_stats": {
    "mean": [0.5, 1.2, ...],
    "std": [0.3, 0.8, ...]
  },
  "max_length": 1024
}
```

The `target_stats` are used for normalization during training and denormalization during evaluation.

## Tips

1. **Normalization**: Targets are automatically normalized during training. Use `--denormalize` during evaluation to get original scale predictions.

2. **Sequence Length**: Adjust `--max_length` based on your data. Longer sequences require more memory.

3. **Head Selection**: 
   - Use `mlp` for simple, fast training
   - Use `bilstm` for sequence-aware modeling
   - Use `cnn` for local pattern detection

4. **Loss Function**:
   - `mse`: Standard mean squared error (default)
   - `mae`: Mean absolute error (more robust to outliers)
   - `huber`: Huber loss (combines MSE and MAE benefits)

5. **Batch Size**: Adjust based on GPU memory. Use gradient accumulation for effective larger batch sizes.

6. **Mixed Precision**: Use `--fp16` to speed up training and reduce memory usage on compatible GPUs.

## Example Workflow

1. **Prepare datasets**:
   ```bash
   # Your dataset preparation script
   python prepare_datasets.py
   ```

2. **Train model**:
   ```bash
   python train_multivalue_regression.py \
       --train_dataset ./data/train \
       --val_dataset ./data/val \
       --test_dataset ./data/test \
       --pretrained_dir ./bert_model \
       --output_dir ./trained_model \
       --num_targets 8 \
       --head mlp \
       --epochs 5
   ```

3. **Evaluate model**:
   ```bash
   python evaluate_multivalue_regression.py \
       --model_dir ./trained_model \
       --pretrained_dir ./bert_model \
       --test_dataset ./data/test \
       --output_dir ./results \
       --save_predictions \
       --save_parquet
   ```

## Troubleshooting

- **Out of Memory**: Reduce `--max_length` or `--train_batch_size`
- **Tokenizer Errors**: Ensure `--pretrained_dir` contains a valid tokenizer
- **Shape Mismatch**: Verify `--num_targets` matches your dataset's target count
- **Missing Config**: The model directory must contain `regression_config.json`

