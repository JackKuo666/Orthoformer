# Orthoformer CRISPR Fine-tuning

## Introduction

This directory contains scripts for fine-tuning Orthoformer models on CRISPR-related token-level multiclass classification tasks. The pipeline includes:

1. **Fine-tuning**: Fine-tune pretrained Orthoformer models for token classification
2. **Class Imbalance Handling**: Address class imbalance using Focal Loss
3. **Evaluation**: Comprehensive evaluation metrics and analysis
4. **Model Conversion**: Convert checkpoints to safetensors format

The task involves token-level classification with 8 classes, where each token in a protein sequence is classified into one of the categories.

## Directory Structure

```
Orthoformer_CRISPR/
├── finetune_token_multiclass.py      # Main fine-tuning script
├── run_train.3M_2048_v10.sh          # Training script
├── convert_checkpoint_to_safetensors.py  # Checkpoint conversion utility
├── extract_eval_metrics.py           # Extract metrics from training logs
├── view_metrics_table.py             # View metrics in formatted table
├── analyze_class_imbalance.py        # Analyze class imbalance issues
├── orthoformer_model.py              # Pretraining utilities
├── CRISPR_datasets/                   # Training datasets
│   ├── train.dataset                 # Training set
│   ├── val.dataset                    # Validation set
│   └── test.dataset                   # Test set
└── output_3M_2048_v10/                # Training outputs
    ├── checkpoints/                   # Model checkpoints
    ├── logs/                          # Training logs
    └── metrics/                       # Evaluation metrics
```

## Requirements

- Python 3.8+
- PyTorch >= 1.12
- Required Python packages:
  - `transformers` (HuggingFace)
  - `datasets` (HuggingFace)
  - `numpy`, `pandas`
  - `scikit-learn` (for metrics)
  - `matplotlib` (for visualization)
  - `safetensors` (optional, for checkpoint conversion)
  - `captum` (optional, for attribution analysis)
- GPU recommended (multi-GPU supported via torchrun)
- Pretrained Orthoformer model: `model_3M_2048_v10`

## Dataset Download

The required datasets for this project can be downloaded from the Hugging Face repository:

### CRISPR Dataset

**Download URL:** https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Downstream_Tasks_dataset/CRISPR_datasets

**Dataset Contents:**

- **CRISPR_datasets.tar.gz**: Compressed file containing CRISPR training, validation, and test datasets

**Note**: 
- Extract the `CRISPR_datasets.tar.gz` file after downloading using: `tar -xzf CRISPR_datasets.tar.gz`
- Place the extracted dataset files in the `CRISPR_datasets/` directory as specified in the workflow section.

## Workflow

### 1. Prepare Datasets

Ensure your datasets are in HuggingFace format and located in `CRISPR_datasets/`:
- `train.dataset/`: Training dataset
- `val.dataset/`: Validation dataset
- `test.dataset/`: Test dataset

Each dataset should contain:
- `input_ids`: Tokenized protein sequences
- `attention_mask`: Attention masks
- `labels`: Token-level class labels (0-7 for 8 classes)

### 2. Fine-tune Model

Run the fine-tuning script:

```bash
bash run_train.3M_2048_v10.sh
```

Or run directly with torchrun (multi-GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_token_multiclass.py token_multiclass \
    --train_dataset CRISPR_datasets/train.dataset \
    --val_dataset CRISPR_datasets/val.dataset \
    --test_dataset CRISPR_datasets/test.dataset \
    --pretrained_dir ../../foundation_model/model/model_3M_2048_v10/final_model \
    --output_dir output_3M_2048_v10 \
    --num_labels 8 \
    --max_length 2048 \
    --compute_class_counts \
    --epochs 30 \
    --batch_size 2
```

**Key Parameters:**
- `--pretrained_dir`: Path to pretrained Orthoformer model
- `--num_labels`: Number of classes (default: 8)
- `--max_length`: Maximum sequence length (default: 2048)
- `--head`: Classification head type: `mlp`, `bilstm`, or `cnn` (default: `bilstm`)
- `--loss_type`: Loss function: `cross_entropy` or `focal` (default: `focal`)
- `--compute_class_counts`: Automatically compute class distribution
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size per device (default: 2)

**Classification Heads:**
- **MLP**: Simple multi-layer perceptron
- **BiLSTM**: Bidirectional LSTM for sequence modeling
- **CNN**: Convolutional neural network with 1D convolutions

**Loss Functions:**
- **Cross Entropy**: Standard cross-entropy loss
- **Focal Loss**: Addresses class imbalance by down-weighting easy examples
  - Formula: `FL = -α_y * (1 - p_y)^γ * log(p_y)`
  - `gamma`: Focusing parameter (default: 2.0)
  - `alpha`: Class weights (optional)

### 3. Monitor Training

Training logs are saved to:
- Console output (redirected to `run_train.3M_2048_v10.sh.log`)
- TensorBoard logs in `output_3M_2048_v10/logs/`

View TensorBoard:
```bash
tensorboard --logdir output_3M_2048_v10/logs
```

### 4. Extract Evaluation Metrics

Extract metrics from training log:

```bash
python extract_eval_metrics.py run_train.3M_2048_v10.sh.log
```

This generates a CSV file with all evaluation metrics per epoch.

### 5. View Metrics Table

View metrics in a formatted table:

```bash
python view_metrics_table.py metrics.csv
```

**Metrics Included:**
- `eval_loss`: Validation loss
- `eval_accuracy`: Overall accuracy
- `eval_f1_micro`: Micro-averaged F1 score
- `eval_f1_macro`: Macro-averaged F1 score
- `eval_f1_c{i}`: Per-class F1 scores (i = 0-7)
- `eval_support_c{i}`: Per-class sample counts

### 6. Analyze Class Imbalance

Analyze class distribution and model performance:

```bash
python analyze_class_imbalance.py metrics.csv
```

This script provides:
- Class distribution analysis
- Early vs. late training performance
- Best F1 scores per class
- Class imbalance identification

### 7. Convert Checkpoints (Optional)

Convert PyTorch checkpoints to safetensors format:

```bash
python convert_checkpoint_to_safetensors.py output_3M_2048_v10/checkpoint-XXXXX
```

This is useful for:
- Resuming training without PyTorch >= 2.6 requirement
- Reducing checkpoint file size
- Improving loading speed

## Configuration

### Model Architecture

The fine-tuning uses:
- **Base Model**: Orthoformer (BERT-based) with ALiBi positional encoding
- **Classification Head**: BiLSTM (default), MLP, or CNN
- **Hidden Size**: 512 (configurable)
- **Dropout**: 0.1 (configurable)

### Training Configuration

Default training parameters:
- **Learning Rate**: 3e-5
- **Batch Size**: 2 per device (adjust based on GPU memory)
- **Max Length**: 2048 tokens
- **Epochs**: 30
- **Warmup Steps**: 0
- **Weight Decay**: 0.0
- **Evaluation Strategy**: Steps (every 5000 steps)
- **Save Strategy**: Steps (every 5000 steps)
- **Best Model Metric**: `eval_f1_macro` (macro-averaged F1)

### Handling Class Imbalance

The pipeline includes several strategies for class imbalance:

1. **Focal Loss**: Automatically down-weights easy examples
   ```python
   --loss_type focal --focal_gamma 2.0
   ```

2. **Class Weights**: Provide class-specific weights
   ```python
   --focal_alpha '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'
   ```

3. **Automatic Class Counting**: Compute class distribution
   ```python
   --compute_class_counts
   ```

## Quick Start

1. **Prepare datasets**:
   - Ensure datasets are in `CRISPR_datasets/` directory
   - Verify datasets contain required fields (`input_ids`, `attention_mask`, `labels`)

2. **Start training**:
   ```bash
   bash run_train.3M_2048_v10.sh
   ```

3. **Monitor progress**:
   ```bash
   tail -f run_train.3M_2048_v10.sh.log
   # or
   tensorboard --logdir output_3M_2048_v10/logs
   ```

4. **Extract and view metrics**:
   ```bash
   python extract_eval_metrics.py run_train.3M_2048_v10.sh.log
   python view_metrics_table.py metrics.csv
   ```

5. **Analyze results**:
   ```bash
   python analyze_class_imbalance.py metrics.csv
   ```

## Advanced Usage

### Custom Classification Head

Choose different classification heads:

```bash
# MLP head
--head mlp --hidden_size 512

# BiLSTM head (default)
--head bilstm --hidden_size 512 --lstm_layers 2

# CNN head
--head cnn --hidden_size 512 --cnn_kernel 3
```

### Custom Loss Configuration

Fine-tune focal loss parameters:

```bash
# Adjust gamma (focusing parameter)
--focal_gamma 3.0

# Provide class-specific alpha weights
--focal_alpha '[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]'
```

### Multi-GPU Training

Use multiple GPUs for faster training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    finetune_token_multiclass.py token_multiclass \
    --train_dataset CRISPR_datasets/train.dataset \
    --val_dataset CRISPR_datasets/val.dataset \
    --test_dataset CRISPR_datasets/test.dataset \
    --pretrained_dir ../../foundation_model/model/model_3M_2048_v10/final_model \
    --output_dir output_3M_2048_v10 \
    --num_labels 8 \
    --batch_size 4  # Per device
```

### Resume Training

Resume from a checkpoint:

```bash
--resume_from_checkpoint output_3M_2048_v10/checkpoint-15000
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
- Reduce `--batch_size` (e.g., from 2 to 1)
- Reduce `--max_length` (e.g., from 2048 to 1024)
- Use gradient accumulation (add to training script)
- Use mixed precision training: `--fp16`

### Class Imbalance Issues

If certain classes have poor performance:
- Use Focal Loss: `--loss_type focal`
- Adjust focal gamma: `--focal_gamma 3.0` (higher = more focus on hard examples)
- Provide class weights: `--focal_alpha '[...]'`
- Check class distribution: `python analyze_class_imbalance.py metrics.csv`

### Poor Convergence

If model doesn't converge:
- Reduce learning rate: `--learning_rate 1e-5`
- Increase warmup steps: `--warmup_steps 1000`
- Check data quality and label alignment
- Verify pretrained model is loaded correctly

### Checkpoint Loading Issues

If checkpoint loading fails:
- Convert to safetensors: `python convert_checkpoint_to_safetensors.py <checkpoint_dir>`
- Check PyTorch version compatibility
- Verify checkpoint directory structure

## Output Files

### Training Outputs

- `output_3M_2048_v10/checkpoint-XXXXX/`: Model checkpoints
  - `pytorch_model.bin` or `model.safetensors`: Model weights
  - `config.json`: Model configuration
  - `training_args.bin`: Training arguments
- `output_3M_2048_v10/logs/`: TensorBoard logs
- `output_3M_2048_v10/trainer_state.json`: Training state

### Evaluation Metrics

- `metrics.csv`: Extracted evaluation metrics (from `extract_eval_metrics.py`)
- `rf_metrics.txt`: RF metrics summary (if applicable)

## Evaluation Metrics

### Overall Metrics

- **Accuracy**: Overall token classification accuracy
- **F1 Micro**: Micro-averaged F1 score (treats all tokens equally)
- **F1 Macro**: Macro-averaged F1 score (treats all classes equally)
- **Accuracy Strict**: Accuracy where all tokens in a sequence must be correct

### Per-Class Metrics

For each class (0-7):
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in this class

### Metric Interpretation

- **F1 Micro**: Good for overall performance assessment
- **F1 Macro**: Better for class-imbalanced scenarios (gives equal weight to all classes)
- **Per-class F1**: Identifies which classes are difficult to predict

## Notes

- The model uses ALiBi positional encoding (from pretrained model)
- Token-level classification means each token gets a class label
- Labels should be aligned with input_ids (accounting for CLS/SEP tokens)
- The `ignore_index=-100` is used for padding tokens
- Best model is selected based on `eval_f1_macro` by default
- Training uses distributed data parallel (DDP) for multi-GPU setups

## References

- Orthoformer: Foundation model for protein sequences
- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- HuggingFace Transformers: [Documentation](https://huggingface.co/docs/transformers/)
- Token Classification: [HuggingFace Guide](https://huggingface.co/docs/transformers/tasks/token_classification)

