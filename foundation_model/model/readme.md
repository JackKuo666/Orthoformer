# Orthoformer Models

This directory contains pre-trained Orthoformer model files.

## Model Download

All pre-trained models can be downloaded from Hugging Face:

**Model Repository**: https://huggingface.co/jackkuo/Orthoformer

### Available Models

| Model | Training Genomes | Max Length | Hidden | Layers | Heads | Description |
|------|-----------------|-----------|--------|--------|-------|-------------|
| `model_3M_2048_v8` | 3M | 2048 | 512 | 6 | 8 | Base Orthoformer foundation model |
| `model_3M_2048_v10` | 3M | 2048 | 1024 | 12 | 16 | Large Orthoformer foundation model |
| `model_140k_2048_v18` | 140k | 2048 | 512 | 6 | 8 | Compact foundation model |

All foundation models use:

- **ALiBi positional encoding**: enables long-context modeling across variable-length microbial genomes, preserving functional relationships between orthologous groups.
- **Span-masked language modeling (span-MLM, span=3)**: 15% of OG tokens are masked or corrupted following a BERT-style scheme, allowing the model to learn co-occurrence patterns, functional modules, and evolutionary dependencies in a self-supervised manner.

## Download Methods

### Method 1: Using Hugging Face CLI

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download entire model repository
huggingface-cli download jackkuo/Orthoformer --local-dir ./model

# Or download specific model
huggingface-cli download jackkuo/Orthoformer/model_3M_2048_v8 --local-dir ./model/model_3M_2048_v8
huggingface-cli download jackkuo/Orthoformer/model_140k_2048_v18 --local-dir ./model/model_140k_2048_v18
huggingface-cli download jackkuo/Orthoformer/model_3M_2048_v10 --local-dir ./model/model_3M_2048_v10
```

### Method 2: Using Python Code

```python
from huggingface_hub import snapshot_download

# Download entire model repository
snapshot_download(
    repo_id="jackkuo/Orthoformer",
    local_dir="./model",
    local_dir_use_symlinks=False
)

# Or download specific model
snapshot_download(
    repo_id="jackkuo/Orthoformer",
    allow_patterns="model_3M_2048_v8/*",
    local_dir="./model",
    local_dir_use_symlinks=False
)
```

### Method 3: Using Git LFS

```bash
# Clone model repository
git lfs install
git clone https://huggingface.co/jackkuo/Orthoformer ./model
```

## Model Usage

After downloading the models, you can use `feature_extraction_example.py` to load and use the models:

```bash
# Using model_3M_2048_v8 (ALiBi positional encoding)
# Note: --use_alibi flag can be omitted (auto-detects v8), or explicitly set
python feature_extraction_example.py --model_dir model/model_3M_2048_v8 --use_alibi

# Using model_140k_2048_v18 (ALiBi positional encoding)
python feature_extraction_example.py --model_dir model/model_140k_2048_v18 --use_alibi

# Using model_3M_2048_v10 (ALiBi positional encoding)
python feature_extraction_example.py --model_dir model/model_3M_2048_v10 --use_alibi
```

## Notes

- Model files are large, ensure you have sufficient disk space
- Download speed depends on network connection, recommend using a stable network environment
- If download is interrupted, you can re-run the download command, the tool will automatically resume
