# Orthoformer Models

This directory contains pre-trained Orthoformer model files.

## Model Download

All pre-trained models can be downloaded from Hugging Face:

**Model Repository**: https://huggingface.co/jackkuo/Orthoformer

### Available Models

1. **model_3M_2048_v5**
   - Positional Encoding: Standard Position Embeddings
   - use_alibi: False
   - Features: Uses traditional learnable position embeddings

2. **model_3M_2048_v8**
   - Positional Encoding: ALiBi (Attention with Linear Biases)
   - use_alibi: True
   - Features: Uses ALiBi positional encoding, better handling of long sequences, no position embedding parameters required

## Download Methods

### Method 1: Using Hugging Face CLI

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download entire model repository
huggingface-cli download jackkuo/Orthoformer --local-dir ./model

# Or download specific model
huggingface-cli download jackkuo/Orthoformer/model_3M_2048_v5 --local-dir ./model/model_3M_2048_v5
huggingface-cli download jackkuo/Orthoformer/model_3M_2048_v8 --local-dir ./model/model_3M_2048_v8
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
    allow_patterns="model_3M_2048_v5/*",
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
# Using model_3M_2048_v5 (standard positional encoding)
python feature_extraction_example.py --model_dir model/model_3M_2048_v5 --use_alibi False

# Using model_3M_2048_v8 (ALiBi positional encoding)
python feature_extraction_example.py --model_dir model/model_3M_2048_v8 --use_alibi True
```

## Notes

- Model files are large, ensure you have sufficient disk space
- Download speed depends on network connection, recommend using a stable network environment
- If download is interrupted, you can re-run the download command, the tool will automatically resume
