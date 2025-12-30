# Orthoformer

## Project Overview

Orthoformer is a BERT-based pre-trained model optimized for OG (Orthologous Groups) data, capable of:
- Tokenizing and encoding OG sequences
- Extracting OG feature representations (embeddings)
- Supporting GPU-accelerated inference
- Providing multiple feature extraction methods (CLS token, Mean pooling, and Attention pooling)

## Project Structure

```
foundation_model/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── feature_extraction_example.py      # Feature extraction example code
├── orthoformer_model.py               # Orthoformer model implementation (supports ALiBi and RoPE)
├── model/                             # Pre-trained models directory
│   ├── readme.md                      # Model download instructions
│   ├── model_3M_2048_v10/             # Model version v10 (ALiBi positional encoding)
│   │   ├── config.json                # Model configuration file
│   │   ├── model_config.json          # Custom model configuration
│   │   ├── model.safetensors          # Model weights
│   │   ├── vocab.txt                  # Vocabulary
│   │   ├── tokenizer_config.json     # Tokenizer configuration
│   │   └── special_tokens_map.json   # Special tokens mapping
│   └── model_3M_2048_v8/             # Model version v8 (ALiBi positional encoding)
│       ├── config.json                # Model configuration file
│       ├── model_config.json          # Custom model configuration
│       ├── generation_config.json     # Generation configuration
│       ├── model.safetensors          # Model weights
│       ├── vocab.txt                  # Vocabulary
│       ├── tokenizer_config.json     # Tokenizer configuration
│       └── special_tokens_map.json   # Special tokens mapping
└── datasets/                          # Datasets directory
    └── example/                        # Example dataset (Hugging Face datasets format)
        ├── data-00000-of-00001.arrow  # Dataset data file
        ├── dataset_info.json          # Dataset metadata
        └── state.json                 # Dataset state
```

## Requirements

- Python 3.12
- CUDA-supported GPU (recommended)
- At least 8GB GPU memory (for processing long sequences)

## Installation

```bash
pip install -r requirements.txt
```

## Model Download

Pre-trained models can be downloaded from Hugging Face:

**Model Repository**: https://huggingface.co/jackkuo/Orthoformer

For detailed download instructions, please refer to [model/readme.md](model/readme.md)

Quick download example:

```bash
# Using Hugging Face CLI
pip install huggingface-hub
huggingface-cli download jackkuo/Orthoformer --local-dir ./model
```

### Main Dependencies

- `torch`: PyTorch deep learning framework
- `transformers`: Hugging Face Transformers library
- `datasets`: Dataset processing library
- `accelerate`: Distributed training acceleration
- `deepspeed`: Deep learning optimization library
- `scikit-learn`: Machine learning tools
- `tensorboard`: Training monitoring
- `matplotlib`, `seaborn`: Data visualization

## Usage

### 1. Feature Extraction Example

Run the feature extraction example code:

```bash
# Using default model (model_3M_2048_v8, ALiBi=True)
python feature_extraction_example.py

# Using model_3M_2048_v10 (ALiBi positional encoding)
python feature_extraction_example.py --model_dir model/model_3M_2048_v10 --use_alibi
```

The script will:
- Automatically detect and use available GPU
- Load the pre-trained Orthoformer model
- Automatically apply ALiBi positional encoding if needed based on model type
- Process OG data
- Extract multiple types of feature representations:
  - CLS token embedding
  - Mean pooling embedding
  - Attention pooling embedding (if the model includes a classification head)

### 2. Code Functionality

#### Device Detection
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Automatically detects and uses GPU or CPU.

#### Model Loading
```python

# ALiBi positional encoding model (model_3M_2048_v8)
model_dir = "model/model_3M_2048_v8"
tokenizer = BertTokenizer.from_pretrained(model_dir)
# Manually apply ALiBi positional encoding
from orthoformer_model import OrthoformerSelfAttention
mlm_model = BertForMaskedLM.from_pretrained(model_dir)
model = mlm_model.bert
# Replace attention layers with ALiBi-supporting version
for layer in model.encoder.layer:
    layer.attention.self = OrthoformerSelfAttention(
        layer.attention.self,
        pos_kind="alibi",
        max_position_embeddings=model.config.max_position_embeddings
    )
```
Load pre-trained tokenizer and model. Note: Models using ALiBi positional encoding require manual application of custom attention layers.

#### Sequence Processing
- Automatically handles sequences of different lengths
- Supports padding and truncation operations
- Maximum sequence length limit is 2048 (adjustable)

#### Feature Extraction Methods

**CLS Token Embedding:**
```python
cls_embeddings = outputs.last_hidden_state[:, 0, :]
```
Extracts feature representation from the [CLS] token at the beginning of the sequence.

**Mean Pooling Embedding:**
```python
# Calculate average features of all valid tokens
mean_embeddings = sum_hidden / sum_mask
```
Calculates the average feature representation of all valid tokens in the sequence.

**Attention Pooling Embedding:**
```python
# Weighted pooling using trained classification head attention vector
attn_embeddings = torch.einsum("bl,blh->bh", weights, hidden_states)
```
Uses trained attention vectors to perform weighted summation of tokens, resulting in a more focused feature representation (only available when the model includes a classification head).

## Model Specifications

- **Model Type**: BERT (Bidirectional Encoder Representations from Transformers)
- **Maximum Sequence Length**: 2048 tokens
- **Parameter Count**: ~3M parameters
- **Optimization**: Optimized for OG data
- **Supported Devices**: CPU/GPU

### Available Models

1. **model_3M_2048_v8**
   - Positional Encoding: ALiBi (Attention with Linear Biases)
   - use_alibi: True
   - Features: Uses ALiBi positional encoding, better handling of long sequences, no position embedding parameters required

## Dataset

The project uses example datasets in Hugging Face datasets format, containing:
- Preprocessed OG sequences (input_ids)
- Sample names and metadata
- Corresponding labels for downstream tasks

## Performance Optimization

- **GPU Acceleration**: Automatically detects and uses CUDA GPU
- **Memory Management**: Intelligently handles long sequences to avoid memory overflow
- **Batch Processing**: Supports batch processing of multiple sequences
- **Attention Masking**: Properly handles padding tokens

## Output Format

Feature extraction results include:
- **Shape Information**: (batch_size, hidden_size)
- **Value Range**: Normalized feature vectors
- **Device Location**: Supports CPU/GPU output

## Notes

1. **Memory Usage**: Pay attention to GPU memory usage when processing long sequences
2. **Sequence Length**: All models have a default maximum length of 2048 tokens
3. **Device Compatibility**: Code automatically adapts to CPU/GPU environments
4. **Data Format**: Ensure input data format is correct
5. **ALiBi Model**: When using `model_3M_2048_v8`, must set `--use_alibi True`, otherwise the model cannot correctly load ALiBi positional encoding
6. **Standard Model**: When using `model_3M_2048_v5`, set `--use_alibi False` or omit this parameter

## Model Records

| Model | Training Genomes | Max Length | Hidden | Layers | Heads | Description |
|------|-----------------|-----------|--------|--------|-------|-------------|
| `model_3M_2048_v8` | 3M | 2048 | 512 | 6 | 8 | Base Orthoformer foundation model |
| `model_3M_2048_v10` | 3M | 2048 | 1024 | 12 | 16 | Large Orthoformer foundation model |
| `model_140k_2048_v18` | 140k | 2048 | 512 | 6 | 8 | Compact foundation model |

All foundation models use:

- **ALiBi positional encoding**: enables long-context modeling across variable-length microbial genomes, preserving functional relationships between orthologous groups.
- **Span-masked language modeling (span-MLM, span=3)**: 15% of OG tokens are masked or corrupted following a BERT-style scheme, allowing the model to learn co-occurrence patterns, functional modules, and evolutionary dependencies in a self-supervised manner.
