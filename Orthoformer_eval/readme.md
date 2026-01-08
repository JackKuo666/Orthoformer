# Model Evaluation

## Introduction

This repository contains scripts for generating embeddings from multiple foundation models (Orthoformer, BacFormer, Evo-1) and evaluating their performance across 5 taxonomic levels: **phylum**, **order**, **class**, **family**, and **genus**.

The evaluation pipeline consists of:
1. **Embedding Generation**: Generate embeddings from different foundation models
2. **Model Comparison**: Compare performance across different models
3. **Visualization**: Generate visualizations for analysis

## Directory Structure

```
Orthoformer_eval/
├── scripts/              # Evaluation scripts
│   ├── s1_generate_embeddings_cli.py          # Orthoformer embedding generation
│   ├── s2_generate_bacformer_embeddings_dual_gpu.py  # BacFormer embedding generation
│   ├── s3_generate_evo_embeddings_multi_level.py    # Evo-1 embedding generation
│   ├── s4_multi_level_three_orthoformer_comparison.py  # Compare 3 Orthoformer models
│   ├── s5_multi_level_six_model_comparison.py         # Compare 6 models
│   └── s6_generate_additional_visualizations.py        # Generate visualizations
├── CNGBdb_fna/           # Genome sequence files (*.fna)
├── faa/                  # Protein sequence files (*.faa)
├── embeddings/           # Generated embeddings
│   ├── model_3M_2048_v10/
│   ├── model_3M_2048_v8/
│   ├── model_2048_v18/
│   ├── bacformer/
│   ├── evo/
│   └── evo_131k/
└── sampled_genomes.all.txt  # List of sampled genome IDs
```

## Dataset

The evaluation dataset is available on Hugging Face:

**Dataset Link**: [https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Orthoformer_eval_dataset](https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Orthoformer_eval_dataset)

The dataset contains the following files:

- `CNGBdb_dataset.tar.gz` - HuggingFace dataset for Orthoformer embedding generation
- `CNGBdb_faa.tar.gz` - Protein sequence files (*.faa) for BacFormer evaluation
- `CNGBdb_fna.tar.gz` - Genome sequence files (*.fna) for Evo-1 evaluation
- `CNGBdb_gtdbtkR226.csv` - GTDB-Tk R226 taxonomy annotations

After downloading, extract the tar.gz files to the appropriate directories as shown in the Directory Structure section above.

## Scripts

### 1. Embedding Generation

#### Orthoformer Embedding Generation

Generate embeddings using Orthoformer models with configurable parameters.

```bash
python scripts/s1_generate_embeddings_cli.py \
    --model_dir /path/to/model \
    --dataset_path CNGBdb_dataset \
    --sample_list sampled_genomes.all.txt \
    --output_dir embeddings/model_3M_2048_v10 \
    --batch_size 16 \
    --model_max_length 2048 \
    --device cuda:0 \
    --use_alibi \
    --output_mode mean
```

**Parameters:**
- `--model_dir`: Path to the pretrained Orthoformer model directory (required)
- `--dataset_path`: Path to the HuggingFace dataset directory (required)
- `--sample_list`: Optional text file listing sample names to process
- `--output_dir`: Directory to store generated embeddings (required)
- `--batch_size`: Batch size for inference (default: 32)
- `--model_max_length`: Maximum sequence length (default: 2048)
- `--device`: Device specifier, e.g., 'cuda:0' or 'cpu' (default: 'cuda:0')
- `--use_alibi`: Enable ALiBi positional encoding
- `--output_mode`: Embedding mode - 'mean' (mean pooling) or 'tokens' (token-level)

#### BacFormer Embedding Generation

Generate embeddings using BacFormer model with dual GPU support.

```bash
python scripts/s2_generate_bacformer_embeddings_dual_gpu.py \
    --faa_dir ../CNGBdb_faa \
    --sample_file ../sampled_genomes.all.txt \
    --output_dir ../embeddings/bacformer \
    --model_name macwiatrak/bacformer-masked-complete-genomes \
    --batch_size 128 \
    --max_n_proteins 6000 \
    --genome_pooling mean \
    --gpus 0,1
```

**Parameters:**
- `--faa_dir`: Directory containing ORF sequence files (*.faa files)
- `--sample_file`: Path to file containing sampled genome IDs
- `--output_dir`: Output directory for embeddings
- `--model_name`: BacFormer model name (default: "macwiatrak/bacformer-masked-complete-genomes")
- `--batch_size`: Batch size for inference (default: 128)
- `--max_n_proteins`: Maximum number of proteins per genome (default: 6000)
- `--genome_pooling`: Pooling method - 'mean', 'max', or 'cls' (default: 'mean')
- `--gpus`: Comma-separated list of GPU IDs (default: "0,1")

#### Evo-1 Embedding Generation

Generate embeddings using Evo-1 models for DNA sequences.

```bash
python scripts/s3_generate_evo_embeddings_multi_level.py \
    --fna_dir ../CNGBdb_fna \
    --sample_file ../sampled_genomes.all.txt \
    --output_dir ../embeddings/evo_131k \
    --model_name evo-1-131k-base \
    --device_id 0 \
    --max_sequence_length 8192 \
    --genome_pooling mean
```

**Parameters:**
- `--fna_dir`: Directory containing genome sequences (*.fna files)
- `--sample_file`: Path to file containing sampled genome IDs
- `--output_dir`: Output directory for embeddings
- `--model_name`: Evo model name (default: "evo-1-131k-base")
- `--device_id`: GPU device ID to use, use -1 for CPU (default: 1)
- `--max_sequence_length`: Maximum sequence length (default: 8192)
- `--genome_pooling`: Pooling method - 'mean', 'max', 'cls', or 'last' (default: 'mean')

**Pooling Methods:**
- `mean`: Average pooling over all tokens
- `max`: Max pooling over all tokens
- `cls`: Use first token (CLS token) embedding
- `last`: Use last token embedding

### 2. Model Comparison

#### Compare Three Orthoformer Models

Compare performance of three Orthoformer models: Orthoformer-140K, Orthoformer-3M (v8 & v10).

```bash
python scripts/s4_multi_level_three_orthoformer_comparison.py
```

This script evaluates embeddings across 5 taxonomic levels and generates comparison results.

#### Compare Six Models

Compare performance of six models: 3 Orthoformer models, Evo-1-8k, Evo-1-131k, and BacFormer.

```bash
python scripts/s5_multi_level_six_model_comparison.py
```

This script performs comprehensive evaluation across all models and taxonomic levels.

### 3. Visualization

Generate additional visualizations for model comparison and analysis.

```bash
python scripts/s6_generate_additional_visualizations.py
```

## Quick Start

1. **Download dataset**: Download the evaluation dataset from [Hugging Face](https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Orthoformer_eval_dataset) and extract the tar.gz files to the appropriate directories.

2. **Prepare data**: Ensure genome sequences (`.fna` files) and/or protein sequences (`.faa` files) are in the appropriate directories.

3. **Generate embeddings**: Run the embedding generation scripts for each model you want to evaluate.

4. **Compare models**: Run comparison scripts to evaluate model performance.

5. **Visualize results**: Generate visualizations for analysis.

## Notes

- All scripts support command-line arguments with sensible defaults
- GPU is recommended for faster processing, especially for large datasets
- Embeddings are saved as `.npy` files in the specified output directories
- Scripts automatically skip already-generated embeddings to support resuming interrupted runs
- For multi-GPU setups, use the `--gpus` parameter (e.g., `--gpus 0,1`)

## Troubleshooting

- **Out of Memory (OOM)**: Reduce `--batch_size` or `--max_sequence_length`
- **Missing files**: Check that input directories and sample files exist
- **Import errors**: Ensure all required packages are installed in the correct conda environment
- **GPU issues**: Use `--device_id -1` to fall back to CPU
