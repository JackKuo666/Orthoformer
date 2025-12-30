# Orthoformer-BGC: Biosynthetic Gene Cluster Abundance Prediction

A computational pipeline for predicting biosynthetic gene cluster (BGC) abundance from microbial genomic data using deep learning (BERT-based) and machine learning (XGBoost) approaches.

## Overview

This project provides a complete workflow for:
1. **BGC Detection**: Running antiSMASH to identify biosynthetic gene clusters
2. **Data Processing**: Converting and processing antiSMASH outputs for downstream analysis
3. **Feature Engineering**: Generating eggNOG annotations and tokenized sequence representations
4. **Model Training**: Training multi-value regression models (BERT + custom heads) for BGC abundance prediction
5. **Classification**: Binary classification of samples using XGBoost

## Project Structure

```
Orthoformer_BGC/
├── 1.BGC_regression_targets/       # BGC target generation
│   ├── 1.batch_antismash.py       # Batch run antiSMASH via Docker
│   ├── 2.extract_gbk.py           # Extract GenBank files from antiSMASH output
│   ├── 3.antismash_gbk_to_gff3.py # Convert GBK to GFF3 format
│   ├── 4.process_antismash_gff_compatibility.py  # Process GFF files
│   ├── 5.1.BGC_OG_statistics.py  # Ortholog group statistics
│   ├── 5.2.merge_faa.py           # Merge protein FAA files
│   ├── 6.1.surperclass_count.py   # Count BGC superclasses
│   ├── 6.2.run_eggnog.py          # Run eggNOG annotation
│   └── product_superclass_antismash.csv  # BGC product-superclass mapping
│
├── 2.BGC_regression_dataset/       # Dataset preparation scripts
│   ├── 7.eggnog2csv.py            # Convert eggNOG annotations to CSV matrix
│   ├── 8.tokenizer.py             # Tokenize sequences for BERT input
│   └── 9.add_targets.py           # Add regression targets to dataset
│
├── 3.BGC_abundance_regression/     # Deep learning regression model
│   ├── multi_value_regression_model.py  # Model architecture
│   ├── train_multivalue_regression.py   # Training script
│   ├── evaluate_multivalue_regression.py  # Evaluation script
│   └── README.md              # Detailed usage instructions
│
├── 4.XGBoost_binary_classifiers/   # Machine learning classifier
│   └── xgboost_binary_classifiers.py  # XGBoost classifier
│
└── examples/                       # Example data for reproduction
    └── classification_data/
        └── bgc_features.csv
```

## Pipeline Workflow

Execute scripts in numerical order:

```
Step 1: 1.batch_antismash.py          → Run antiSMASH to identify BGCs
    ↓
Step 2: 2.extract_gbk.py              → Extract GenBank files
    ↓
Step 3: 3.antismash_gbk_to_gff3.py    → Convert GBK to GFF3
    ↓
Step 4: 4.process_antismash_gff_*.py  → Process and clean GFF files
    ↓
Step 5: 5.1.BGC_OG_statistics.py   → Statistical analysis
         5.2.merge_faa.py             → Merge protein sequences
    ↓
Step 6: 6.1.surperclass_count.py      → Count superclass distribution
         6.2.run_eggnog.py            → Run eggNOG annotation
    ↓
Step 7: 7.eggnog2csv.py               → Convert to CSV matrix
    ↓
Step 8: 8.tokenizer.py                → Tokenize for model input
    ↓
Step 9: 9.add_targets.py              → Add regression targets
    ↓
Model Training:
  - 3.BGC_abundance_regression/       → Train BERT regression model
  - 4.XGBoost_binary_classifiers/     → Train XGBoost classifier
```

## Requirements

### Software Dependencies
- Python >= 3.7
- Docker (for antiSMASH)
- antiSMASH Docker image (`antismash/standalone:8.0.2`)

### Python Packages
```bash
pip install -r requirements.txt
```

## Usage

### 1. BGC Detection with antiSMASH

```bash
python 1.BGC_regression_targets/1.batch_antismash.py \
    -i /path/to/fna_files \
    -o /path/to/output \
    --cpus 4 \
    --jobs 8
```

### 2. Data Processing

```bash
# Extract GBK files
python 1.BGC_regression_targets/2.extract_gbk.py \
    -i /path/to/antismash_output \
    -o /path/to/gbk_files

# Convert GBK to GFF3
python 1.BGC_regression_targets/3.antismash_gbk_to_gff3.py \
    -i /path/to/gbk_files \
    -o /path/to/gff_output

# Merge FAA files
python 1.BGC_regression_targets/5.2.merge_faa.py \
    -i /path/to/faa_files \
    -o /path/to/merged.faa

# Run eggNOG annotation
python 1.BGC_regression_targets/6.2.run_eggnog.py \
    -i /path/to/faa_files \
    -o /path/to/eggnog_output \
    --data_dir /path/to/eggnog_db \
    --cpu 8

```

### 3. Dataset Preparation

```bash
# Convert eggNOG annotations to sample-COG count matrix
python 2.BGC_regression_dataset/7.eggnog2csv.py \
    -i input.emapper.annotations \
    -o output.csv \
    -s '|' \
    -dx 0 \
    -c 100000

# Parameters:
#   -i: Input eggNOG annotation file (.emapper.annotations)
#   -o: Output CSV file
#   -s: Separator in gene name field (default: '|')
#   -dx: Sample ID index after splitting gene name (default: 0)
#   -c: Chunk size for reading large files (default: 100000)
```

```bash
# Tokenize COG matrix for BERT input
python 2.BGC_regression_dataset/8.tokenizer.py \
    --input_file cog_matrix.csv \
    --token_dictionary_file token_dict.pkl \
    --output_dir tokenized_output \
    --output_prefix my_dataset \
    --model_input_size 4096 \
    --special_token

# Parameters:
#   --input_file: Input CSV file (samples × COGs matrix)
#   --token_dictionary_file: Token dictionary file (.pkl or .csv)
#   --output_dir: Output directory for tokenized dataset
#   --output_prefix: Prefix for output dataset name
#   --model_input_size: Maximum sequence length (default: 4096)
#   --special_token: Add <cls> and <eos> tokens
#   --label_file: Optional label file for supervised learning
#   --chunk_size: Chunk size for processing (default: 100000)
```

```bash
# Add regression targets to tokenized dataset
python 2.BGC_regression_dataset/9.add_targets.py \
    --dataset tokenized_output/my_dataset.dataset \
    --targets_csv bgc_targets.csv \
    --output final_dataset

# Parameters:
#   --dataset: Path to tokenized HuggingFace dataset
#   --targets_csv: CSV file with Sample_ID column and target columns
#   --output: Output directory for final dataset with targets

# Note: targets_csv format:
# Sample_ID,target1,target2,...
# sample1,0.5,1.2,...
# sample2,0.8,0.9,...
```

### 4. Model Training

#### BERT Multi-Value Regression
```bash
python 3.BGC_abundance_regression/train_multivalue_regression.py \
    --train_dataset ./data/train \
    --val_dataset ./data/val \
    --test_dataset ./data/test \
    --pretrained_dir ./pretrained_bert \
    --output_dir ./trained_model \
    --num_targets 8 \
    --head mlp \
    --epochs 30
```

### 5. Model Evaluation

```bash
python 3.BGC_abundance_regression/evaluate_multivalue_regression.py \
    --model_dir ./trained_model \
    --pretrained_dir ./pretrained_bert \
    --test_dataset ./data/test \
    --output_dir ./results \
    --save_predictions
```

### 6. XGBoost Binary Classification
```bash
python 4.XGBoost_binary_classifiers/xgboost_binary_classifiers.py \
    input_data.csv \
    output_prefix
```

## Model Architecture

### BERT Multi-Value Regression
- **Encoder**: BERT with ALiBi positional encoding (via OrthoformerSelfAttention)
- **Regression Head**: MLP / BiLSTM / CNN options
- **Loss Functions**: MSE, MAE, or Huber loss
- **Multi-Target Support**: Predicts multiple continuous values simultaneously

### XGBoost Binary Classifier
- Adaptive hyperparameter tuning based on sample size
- Class imbalance handling with scale_pos_weight
- Two-phase grid search (coarse-to-fine)
- Cross-validation: Leave-One-Out (small datasets) or Stratified K-Fold

## Evaluation Metrics

- **Regression**: MSE, MAE, R²
- **Classification**: ROC-AUC, F1-score, Matthews Correlation Coefficient (MCC)

## BGC Superclasses

The pipeline categorizes BGCs into the following superclasses:
- Mixed
- NRPS (Non-ribosomal peptide synthetases)
- NRPS-related peptides
- PKS (Polyketide synthases)
- RiPPs (Ribosomally synthesized and post-translationally modified peptides)
- Saccharides & derivatives
- Terpenes
- Other

## External Resources

### Pretrained Model & Fine-tuned model & Token Dictionary & Example Regression Datasets

The foundation model, fine-tuned model, token dictionary and example regression datasets are available at:
- **Foundation Model**: https://huggingface.co/jackkuo/Orthoformer/tree/main/model_3M_2048_v10

- **Fine-tuned model**: https://huggingface.co/jackkuo/Orthoformer/tree/main/BGC_abundance_regression_model

- **Token Dictionary**: https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/foundation_model_dataset/token_dictionary

- **Example Regression Datasets**: 
https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Downstream_Tasks_dataset/BGC_abundance_regression_dataset

### eggNOG Database (v5.0.2)

This project requires the eggNOG 5.0.2 database for functional annotation.

#### Download Options

**Option 1: Using download script (recommended)**
```bash
pip install eggnog-mapper
export EGGNOG_DATA_DIR=/path/to/eggnog_data
mkdir -p $EGGNOG_DATA_DIR
download_eggnog_data.py -y
```

**Option 2: Manual download**

Download from: http://eggnog5.embl.de/download/emapperdb-5.0.2/

Required files:
- `eggnog.db.gz` (~6GB)
- `eggnog_proteins.dmnd.gz` (~5GB)

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added]
```

## References

1. Huerta-Cepas J, et al. eggNOG 5.0: a hierarchical, functionally and
   phylogenetically annotated orthology resource based on 5090 organisms
   and 2502 viruses. *Nucleic Acids Res.* 2019;47(D1):D309-D314.
   doi:10.1093/nar/gky1085

2. Cantalapiedra CP, et al. eggNOG-mapper v2: functional annotation,
   orthology assignments, and domain prediction at the metagenomic scale.
   *Mol Biol Evol.* 2021;38(12):5825-5829. doi:10.1093/molbev/msab293

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback, please open an issue on GitHub.
