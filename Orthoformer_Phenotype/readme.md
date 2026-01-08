# Embedding for Genotype-Phenotype Prediction

This folder contains an example demonstrating how to use the embeddings provided by Orthoformer to predict the relationship between genotype and phenotype. The extraction of embeddings can be found in the foundation model under the main folder. Since the full set of phenotypes in the paper includes 126 traits, only microbial morphology (*gideon_Bacillus* or *coccobacillus*) is used here as an example for illustration purposes. The complete phenotypic data can be found at: [https://huggingface.co/datasets/macwiatrak/bacformer-genome-embeddings-with-phenotypic-traits-labels](https://huggingface.co/datasets/macwiatrak/bacformer-genome-embeddings-with-phenotypic-traits-labels)

## Project Structure

This task includes 5 folders, 1 bash script, a README, and python package requirements:

### 1. **bacformer_embedding**
- `gideon_Bacillus_or_coccobacillus.csv` – This file contains Bacformer embeddings, geneome IDs, and their corresponding phenotypic data in a one‑to‑one mapping.

### 2. **embeddings_4000_v2**
- Genome embedding data from Orthoformer‑140k version 2, where the model length is 4000 orthologous groups (OGs).

### 3. **OG_embedding**
- OG embedding data from Orthoformer‑3M version 10.

### 4. **salient_plot**
- Scripts and data for loading a 1D‑CNN model to compute and visualize saliency scores.

### 5. **scripts**
- Scripts for running examples of genotype‑phenotype prediction.

### 6. **batch_run.sh**
- A command‑line script for executing all examples.

### 7. **requirements.txt**
- Additional packages that may need to be installed (note that all required packages are typically already prepared when installing Orthoformer).

### 8. **README**
- This file.

## Dataset Download

The required datasets for this project can be downloaded from the Hugging Face repository:

### Phenotype Dataset

**Download URL:** https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Downstream_Tasks_dataset/Orthoformer_Phenotype_dataset

**Dataset Structure:**

After downloading, the dataset contains the following folders:

- **embeddings_4000_v2/**: Contains embeddings generated with sequence length 4000
- **OG_embedding/**: Contains orthologous group (OG) embeddings
- **bacformer_embedding/**: Contains BacFormer model embeddings

**Note**: 
- Place the downloaded dataset files in the appropriate directories as specified in the subproject documentation.


## Quick Start

1. Install required packages:
- pip install -r requirements.txt
2. Run the example:
- bash batch_run.sh

