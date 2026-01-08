# Orthoformer Taxon Dataset

This directory contains the datasets used for taxonomy classification tasks in the Orthoformer project.

## Download Instructions

Please download the required datasets from the following repository:

**Download URL:** https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Downstream_Tasks_dataset/Orthoformer_Taxon_dataset

## Dataset Structure

After downloading, place the dataset files in this directory. The expected structure includes:

- Training metadata files (e.g., `train_aug2_genus_s2000_taxid.csv`)
- Test metadata files (e.g., `UHGG2_5genus.csv`)
- Precomputed embedding files (if available)
- Distance map files (generated during training)

## Usage

Refer to the main `README.md` in the parent directory for detailed usage instructions on:
- Embedding distance calculation
- Model training
- Taxonomy inference

## Notes

- All embeddings should be generated using the Orthoformer foundation model
- Ensure that file paths in the scripts match your local dataset location
- Update embedding file paths in `src/CLEAN/utils.py` if necessary
