# Orthoformer Phylogeny

## Introduction

This directory contains tools and scripts for phylogenetic analysis using Orthoformer embeddings and traditional marker gene-based methods. The repository includes two main subprojects that serve different purposes in phylogenetic tree construction and evaluation.

## Overview

The `Orthoformer_Phylogeny` directory provides two complementary approaches to phylogenetic analysis:

1. **ar53_r226**: Evaluation of Orthoformer embeddings for phylogenetic tree construction
2. **pathogens**: Traditional marker gene-based phylogenetic tree construction

## Directory Structure

```
Orthoformer_Phylogeny/
├── readme.md                          # This file
├── build_tree_from_embeddings.py     # Tree construction from embeddings
├── generate_embeddings_cli.py        # Embedding generation utility
├── orthoformer_model.py              # Pretraining utilities
├── ar53_r226/                         # Orthoformer evaluation project
│   ├── readme.md                      # Detailed documentation
│   ├── run.embedding_generate.sh      # Generate embeddings
│   ├── run.build_tree.nj.sh           # Build NJ tree
│   ├── run.build_tree.upgma.sh        # Build UPGMA tree
│   ├── calculate_rf_metrics.py        # RF metrics calculation
│   └── ...                            # See ar53_r226/readme.md
└── pathogens/                         # Marker gene-based phylogeny
    ├── readme.md                      # Detailed documentation
    ├── build_pathogen_phylogeny.py    # Main pipeline
    ├── create_itol_annotations.py     # iTOL annotations
    ├── run_iqtree.sh                  # IQ-TREE execution
    └── ...                            # See pathogens/readme.md
```

## Subprojects

### 1. ar53_r226: Orthoformer Embedding Evaluation

**Purpose**: Evaluate how well Orthoformer embeddings capture phylogenetic relationships by comparing predicted trees with GTDB reference trees.

**Key Features**:
- Uses Orthoformer model embeddings to build phylogenetic trees
- Compares predicted trees with gold-standard GTDB reference trees
- Calculates RF (Robinson-Foulds) and nRF (normalized RF) metrics
- Supports multiple tree construction methods (NJ, UPGMA)
- Evaluates on GTDB ar53_r226 dataset (archaeal genomes)

**Workflow**:
1. Generate genome embeddings using Orthoformer
2. Build phylogenetic trees from embeddings (distance-based methods)
3. Compare with GTDB reference tree using RF metrics

**Use Cases**:
- Evaluate Orthoformer model performance on phylogenetic tasks
- Compare embedding-based vs. marker gene-based approaches
- Assess how well learned representations capture evolutionary relationships

**Documentation**: See [ar53_r226/readme.md](ar53_r226/readme.md) for detailed instructions.

### 2. pathogens: Marker Gene-Based Phylogeny

**Purpose**: Build phylogenetic trees of pathogen genomes using traditional GTDB bac120 marker genes.

**Key Features**:
- Uses GTDB bac120 marker genes (standard phylogenetic markers)
- Follows standard GTDB workflow for tree construction
- Supports IQ-TREE and FastTree for tree building
- Generates iTOL annotation files for visualization
- Processes 829 pathogen genomes

**Workflow**:
1. Identify bac120 marker genes using HMMER
2. Extract and align marker gene sequences
3. Concatenate alignments from all markers
4. Build phylogenetic tree using IQ-TREE or FastTree
5. Generate visualization annotations

**Use Cases**:
- Build reference-quality phylogenetic trees
- Analyze pathogen genome relationships
- Create publication-ready phylogenetic trees
- Study evolutionary relationships in pathogen genomes

**Documentation**: See [pathogens/readme.md](pathogens/readme.md) for detailed instructions.

## Quick Start

### For Model Evaluation (ar53_r226)

```bash
cd ar53_r226
# Generate embeddings
bash run.embedding_generate.sh

# Build trees
bash run.build_tree.nj.sh
# or
bash run.build_tree.upgma.sh

# Evaluate
python calculate_rf_metrics.py
```

### For Tree Construction (pathogens)

```bash
cd pathogens
# Run full pipeline
python build_pathogen_phylogeny.py

# Or run IQ-TREE separately
bash run_iqtree.sh

# Create visualizations
python create_itol_annotations.py
```

## Shared Utilities

### build_tree_from_embeddings.py

Utility script for building phylogenetic trees from embeddings:
- Supports multiple distance metrics (Euclidean, cosine, etc.)
- Multiple tree construction methods (NJ, UPGMA)
- Automatic tree pruning for comparison
- iTOL annotation generation

### generate_embeddings_cli.py

CLI tool for generating embeddings:
- Loads pretrained Orthoformer models
- Supports ALiBi positional encoding
- Batch processing with configurable parameters
- Multiple output modes (mean pooling, token-level)

## Dataset Download

The required datasets for this project can be downloaded from the Hugging Face repository:

### Phylogeny Dataset

**Download URL:** https://huggingface.co/datasets/jackkuo/Orthoformer/tree/main/Downstream_Tasks_dataset/Phylogeny_dataset

**Dataset Contents:**

- **pathogens_faa.tar.gz**: Compressed file containing pathogen FASTA sequences for phylogenetic analysis

**Note**: 
- Extract the `pathogens_faa.tar.gz` file after downloading using: `tar -xzf pathogens_faa.tar.gz`
- Place the downloaded dataset files in the appropriate directories as specified in the subproject documentation.

## Requirements

### Common Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- NumPy, Pandas
- Biopython
- Conda environment: `orthoformer`

### ar53_r226 Specific

- `ete3` - for tree operations and RF calculation
- `scikit-bio` - for NJ tree construction
- `scipy` - for UPGMA tree construction
- Pretrained Orthoformer model

### pathogens Specific

- **HMMER** - for marker gene identification
- **IQ-TREE2** - for phylogenetic tree construction
- **FastTree** - alternative tree construction tool

## Use Case Selection

### Choose ar53_r226 if you want to:

- Evaluate Orthoformer model performance
- Compare embedding-based vs. traditional methods
- Assess how well learned representations capture phylogeny
- Work with archaeal genomes (GTDB ar53)
- Calculate quantitative metrics (RF/nRF)

### Choose pathogens if you want to:

- Build reference-quality phylogenetic trees
- Use standard GTDB marker gene workflow
- Analyze pathogen genomes
- Create publication-ready trees
- Work with protein sequences directly

## Output Files

### ar53_r226 Outputs

- `embeddings/`: Generated genome embeddings
- `nj_tree/` or `upgma_tree/`: Phylogenetic trees
- `rf_metrics.txt`: RF and nRF metrics
- `taxonomy_maps/`: Taxonomy mapping files

### pathogens Outputs

- `phylo_tree/concatenated_alignment.faa`: Concatenated alignment
- `phylo_tree/pathogen_phylogeny_iqtree.treefile`: IQ-TREE tree
- `phylo_tree/pathogen_phylogeny_fasttree.nwk`: FastTree tree
- `itol_annotations/`: iTOL visualization files

## Visualization

Both projects support visualization using iTOL (Interactive Tree of Life):

1. Upload tree files (`.treefile` or `.nwk`) to [iTOL](https://itol.embl.de/)
2. Add annotation files from respective directories
3. Customize visualization as needed

## Notes

- **ar53_r226** focuses on **evaluation** - comparing predicted trees with reference
- **pathogens** focuses on **construction** - building trees from marker genes
- Both use GTDB taxonomy as reference
- Both support iTOL visualization
- Tree formats are compatible (Newick format)

## References

- GTDB: [Genome Taxonomy Database](https://gtdb.ecogenomic.org/)
- IQ-TREE: [IQ-TREE Documentation](http://www.iqtree.org/)
- iTOL: [Interactive Tree of Life](https://itol.embl.de/)
- Orthoformer: Foundation model for protein sequences

## Getting Help

For detailed documentation on each subproject:
- **ar53_r226**: See [ar53_r226/readme.md](ar53_r226/readme.md)
- **pathogens**: See [pathogens/readme.md](pathogens/readme.md)

