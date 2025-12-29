# GTDB ar53_r226 Phylogenetic Tree Evaluation

## Introduction

This directory contains scripts for evaluating Orthoformer model performance on phylogenetic tree construction using the GTDB (Genome Taxonomy Database) ar53_r226 dataset. The evaluation pipeline:

1. **Generate Embeddings**: Create genome embeddings using Orthoformer model
2. **Build Trees**: Construct phylogenetic trees from embeddings using distance-based methods (NJ or UPGMA)
3. **Evaluate**: Compare predicted trees with GTDB reference tree using RF (Robinson-Foulds) and nRF (normalized RF) metrics

This evaluation demonstrates how well Orthoformer embeddings capture phylogenetic relationships compared to the gold-standard GTDB taxonomy.

## Directory Structure

```
ar53_r226/
├── run.embedding_generate.sh        # Generate embeddings script
├── run.build_tree.nj.sh              # Build tree using Neighbor-Joining
├── run.build_tree.upgma.sh           # Build tree using UPGMA
├── calculate_rf_metrics.py           # Calculate RF/nRF metrics
├── datasets/                          # GTDB ar53 dataset
│   └── ar53.dataset                   # HuggingFace dataset
├── embeddings/                        # Generated embeddings
├── gtdb_ref_trees/                   # GTDB reference trees
│   └── ar53_r226.nwk                 # Reference phylogenetic tree
├── metadata/                          # Genome metadata
│   ├── ar53_metadata.tsv             # Full metadata
│   ├── ar53_rep.tsv                   # Representative genomes
│   └── sele_accession_ids.txt         # Selected accession IDs
├── taxonomy_maps/                     # Taxonomy mapping files
│   ├── taxonomy_map.phylum.tsv       # Phylum-level taxonomy
│   ├── taxonomy_map.class.tsv         # Class-level taxonomy
│   ├── taxonomy_map.order.tsv        # Order-level taxonomy
│   ├── taxonomy_map.family.tsv        # Family-level taxonomy
│   ├── taxonomy_map.genus.tsv         # Genus-level taxonomy
│   ├── taxonomy_map.species.tsv       # Species-level taxonomy
│   ├── itol.phylum.txt                # iTOL phylum annotations
│   ├── itol.class.txt                 # iTOL class annotations
│   ├── itol.order.txt                 # iTOL order annotations
│   ├── itol.family.txt                # iTOL family annotations
│   ├── itol.genus.txt                 # iTOL genus annotations
│   └── ar53_r226.csv                  # Taxonomy mapping CSV
├── nj_tree/                           # NJ tree outputs
│   ├── ar53_r226.nj_skbio.nwk         # NJ tree file
│   ├── ar53_r226.pruned_pred.nwk      # Pruned predicted tree
│   ├── ar53_r226.pruned_ref.nwk       # Pruned reference tree
│   └── itol_labels.txt                # iTOL labels
└── upgma_tree/                        # UPGMA tree outputs
    ├── ar53_r226.upgma_scipy.nwk      # UPGMA tree file
    ├── ar53_r226.pruned_pred.nwk      # Pruned predicted tree
    ├── ar53_r226.pruned_ref.nwk       # Pruned reference tree
    └── itol_labels.txt                # iTOL labels
```

## Requirements

- Python 3.8+
- Required Python packages:
  - `ete3` - for tree operations and RF calculation
  - `scikit-bio` - for NJ tree construction
  - `scipy` - for UPGMA tree construction
  - `numpy`, `pandas`
- Orthoformer model: `model_3M_2048_v10`
- GTDB ar53_r226 reference tree

## Workflow

### 1. Generate Embeddings

Generate genome embeddings using Orthoformer model:

```bash
bash run.embedding_generate.sh
```

**What it does:**
- Loads GTDB ar53 dataset
- Generates embeddings using Orthoformer model_3M_2048_v10
- Saves embeddings to `embeddings/` directory

**Script details:**
```bash
python ../scripts/generate_embeddings_cli.py \
    --model_dir ../../foundation_model/model/model_3M_2048_v10 \
    --dataset_path datasets/ar53.dataset \
    --output_dir embeddings \
    --batch_size 32 \
    --model_max_length 2048 \
    --use_alibi \
    --output_mode tokens
```

**Parameters:**
- `--model_dir`: Path to Orthoformer model (model_3M_2048_v10)
- `--dataset_path`: Path to GTDB ar53 dataset
- `--output_dir`: Directory to save embeddings
- `--batch_size`: Batch size for inference (default: 32)
- `--model_max_length`: Maximum sequence length (default: 2048)
- `--use_alibi`: Enable ALiBi positional encoding
- `--output_mode`: Embedding mode - 'tokens' or 'mean'

### 2. Build Phylogenetic Trees

#### Neighbor-Joining (NJ) Tree

Build phylogenetic tree using Neighbor-Joining method:

```bash
bash run.build_tree.nj.sh
```

**What it does:**
- Loads embeddings from `embeddings/` directory
- Calculates pairwise distances (Euclidean)
- Constructs NJ tree using scikit-bio
- Prunes trees to common leaves with reference tree
- Generates iTOL annotation files

**Output files:**
- `nj_tree/ar53_r226.nj_skbio.nwk`: NJ tree in Newick format
- `nj_tree/ar53_r226.pruned_pred.nwk`: Pruned predicted tree
- `nj_tree/ar53_r226.pruned_ref.nwk`: Pruned reference tree
- `nj_tree/itol_labels.txt`: iTOL label annotations

#### UPGMA Tree

Build phylogenetic tree using UPGMA (Unweighted Pair Group Method with Arithmetic Mean):

```bash
bash run.build_tree.upgma.sh
```

**What it does:**
- Similar to NJ tree construction
- Uses UPGMA clustering algorithm (scipy)
- Produces ultrametric tree (molecular clock assumption)

**Output files:**
- `upgma_tree/ar53_r226.upgma_scipy.nwk`: UPGMA tree in Newick format
- `upgma_tree/ar53_r226.pruned_pred.nwk`: Pruned predicted tree
- `upgma_tree/ar53_r226.pruned_ref.nwk`: Pruned reference tree
- `upgma_tree/itol_labels.txt`: iTOL label annotations

**Tree Building Parameters:**
- `--method`: Tree construction method ('nj_skbio' or 'upgma_scipy')
- `--metric`: Distance metric ('euclidean', 'cosine', etc.)
- `--taxonomy_map`: Taxonomy mapping file for annotations
- `--ref_tree`: Reference tree for comparison
- `--no_l2`: Disable L2 normalization (if specified)
- `--input_samples`: File with selected accession IDs to include

### 3. Evaluate Tree Quality

Calculate RF (Robinson-Foulds) and nRF (normalized RF) metrics:

```bash
python calculate_rf_metrics.py
```

**What it does:**
- Compares predicted trees with GTDB reference tree
- Calculates RF distance (number of different bipartitions)
- Calculates normalized RF (nRF = RF / maxRF)
- Generates summary table and saves to `rf_metrics.txt`

**Output:**
- Console output with detailed comparison
- `rf_metrics.txt`: Summary table with RF metrics

**Metrics Explained:**
- **RF (Robinson-Foulds distance)**: Number of bipartitions that differ between trees
  - Lower is better (0 = identical trees)
  - Range: 0 to maxRF
- **nRF (normalized RF)**: RF divided by maximum possible RF
  - Range: 0.0 to 1.0
  - Lower is better (0.0 = identical trees)
- **maxRF**: Maximum possible RF for trees with n leaves
  - maxRF = 2(n-3) for unrooted trees

**Example output:**
```
Tree                            RF         maxRF      nRF        Leaves    
----------------------------------------------------------------------------
NJ (selected samples)           1234       5678       0.217234   829       
UPGMA (selected samples)       1456       5678       0.256428   829       
```

## Quick Start

1. **Generate embeddings**:
   ```bash
   bash run.embedding_generate.sh
   ```

2. **Build trees** (choose one or both):
   ```bash
   # NJ tree
   bash run.build_tree.nj.sh
   
   # UPGMA tree
   bash run.build_tree.upgma.sh
   ```

3. **Evaluate results**:
   ```bash
   python calculate_rf_metrics.py
   ```

4. **Visualize trees**:
   - Upload tree files (`.nwk`) to [iTOL](https://itol.embl.de/)
   - Add taxonomy annotation files from `taxonomy_maps/itol.*.txt`
   - Compare predicted vs reference trees

## Taxonomy Levels

The evaluation supports multiple taxonomic levels:
- **Phylum**: `taxonomy_map.phylum.tsv`
- **Class**: `taxonomy_map.class.tsv`
- **Order**: `taxonomy_map.order.tsv`
- **Family**: `taxonomy_map.family.tsv`
- **Genus**: `taxonomy_map.genus.tsv`
- **Species**: `taxonomy_map.species.tsv`

Each level has corresponding iTOL annotation files for visualization.

## Configuration

### Model Selection

The default model is `model_3M_2048_v10`. To use a different model:
1. Update `--model_dir` in `run.embedding_generate.sh`
2. Regenerate embeddings
3. Rebuild trees

### Distance Metrics

Available distance metrics for tree construction:
- `euclidean`: Euclidean distance (default)
- `cosine`: Cosine distance
- `manhattan`: Manhattan distance

Modify `--metric` parameter in tree building scripts.

### Sample Selection

To evaluate on a subset of genomes:
1. Create `metadata/sele_accession_ids.txt` with selected accession IDs (one per line)
2. Use `--input_samples` parameter in tree building scripts

## Troubleshooting

### Missing Embeddings

If embeddings are missing:
- Check that `run.embedding_generate.sh` completed successfully
- Verify embeddings directory contains `.npy` files
- Ensure dataset path is correct

### Tree Construction Fails

If tree construction fails:
- Check that embeddings are in correct format
- Verify reference tree file exists
- Check that taxonomy maps are properly formatted

### RF Calculation Errors

If RF calculation fails:
- Ensure both predicted and reference trees exist
- Check that trees have sufficient common leaves (≥4)
- Verify tree files are in correct Newick format

### Memory Issues

For large datasets:
- Reduce batch size in embedding generation
- Process samples in batches
- Use subset of genomes for initial testing

## Output Interpretation

### Tree Quality

- **nRF < 0.2**: Excellent agreement with reference
- **nRF 0.2-0.4**: Good agreement
- **nRF 0.4-0.6**: Moderate agreement
- **nRF > 0.6**: Poor agreement

### Method Comparison

- **NJ vs UPGMA**: 
  - NJ is generally more accurate for non-ultrametric data
  - UPGMA assumes molecular clock (ultrametric tree)
  - Compare both methods to assess robustness

### Visualization Tips

- Use iTOL to visualize both predicted and reference trees side-by-side
- Apply taxonomy annotations to identify clustering patterns
- Compare branch lengths and topology
- Look for systematic errors (e.g., certain phyla consistently misplaced)

## Notes

- The evaluation uses GTDB release 226 (r226) reference taxonomy
- ar53 dataset contains archaeal genomes from GTDB
- Embeddings are generated using token-level representations (not mean-pooled)
- Trees are automatically pruned to common leaves for fair comparison
- RF metrics are calculated on unrooted trees

## References

- GTDB: [Genome Taxonomy Database](https://gtdb.ecogenomic.org/)
- Robinson-Foulds distance: Standard metric for tree comparison
- iTOL: [Interactive Tree of Life](https://itol.embl.de/)
- Orthoformer: Foundation model for protein sequences

