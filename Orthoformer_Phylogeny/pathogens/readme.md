# Pathogen Phylogeny Construction

## Introduction

This directory contains scripts for building phylogenetic trees of pathogen genomes using GTDB (Genome Taxonomy Database) bac120 marker genes. The pipeline follows the standard GTDB workflow:

1. **Marker Gene Identification**: Identify bac120 marker genes using HMMER
2. **Sequence Extraction & Alignment**: Extract and align marker gene sequences
3. **Concatenation**: Concatenate alignments from all marker genes
4. **Tree Construction**: Build phylogenetic tree using IQ-TREE or FastTree
5. **Visualization**: Create iTOL annotation files for tree visualization

## Directory Structure

```
pathogens/
├── build_pathogen_phylogeny.py      # Main pipeline script
├── create_itol_annotations.py        # Generate iTOL annotation files
├── run_iqtree.sh                     # IQ-TREE execution script
├── isolates_multi_829_metadata.csv   # Genome metadata
├── pathogens_faa/                    # Protein sequence files (*.faa)
├── phylo_tree/                       # Phylogenetic tree outputs
│   ├── concatenated_alignment.faa    # Concatenated alignment
│   ├── pathogen_phylogeny_iqtree.treefile  # IQ-TREE tree
│   ├── pathogen_phylogeny_fasttree.nwk     # FastTree tree (backup)
│   └── pipeline.log                  # Pipeline log file
├── itol_annotations/                 # iTOL annotation files
│   ├── genus_colors.txt              # Genus color annotations
│   └── genus_labels.txt              # Genus label annotations
└── release226/                       # GTDB release 226 marker data
    └── markers/
        ├── bac120_msa_marker_info_r226.tsv  # Marker gene information
        └── r226_hmm/                 # HMM profiles
```

## Requirements

- Python 3.8+
- Required Python packages:
  - `Bio` (Biopython)
  - `pandas`
  - `numpy`
- External tools:
  - **HMMER** (hmmsearch) - for marker gene identification
  - **IQ-TREE2** - for phylogenetic tree construction (recommended)
  - **FastTree** - alternative tree construction tool (backup)
- Conda environment: `orthoformer`

## Workflow

### 1. Build Phylogenetic Tree

The main pipeline script performs all steps automatically:

```bash
python build_pathogen_phylogeny.py
```

**What it does:**
1. Scans `pathogens_faa/` directory for `.faa` files
2. Identifies bac120 marker genes using HMMER (e-value cutoff: 1e-10)
3. Extracts and aligns sequences for each marker gene
4. Concatenates all alignments
5. Builds phylogenetic tree using IQ-TREE (with FastTree as backup)
6. Generates alignment statistics and summary files

**Output files:**
- `phylo_tree/genome_list.txt`: List of analyzed genomes
- `phylo_tree/analyzed_genomes_metadata.csv`: Metadata for analyzed genomes
- `phylo_tree/marker_hits_summary.tsv`: Marker gene detection summary
- `phylo_tree/concatenated_alignment.faa`: Concatenated protein alignment
- `phylo_tree/alignment_stats.txt`: Detailed alignment statistics
- `phylo_tree/pathogen_phylogeny_iqtree.treefile`: Final phylogenetic tree (IQ-TREE)
- `phylo_tree/pathogen_phylogeny_iqtree.contree`: Consensus tree with bootstrap values
- `phylo_tree/pathogen_phylogeny_iqtree.iqtree`: Complete IQ-TREE analysis report
- `phylo_tree/pathogen_phylogeny_fasttree.nwk`: FastTree tree (if IQ-TREE fails)
- `phylo_tree/pipeline.log`: Complete pipeline log

**Configuration:**
The script uses GTDB release 226 bac120 markers. Key parameters:
- **E-value cutoff**: 1e-10
- **Threads**: Automatically set to min(32, CPU cores)
- **IQ-TREE model**: MFP (ModelFinder Plus - automatically selects best model)
- **Bootstrap replicates**: 1000
- **Timeout**: 4 hours for IQ-TREE (falls back to FastTree if exceeded)

### 2. Run IQ-TREE Separately

If you need to run IQ-TREE separately (e.g., after modifying alignment):

```bash
bash run_iqtree.sh
```

**Features:**
- Checks if IQ-TREE is already running (prevents duplicate runs)
- Runs in background with progress logging
- Automatically determines optimal thread count
- Creates PID file for process management

**Monitor progress:**
```bash
tail -f phylo_tree/iqtree_run.log
```

### 3. Create iTOL Annotations

Generate iTOL annotation files for tree visualization:

```bash
python create_itol_annotations.py
```

**What it does:**
1. Reads metadata from `isolates_multi_829_metadata.csv`
2. Extracts genus information from scientific names
3. Generates distinct colors for each genus
4. Creates iTOL annotation files for visualization

**Output files:**
- `itol_annotations/genus_colors.txt`: Color annotations by genus
- `itol_annotations/genus_labels.txt`: Label annotations by genus

**Usage with iTOL:**
1. Upload your tree file (`.treefile` or `.nwk`) to [iTOL](https://itol.embl.de/)
2. Upload annotation files from `itol_annotations/` directory
3. Customize visualization as needed

## Quick Start

1. **Prepare input files**:
   - Place `.faa` files in `pathogens_faa/` directory
   - Ensure `isolates_multi_829_metadata.csv` contains genome metadata

2. **Run the pipeline**:
   ```bash
   python build_pathogen_phylogeny.py
   ```

3. **Generate visualizations**:
   ```bash
   python create_itol_annotations.py
   ```

4. **Visualize tree**:
   - Upload tree file to iTOL
   - Add annotation files for enhanced visualization

## Configuration

### Marker Genes

The pipeline uses GTDB release 226 bac120 marker genes. Marker information is stored in:
- `release226/markers/bac120_msa_marker_info_r226.tsv`
- HMM profiles in `release226/markers/r226_hmm/`

### Customization

To modify pipeline parameters, edit `build_pathogen_phylogeny.py`:

```python
# HMMER parameters
EVALUE_CUTOFF = 1e-10  # E-value threshold for marker detection
THREADS = min(32, mp.cpu_count())  # Number of threads

# IQ-TREE parameters (in build_tree function)
"-m", "MFP",      # Model selection method
"-bb", "1000",    # Bootstrap replicates
```

## Troubleshooting

### IQ-TREE Timeout

If IQ-TREE times out (>4 hours), the script automatically falls back to FastTree:
- FastTree is faster but less accurate
- Output: `pathogen_phylogeny_fasttree.nwk`

### Missing Marker Genes

If many genomes have missing markers:
- Check HMM profiles are properly extracted
- Verify `.faa` files are in correct format
- Review `marker_hits_summary.tsv` for detection statistics

### Memory Issues

For large datasets (>1000 genomes):
- Reduce thread count: `THREADS = 16`
- Use FastTree instead of IQ-TREE for initial tree
- Consider splitting dataset into smaller batches

### HMMER Not Found

Ensure HMMER is installed and in PATH:
```bash
which hmmsearch
# If not found, install via conda:
conda install -c bioconda hmmer
```

## Output Interpretation

### Tree Files

- **`.treefile`**: Main phylogenetic tree in Newick format
- **`.contree`**: Consensus tree with bootstrap support values
- **`.iqtree`**: Complete analysis report with model selection details

### Alignment Statistics

`alignment_stats.txt` contains:
- Number of genomes analyzed
- Number of marker genes detected
- Alignment length and coverage
- Missing data statistics

### Marker Hits Summary

`marker_hits_summary.tsv` shows:
- Per-genome marker detection counts
- Overall marker coverage
- Genomes with low marker coverage (may need exclusion)

## Notes

- The pipeline processes all `.faa` files found in `pathogens_faa/` directory
- Genomes with very low marker coverage (<50%) may produce unreliable trees
- IQ-TREE is recommended for accuracy, but FastTree is faster for large datasets
- Bootstrap values indicate branch support (higher = more reliable)
- iTOL annotations use genus-level classification from metadata

## References

- GTDB: [Genome Taxonomy Database](https://gtdb.ecogenomic.org/)
- IQ-TREE: [IQ-TREE Documentation](http://www.iqtree.org/)
- FastTree: [FastTree Documentation](http://www.microbesonline.org/fasttree/)
- iTOL: [Interactive Tree of Life](https://itol.embl.de/)

