#!/usr/bin/env python3
"""
Build phylogenetic tree for pathogen genomes using GTDB bac120 marker genes.
This script follows the standard GTDB workflow:
1. Identify marker genes using HMMER
2. Extract and align sequences
3. Concatenate alignments
4. Build phylogenetic tree

Author: Adapted from bac120_r226 reference pipeline
Date: 2025-10-24
"""

import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from datetime import datetime

# Configuration
BASE_DIR = Path("/mnt/np1/Orthoformer_Phylogeny/pathogens")
GENOMES_DIR = BASE_DIR / "pathogens_faa"
OUTPUT_DIR = BASE_DIR / "phylo_tree"
METADATA_FILE = BASE_DIR / "isolates_multi_829_metadata.csv"
MARKER_INFO = Path("release226/markers/bac120_msa_marker_info_r226.tsv")
HMM_DIR = Path("release226/markers/r226_hmm")

# HMMER parameters
EVALUE_CUTOFF = 1e-10
THREADS = min(32, mp.cpu_count())

# Log file
LOG_FILE = OUTPUT_DIR / "pipeline.log"

def log_message(message, print_console=True):
    """Write message to log file and optionally print to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    
    if print_console:
        print(log_msg)
    
    if LOG_FILE.parent.exists():
        with open(LOG_FILE, 'a') as f:
            f.write(log_msg + "\n")

log_message("="*80)
log_message("Starting phylogenetic tree construction pipeline for pathogen genomes")
log_message("="*80)
log_message(f"Base directory: {BASE_DIR}")
log_message(f"Genomes directory: {GENOMES_DIR}")
log_message(f"Output directory: {OUTPUT_DIR}")
log_message(f"Metadata file: {METADATA_FILE}")
log_message(f"Using {THREADS} threads")

# Create output directories
(OUTPUT_DIR / "hmm_results").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "extracted_markers").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "aligned_markers").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Read marker information
log_message("\nReading marker gene information...")
marker_df = pd.read_csv(MARKER_INFO, sep='\t')
log_message(f"Total markers available: {len(marker_df)}")

# Get list of protein files
protein_files = sorted(GENOMES_DIR.glob("*.faa"))
log_message(f"\nFound {len(protein_files)} protein files")

# Create genome ID to protein file mapping
# Extract genome ID from filename (e.g., GCA_001077555.2_ASM107755v2_genomic.faa -> GCA_001077555.2)
genome_info = {}
for pfile in protein_files:
    # Extract GCA/GCF ID with version
    fname = pfile.stem  # Remove .faa
    parts = fname.split('_')
    if len(parts) >= 2:
        genome_id = f"{parts[0]}_{parts[1]}"  # e.g., GCA_001077555.2
        genome_info[genome_id] = pfile
    else:
        log_message(f"Warning: Could not parse genome ID from {pfile.name}")

log_message(f"Genomes with valid IDs: {len(genome_info)}")

# Save genome list
genome_list_file = OUTPUT_DIR / "genome_list.txt"
with open(genome_list_file, 'w') as f:
    for gid in sorted(genome_info.keys()):
        f.write(f"{gid}\n")
log_message(f"Genome list saved to {genome_list_file}")

# Load metadata
log_message("\nLoading metadata...")
metadata_df = pd.read_csv(METADATA_FILE)
log_message(f"Metadata contains {len(metadata_df)} entries")

# Save metadata subset for genomes we're analyzing
genome_ids_set = set(genome_info.keys())
metadata_subset = metadata_df[metadata_df['Assembly'].isin(genome_ids_set)]
metadata_subset_file = OUTPUT_DIR / "analyzed_genomes_metadata.csv"
metadata_subset.to_csv(metadata_subset_file, index=False)
log_message(f"Saved metadata for {len(metadata_subset)} analyzed genomes to {metadata_subset_file}")


def run_hmmsearch(marker_id, hmm_file, genome_id, protein_file, output_file):
    """Run hmmsearch for a single marker against a genome."""
    cmd = [
        "hmmsearch",
        "--tblout", str(output_file),
        "-E", str(EVALUE_CUTOFF),
        "--cpu", "1",
        str(hmm_file),
        str(protein_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        log_message(f"Error running hmmsearch for {marker_id} on {genome_id}: {e}", print_console=False)
        return False


def parse_hmmsearch_output(output_file, evalue_cutoff=1e-10):
    """Parse hmmsearch tblout file and return best hit."""
    best_hit = None
    best_evalue = float('inf')
    
    if not output_file.exists():
        return None
    
    try:
        with open(output_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                target_name = parts[0]
                evalue = float(parts[4])
                
                if evalue < evalue_cutoff and evalue < best_evalue:
                    best_evalue = evalue
                    best_hit = target_name
    except Exception as e:
        log_message(f"Error parsing {output_file}: {e}", print_console=False)
        return None
    
    return best_hit


log_message("\n" + "="*80)
log_message("STEP 1: Running HMMER searches to identify marker genes")
log_message("="*80)

# Get list of HMM files
hmm_files = sorted(list(HMM_DIR.glob("*.hmm")))
log_message(f"Found {len(hmm_files)} HMM profiles")

# Run hmmsearch for each marker against each genome
marker_hits = defaultdict(lambda: defaultdict(str))  # marker_id -> genome_id -> protein_id

for idx, hmm_file in enumerate(hmm_files, 1):
    marker_id = hmm_file.stem
    log_message(f"\n[{idx}/{len(hmm_files)}] Processing marker: {marker_id}")
    
    genome_count = 0
    for genome_id, protein_file in genome_info.items():
        output_file = OUTPUT_DIR / "hmm_results" / f"{genome_id}.{marker_id}.tblout"
        
        if not output_file.exists():
            success = run_hmmsearch(marker_id, hmm_file, genome_id, protein_file, output_file)
            if not success:
                continue
        
        # Parse results
        best_hit = parse_hmmsearch_output(output_file, EVALUE_CUTOFF)
        if best_hit:
            marker_hits[marker_id][genome_id] = best_hit
            genome_count += 1
    
    log_message(f"  Found in {genome_count}/{len(genome_info)} genomes ({genome_count/len(genome_info)*100:.1f}%)")

# Save marker hits summary
hits_summary_file = OUTPUT_DIR / "marker_hits_summary.tsv"
with open(hits_summary_file, 'w') as f:
    f.write("Marker_ID\tGenomes_Found\tPercentage\n")
    for marker_id in sorted(marker_hits.keys()):
        count = len(marker_hits[marker_id])
        pct = count / len(genome_info) * 100
        f.write(f"{marker_id}\t{count}\t{pct:.2f}\n")

log_message(f"\nMarker hits summary saved to {hits_summary_file}")
log_message(f"Total markers with hits: {len(marker_hits)}")


log_message("\n" + "="*80)
log_message("STEP 2: Extracting marker gene sequences")
log_message("="*80)

# Load all protein sequences for each genome
log_message("Loading protein sequences from all genomes...")
genome_proteins = {}
for genome_id, protein_file in genome_info.items():
    try:
        genome_proteins[genome_id] = SeqIO.to_dict(SeqIO.parse(protein_file, "fasta"))
    except Exception as e:
        log_message(f"Error loading proteins for {genome_id}: {e}")

log_message(f"Loaded protein sequences for {len(genome_proteins)} genomes")

# Extract sequences for each marker
extracted_count = 0
for marker_id, genome_hits in marker_hits.items():
    output_file = OUTPUT_DIR / "extracted_markers" / f"{marker_id}.faa"
    
    sequences = []
    for genome_id, protein_id in genome_hits.items():
        if genome_id in genome_proteins and protein_id in genome_proteins[genome_id]:
            seq_record = genome_proteins[genome_id][protein_id]
            # Rename sequence with genome ID
            new_record = SeqRecord(
                seq_record.seq,
                id=genome_id,
                description=""
            )
            sequences.append(new_record)
    
    if sequences:
        SeqIO.write(sequences, output_file, "fasta")
        extracted_count += 1
        if extracted_count % 20 == 0:
            log_message(f"Extracted {extracted_count} markers so far...")

log_message(f"Successfully extracted {extracted_count} markers")


log_message("\n" + "="*80)
log_message("STEP 3: Aligning marker gene sequences with MUSCLE")
log_message("="*80)

# Align each marker gene
aligned_count = 0
failed_markers = []

for marker_file in sorted((OUTPUT_DIR / "extracted_markers").glob("*.faa")):
    marker_id = marker_file.stem
    output_file = OUTPUT_DIR / "aligned_markers" / f"{marker_id}.aln"
    
    # Skip if already aligned
    if output_file.exists():
        log_message(f"Alignment exists for {marker_id}, skipping...")
        aligned_count += 1
        continue
    
    # Check if file has enough sequences
    seq_count = len(list(SeqIO.parse(marker_file, "fasta")))
    if seq_count < 4:
        log_message(f"Skipping {marker_id}: only {seq_count} sequences (need at least 4)")
        failed_markers.append((marker_id, f"too few sequences: {seq_count}"))
        continue
    
    log_message(f"Aligning {marker_id} ({seq_count} sequences)...")
    
    # Run MUSCLE v5
    cmd = ["muscle", "-align", str(marker_file), "-output", str(output_file)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            aligned_count += 1
        else:
            log_message(f"  Error aligning {marker_id}: {result.stderr[:200]}")
            failed_markers.append((marker_id, "MUSCLE failed"))
    except subprocess.TimeoutExpired:
        log_message(f"  Timeout aligning {marker_id}")
        failed_markers.append((marker_id, "timeout"))
    except Exception as e:
        log_message(f"  Error: {e}")
        failed_markers.append((marker_id, str(e)))

log_message(f"\nSuccessfully aligned {aligned_count} markers")
if failed_markers:
    log_message(f"Failed to align {len(failed_markers)} markers")
    failed_file = OUTPUT_DIR / "failed_alignments.txt"
    with open(failed_file, 'w') as f:
        for marker_id, reason in failed_markers:
            f.write(f"{marker_id}\t{reason}\n")
    log_message(f"Failed markers list saved to {failed_file}")


log_message("\n" + "="*80)
log_message("STEP 4: Concatenating aligned markers into supermatrix")
log_message("="*80)

# Read all alignments
alignments = {}
for aln_file in sorted((OUTPUT_DIR / "aligned_markers").glob("*.aln")):
    marker_id = aln_file.stem
    try:
        aln = SeqIO.to_dict(SeqIO.parse(aln_file, "fasta"))
        alignments[marker_id] = aln
        if len(alignments) % 20 == 0:
            log_message(f"Loaded {len(alignments)} alignments so far...")
    except Exception as e:
        log_message(f"Error loading {marker_id}: {e}")

log_message(f"\nTotal alignments loaded: {len(alignments)}")

# Get all genome IDs present in alignments
all_genomes = set()
for aln in alignments.values():
    all_genomes.update(aln.keys())

all_genomes = sorted(all_genomes)
log_message(f"Total genomes in alignments: {len(all_genomes)}")

# Build concatenated alignment
concatenated = {}
marker_order = sorted(alignments.keys())

log_message("Building concatenated alignment...")
for genome_id in all_genomes:
    seq_parts = []
    for marker_id in marker_order:
        aln = alignments[marker_id]
        if genome_id in aln:
            seq_parts.append(str(aln[genome_id].seq))
        else:
            # Add gaps for missing markers
            if aln:
                gap_length = len(next(iter(aln.values())).seq)
                seq_parts.append("-" * gap_length)
    
    concatenated[genome_id] = "".join(seq_parts)

alignment_length = len(next(iter(concatenated.values()))) if concatenated else 0
log_message(f"\nConcatenated alignment length: {alignment_length} bp")
log_message(f"Number of sequences: {len(concatenated)}")

# Save concatenated alignment
concat_file = OUTPUT_DIR / "concatenated_alignment.faa"
records = [SeqRecord(Seq(seq), id=gid, description="") for gid, seq in concatenated.items()]
SeqIO.write(records, concat_file, "fasta")
log_message(f"Concatenated alignment saved to {concat_file}")

# Calculate alignment statistics
if concatenated:
    total_positions = len(next(iter(concatenated.values())))
    total_gaps = sum(seq.count('-') for seq in concatenated.values())
    total_positions_all = total_positions * len(concatenated)
    gap_percentage = (total_gaps / total_positions_all) * 100

    log_message(f"\nAlignment statistics:")
    log_message(f"  Total positions: {total_positions}")
    log_message(f"  Total sequences: {len(concatenated)}")
    log_message(f"  Gap percentage: {gap_percentage:.2f}%")

    # Save statistics
    stats_file = OUTPUT_DIR / "alignment_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Concatenated Alignment Statistics\n")
        f.write(f"="*50 + "\n")
        f.write(f"Number of markers: {len(alignments)}\n")
        f.write(f"Number of genomes: {len(concatenated)}\n")
        f.write(f"Alignment length: {total_positions} bp\n")
        f.write(f"Gap percentage: {gap_percentage:.2f}%\n")
        f.write(f"\nMarkers used ({len(marker_order)} total):\n")
        for marker_id in marker_order:
            aln = alignments[marker_id]
            if len(aln) > 0:
                marker_len = len(next(iter(aln.values())).seq)
                f.write(f"  {marker_id}: {len(aln)} sequences, {marker_len} bp\n")

    log_message(f"Statistics saved to {stats_file}")


log_message("\n" + "="*80)
log_message("STEP 5: Building phylogenetic tree with IQ-TREE")
log_message("="*80)

# Run IQ-TREE
tree_prefix = OUTPUT_DIR / "pathogen_phylogeny"
concat_file = OUTPUT_DIR / "concatenated_alignment.faa"

log_message(f"Running IQ-TREE with ModelFinder and 1000 ultrafast bootstrap...")
log_message(f"This may take several hours depending on dataset size...")
log_message(f"Output prefix: {tree_prefix}")

iqtree_cmd = [
    "iqtree2",
    "-s", str(concat_file),
    "-m", "MFP",  # ModelFinder Plus - automatically selects best protein model
    "-bb", "1000",  # Ultrafast bootstrap with 1000 replicates
    "-nt", str(THREADS),
    "-pre", str(tree_prefix),
    "-redo"
]

log_message(f"IQ-TREE command: {' '.join(iqtree_cmd)}")

try:
    log_message("Starting IQ-TREE (this will take a while)...")
    result = subprocess.run(iqtree_cmd, capture_output=True, text=True, timeout=14400)  # 4 hour timeout
    if result.returncode == 0:
        log_message("IQ-TREE completed successfully!")
        log_message(f"\nMain tree file: {tree_prefix}.treefile")
        log_message(f"Consensus tree: {tree_prefix}.contree")
        log_message(f"Log file: {tree_prefix}.iqtree")
    else:
        log_message("IQ-TREE failed!")
        log_message(f"Error: {result.stderr[:1000]}")
        # Try FastTree as backup
        log_message("\nTrying FastTree as backup...")
        fasttree_output = OUTPUT_DIR / "pathogen_phylogeny_fasttree.nwk"
        fasttree_cmd = f"FastTree -wag < {concat_file} > {fasttree_output}"
        log_message(f"FastTree command: {fasttree_cmd}")
        ft_result = subprocess.run(fasttree_cmd, shell=True, capture_output=True, text=True)
        if ft_result.returncode == 0:
            log_message(f"FastTree completed successfully: {fasttree_output}")
        else:
            log_message(f"FastTree also failed: {ft_result.stderr[:500]}")
except subprocess.TimeoutExpired:
    log_message("IQ-TREE timed out after 4 hours! Trying FastTree...")
    fasttree_output = OUTPUT_DIR / "pathogen_phylogeny_fasttree.nwk"
    fasttree_cmd = f"FastTree -wag < {concat_file} > {fasttree_output}"
    subprocess.run(fasttree_cmd, shell=True)
    log_message(f"FastTree output: {fasttree_output}")
except Exception as e:
    log_message(f"Error running IQ-TREE: {e}")


log_message("\n" + "="*80)
log_message("Pipeline completed!")
log_message("="*80)
log_message(f"\nOutput files in: {OUTPUT_DIR}")
log_message(f"  - genome_list.txt: List of genomes analyzed")
log_message(f"  - analyzed_genomes_metadata.csv: Metadata for analyzed genomes")
log_message(f"  - marker_hits_summary.tsv: Marker gene detection summary")
log_message(f"  - concatenated_alignment.faa: Concatenated protein alignment")
log_message(f"  - alignment_stats.txt: Detailed alignment statistics")
log_message(f"  - pathogen_phylogeny.treefile: Final phylogenetic tree (IQ-TREE)")
log_message(f"  - pathogen_phylogeny.contree: Consensus tree with bootstrap values")
log_message(f"  - pathogen_phylogeny.iqtree: Complete IQ-TREE analysis report")
log_message(f"  - pipeline.log: This log file")
log_message("\n" + "="*80)
log_message(f"Full log saved to: {LOG_FILE}")
log_message("Done!")

