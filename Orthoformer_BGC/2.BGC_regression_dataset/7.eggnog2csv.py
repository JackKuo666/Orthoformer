#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert eggNOG annotation files to sample-COG count CSV matrix.
"""

import os
import glob
import argparse
import pandas as pd


def parameters():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert eggNOG annotations to sample-COG count matrix."
    )
    parser.add_argument('--input_file', '-i', type=str, required=True,
                        help='Input eggNOG file (.emapper.annotations).')
    parser.add_argument('--output_file', '-o', type=str, required=True,
                        help='Output file (.csv).')
    parser.add_argument('--separator', '-s', type=str, default='|',
                        help='Separator in the gene name field. Default: |')
    parser.add_argument('--samplename_idx', '-dx', type=int, default=0,
                        help='Sample ID index after splitting gene name. Default: 0')
    parser.add_argument('--chunk_size', '-c', type=int, default=100000,
                        help='Chunk size for reading large files. Default: 100000')
    return parser.parse_args()


def get_first_og(ogs_str):
    """Extract the first ortholog group ID from the eggNOG_OGs field."""
    return ogs_str.split(',')[0].split('@')[0]


def process_eggnog_file(eggnog_file, output_file, separator='|',
                        samplename_idx=0, chunk_size=100000):
    """
    Process eggNOG annotation file and convert to sample-COG count CSV matrix.

    Args:
        eggnog_file: Path to input eggNOG annotation file (.emapper.annotations).
        output_file: Path to output CSV file.
        separator: Separator for extracting sample name from gene ID.
        samplename_idx: Index of sample name after splitting.
        chunk_size: Number of rows to read per chunk.
    """
    # Skip header lines, read in chunks for large files
    reader = pd.read_csv(eggnog_file, sep='\t', skiprows=4, chunksize=chunk_size)

    samples_lst = []
    cogs_lst = []

    for i, chunk in enumerate(reader):
        # Keep only rows with valid eggNOG_OGs
        chunk = chunk[chunk['eggNOG_OGs'].apply(
            lambda x: isinstance(x, str) and pd.notna(x)
        )]
        querys = chunk['#query'].astype(str).tolist()

        # Extract sample names
        samples = [x.split(separator)[samplename_idx] for x in querys]
        # Extract COG IDs (first OG only)
        cogs = [get_first_og(x) for x in chunk['eggNOG_OGs'].tolist()]

        samples_lst += samples
        cogs_lst += cogs

    # Build sample-COG dataframe
    og_df = pd.DataFrame.from_dict({
        'Sample': samples_lst,
        'COG': cogs_lst,
    })
    og_df.astype(str)

    og2sample = {}
    sample2og = {}
    all_ogs = []
    all_samples = []

    # Count COG occurrences per sample
    for s, mt in og_df.groupby(by='Sample'):
        all_samples.append(s)
        og_count = mt.COG.value_counts().to_dict()
        sample2og[s] = og_count

        for og, c in og_count.items():
            if og not in og2sample.keys():
                og2sample[og] = {}
            og2sample[og][s] = c

        all_ogs = list(set(all_ogs + list(og_count.keys())))

    # Sort COG IDs for consistent output
    sorted_all_ogs = sorted(all_ogs)

    og_count_lists = {}
    for og in all_ogs:
        og_count_lists[og] = [
            og2sample[og][s] if s in og2sample[og].keys() else 0
            for s in all_samples
        ]

    # Build final sample-COG count matrix
    all_data = pd.DataFrame.from_dict(og_count_lists)
    all_data.index = all_samples
    all_data.insert(0, 'Sample', all_samples)
    all_data.to_csv(output_file, index=False)


def batch_convert_annotations_to_csv(parent_dir, output_subdir="csv_dataset",
                                     annotations_suffix=".emapper.annotations"):
    """
    Batch convert all annotation files in subdirectories to CSV format.

    Args:
        parent_dir: Parent directory containing *_mapper subdirectories.
        output_subdir: Output subdirectory for CSV files.
        annotations_suffix: Suffix for annotation files.
    """
    for batch_name in os.listdir(parent_dir):
        batch_path = os.path.join(parent_dir, batch_name)
        if os.path.isdir(batch_path) and batch_name.endswith(".faa_mapper"):
            for fname in os.listdir(batch_path):
                if fname.endswith(annotations_suffix):
                    ann_path = os.path.join(batch_path, fname)
                    out_dir = os.path.join(output_subdir)
                    os.makedirs(out_dir, exist_ok=True)
                    csv_name = fname + ".csv"
                    out_csv = os.path.join(out_dir, csv_name)
                    print(f"Processing {ann_path} -> {out_csv}")
                    process_eggnog_file(ann_path, out_csv)


def merge_csvs_in_dir(csv_dir="csv_dataset", output_csv="csv_dataset/all_cogs.csv"):
    """
    Merge all CSV files in a directory by concatenating rows.

    Args:
        csv_dir: Directory containing CSV files.
        output_csv: Path to output merged CSV file.
    """
    all_csvs = glob.glob(os.path.join(csv_dir, '*.csv'))
    df_list = []

    for csv_file in all_csvs:
        print(f"Reading {csv_file}")
        df = pd.read_csv(csv_file)
        df_list.append(df)

    if df_list:
        merged = pd.concat(df_list, axis=0, ignore_index=True)
        merged.to_csv(output_csv, index=False)
        print(f"Merged {len(all_csvs)} files into {output_csv}")
    else:
        print("No CSV files found to merge.")


def main():
    args = parameters()
    process_eggnog_file(
        args.input_file,
        args.output_file,
        separator=args.separator,
        samplename_idx=args.samplename_idx,
        chunk_size=args.chunk_size
    )


if __name__ == '__main__':
    main()
