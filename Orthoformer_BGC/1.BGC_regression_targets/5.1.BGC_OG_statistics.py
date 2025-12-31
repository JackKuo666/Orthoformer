#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGC OG Statistics

Consolidates antiSMASH CSV files and calculates BGC OG statistics.
"""

import pandas as pd
import os
import argparse
import multiprocessing
import gc
import tempfile
import shutil
from pathlib import Path


def process_single_csv(file_path):
    """Process a single CSV file and extract required columns."""
    sample_id = os.path.splitext(os.path.basename(file_path))[0]
    try:
        df = pd.read_csv(file_path, low_memory=False)
        required_cols = ['contig', 'locus_tag', 'BGC', 'OG', 'product']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), sample_id
        df['locus_tag'] = df['locus_tag'].astype(str)
        return df[required_cols], sample_id
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return pd.DataFrame(), sample_id


def create_contig_group(row):
    """Create contig group identifier from contig and locus_tag."""
    try:
        if pd.isna(row['locus_tag']):
            return row['contig']
        locus_prefix = str(row['locus_tag']).split('_')[0]
        return f"{row['contig']}_{locus_prefix}"
    except Exception:
        return row['contig']


def process_batch_files(csv_files_batch, temp_dir, batch_id, num_workers):
    """Process a batch of CSV files and save to temporary parquet file."""
    with multiprocessing.Pool(processes=min(num_workers, len(csv_files_batch))) as pool:
        results = pool.map(process_single_csv, csv_files_batch)

    df_list = []
    for df, sample_id in results:
        if not df.empty:
            df['Sample_ID'] = sample_id
            df_list.append(df)

    if not df_list:
        return None, 0

    batch_data = pd.concat(df_list, ignore_index=True)
    batch_data['contig_group'] = batch_data.apply(create_contig_group, axis=1)

    parsed_tags = batch_data['locus_tag'].str.rpartition('_')
    batch_data['locus_number_for_sort'] = pd.to_numeric(parsed_tags[2], errors='coerce')
    batch_data.sort_values(by=['Sample_ID', 'contig_group', 'locus_number_for_sort'], inplace=True)
    batch_data.drop(columns=['locus_number_for_sort'], inplace=True)

    temp_file = os.path.join(temp_dir, f'batch_{batch_id}.parquet')
    batch_data.to_parquet(temp_file, index=False)

    processed_count = len(df_list)
    del df_list, batch_data
    gc.collect()

    return temp_file, processed_count


def merge_temp_files(temp_files, output_path):
    """Merge all temporary parquet files to final CSV output."""
    first_batch = True
    for temp_file in temp_files:
        batch_data = pd.read_parquet(temp_file)
        if first_batch:
            batch_data.to_csv(output_path, index=False)
            first_batch = False
        else:
            batch_data.to_csv(output_path, mode='a', header=False, index=False)
        del batch_data
        gc.collect()


def calculate_statistics(consolidated_file, output_path, chunk_size=50000):
    """Calculate BGC/OG statistics from consolidated file."""
    stats_list = []

    for chunk in pd.read_csv(consolidated_file, chunksize=chunk_size):
        grouped = chunk.groupby(['Sample_ID', 'contig_group'])

        for (sample_id, contig_name), group in grouped:
            total_gene_count = len(group)
            annotated_gene_count = group['OG'].ne('-').sum()
            bgc_in_group = group[group['BGC'].ne('-')].copy()

            if bgc_in_group.empty:
                stats_list.append({
                    'Sample_ID': sample_id,
                    'contig_group': contig_name,
                    'product': '-',
                    'total_gene_count': total_gene_count,
                    'annotated_gene_count': annotated_gene_count,
                    'bgc_gene_count': 0,
                    'bgc_annotated_gene_count': 0,
                    'OG_num': 0
                })
            else:
                parsed_tags = bgc_in_group['locus_tag'].str.rpartition('_')
                bgc_in_group['locus_prefix'] = parsed_tags[0]
                bgc_in_group['locus_number'] = pd.to_numeric(parsed_tags[2], errors='coerce')

                prefix_changed = bgc_in_group['locus_prefix'] != bgc_in_group['locus_prefix'].shift(1)
                number_not_sequential = bgc_in_group['locus_number'] != (bgc_in_group['locus_number'].shift(1) + 1)
                block_ids = (prefix_changed | number_not_sequential).cumsum()

                for _, bgc_block in bgc_in_group.groupby(block_ids):
                    stats_list.append({
                        'Sample_ID': sample_id,
                        'contig_group': contig_name,
                        'product': bgc_block['product'].mode().iloc[0],
                        'total_gene_count': total_gene_count,
                        'annotated_gene_count': annotated_gene_count,
                        'bgc_gene_count': len(bgc_block),
                        'bgc_annotated_gene_count': bgc_block['OG'].ne('-').sum(),
                        'OG_num': bgc_block[bgc_block['OG'].ne('-')]['OG'].nunique()
                    })

        del chunk
        gc.collect()

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_path, index=False)
    print(f"Statistics saved to '{output_path}'")
    return stats_df


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate antiSMASH CSV files and calculate BGC/OG statistics."
    )
    parser.add_argument("-i", "--input_path", required=True,
                        help="Input directory containing CSV files or a single consolidated CSV file.")
    parser.add_argument("-o", "--output_file",
                        help="Output consolidated CSV file path (required if input is a directory).")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of CPU cores for parallel processing (default: 4).")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of files to process per batch (default: 1000).")
    args = parser.parse_args()

    if os.path.isfile(args.input_path):
        output_basename = os.path.splitext(args.input_path)[0]
        stats_output = f"{output_basename}_stats.csv"
        calculate_statistics(args.input_path, stats_output)

    elif os.path.isdir(args.input_path):
        if not args.output_file:
            parser.error("-o/--output_file is required when input is a directory.")

        output_path = args.output_file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        csv_files = [os.path.join(root, f) for root, _, files in os.walk(args.input_path)
                     for f in files if f.endswith(".csv")]

        if not csv_files:
            print("Error: No CSV files found.")
            return

        print(f"Found {len(csv_files)} CSV files.")

        temp_dir = tempfile.mkdtemp(prefix='og_stats_')
        try:
            temp_files = []
            total_processed = 0

            for i in range(0, len(csv_files), args.batch_size):
                batch_files = csv_files[i:i + args.batch_size]
                batch_id = i // args.batch_size + 1
                print(f"Processing batch {batch_id}/{(len(csv_files) - 1) // args.batch_size + 1}...")

                temp_file, processed = process_batch_files(batch_files, temp_dir, batch_id, args.cpu)
                if temp_file:
                    temp_files.append(temp_file)
                    total_processed += processed

            if temp_files:
                print(f"Merging {len(temp_files)} batches...")
                merge_temp_files(temp_files, output_path)
                print(f"Consolidated data saved to '{output_path}'")

                stats_output = os.path.splitext(output_path)[0] + "_stats.csv"
                calculate_statistics(output_path, stats_output)
            else:
                print("No valid data to process.")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"Error: Invalid input path: {args.input_path}")


if __name__ == "__main__":
    main()