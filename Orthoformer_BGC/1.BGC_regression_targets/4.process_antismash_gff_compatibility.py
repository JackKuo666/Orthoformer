#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import argparse
import multiprocessing
from typing import Dict, Any, List

def parse_gff_attributes(attr_str: str) -> Dict[str, Any]:
    """
    A robust parser for the GFF attribute column (column 9).
    It handles key=value pairs and parses nested attributes within the 'Note' field.
    """
    attributes = {}
    if not isinstance(attr_str, str):
        return attributes

    # Split into key-value pairs
    pairs = attr_str.strip().split(';')
    
    for pair in pairs:
        if '=' not in pair:
            continue
        
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        # If the key is 'Note', its value might be a series of more key-value pairs
        if key == 'Note':
            sub_pairs = value.split(';')
            for sub_pair in sub_pairs:
                if '=' in sub_pair:
                    sub_key, sub_value = sub_pair.split('=', 1)
                    attributes.setdefault(sub_key.strip(), []).append(sub_value.strip())
        else:
            attributes.setdefault(key, []).append(value)

    # Consolidate multi-value attributes into single strings
    final_attrs = {}
    for key, val_list in attributes.items():
        final_attrs[key] = ' | '.join(sorted(list(set(val_list))))
        
    return final_attrs

def process_gff_file_final(gff_path: str, csv_output_dir: str, faa_output_dir: str):
    """
    Processes a single GFF file with the final, correct logic.
    """
    print(f"-> Processing: {os.path.basename(gff_path)}")
    
    try:
        # Step 1: Read the entire GFF into a DataFrame
        col_names = ['contig', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
        gff_df = pd.read_csv(gff_path, sep='\t', comment='#', header=None, names=col_names, quotechar='"', low_memory=False)

        # Separate the different feature types
        regions_df = gff_df[gff_df['type'] == 'region'].copy()
        seq_features_df = gff_df[gff_df['type'] == 'sequence_feature'].copy()
        cds_df_raw = gff_df[gff_df['type'] == 'CDS'].copy()
        
        if cds_df_raw.empty:
            print(f"  Info: No 'CDS' features found in {os.path.basename(gff_path)}.")
            return True

        # Step 2: Create the base output DataFrame from CDS features
        attributes_list = cds_df_raw['attributes'].apply(parse_gff_attributes).tolist()
        attributes_df = pd.DataFrame(attributes_list)
        output_df = pd.concat([cds_df_raw.reset_index(drop=True), attributes_df], axis=1).drop(columns='attributes')
        
        # Initialize new columns with default placeholder
        output_df['BGC'] = '-'
        output_df['product'] = '-'
        output_df['SMILES'] = '-'

        # Step 3: Iterate through each BGC region and annotate the output_df
        for _, region_row in regions_df.iterrows():
            region_contig = region_row['contig']
            region_start = region_row['start']
            region_end = region_row['end']
            
            # Extract product from the region's attributes
            region_attrs = parse_gff_attributes(region_row['attributes'])
            product_str = region_attrs.get('product', '-')

            # Extract SMILES from corresponding sequence_feature rows
            smiles_list = []
            # Find sequence_feature rows within the same contig and coordinates
            sf_mask = (seq_features_df['contig'] == region_contig) & \
                      (seq_features_df['start'] >= region_start) & \
                      (seq_features_df['end'] <= region_end)
            for sf_attr_str in seq_features_df.loc[sf_mask, 'attributes']:
                sf_attrs = parse_gff_attributes(sf_attr_str)
                if 'SMILES' in sf_attrs:
                    smiles_list.append(sf_attrs['SMILES'])
            smiles_str = ' | '.join(sorted(list(set(smiles_list)))) if smiles_list else '-'
            
            # Find all CDS rows that belong to this region
            cds_mask = (output_df['contig'] == region_contig) & \
                       (output_df['start'] >= region_start) & \
                       (output_df['end'] <= region_end)
            
            # Apply annotations ONLY to those specific rows
            output_df.loc[cds_mask, 'BGC'] = 'BGC'
            output_df.loc[cds_mask, 'product'] = product_str
            output_df.loc[cds_mask, 'SMILES'] = smiles_str

        # Step 4: Final cleanup and output generation
        base_name = os.path.splitext(os.path.basename(gff_path))[0]
        
        # Fill any cells that are still empty for any reason
        final_df = output_df.fillna('-')

        # Filter rows: keep only those with 'ctg' in locus_tag column
        original_rows = len(final_df)
        if 'locus_tag' in final_df.columns:
            final_df = final_df[final_df['locus_tag'].str.contains('ctg', na=False)]
            rows_after_filter = len(final_df)
            removed_rows = original_rows - rows_after_filter
            print(f"  Removed {removed_rows} rows without 'ctg' in locus_tag")
        else:
            print(f"  Warning: 'locus_tag' column not found in file")

        # Write CSV
        csv_output_path = os.path.join(csv_output_dir, f"{base_name}.csv")
        final_df.to_csv(csv_output_path, index=False)
        print(f"  Success: Wrote CSV to {csv_output_path}")

        # Write FASTA
        faa_output_path = os.path.join(faa_output_dir, f"{base_name}.faa")
        fasta_written = 0
        if 'locus_tag' in final_df.columns and 'translation' in final_df.columns:
            fasta_df = final_df[(final_df['translation'] != '-') & (final_df['locus_tag'] != '-')].copy()
            
            with open(faa_output_path, 'w') as f_out:
                for _, row in fasta_df.iterrows():
                    header = f">{row['contig']}|:|{row['locus_tag']}"
                    f_out.write(f"{header}\n{row['translation']}\n")
                    fasta_written += 1
            if fasta_written > 0:
                 print(f"  Success: Wrote {fasta_written} sequences to {faa_output_path}")
        
        return True

    except Exception as e:
        print(f"  Error: Failed during processing of {os.path.basename(gff_path)}. Reason: {e}")
        # To help debug, you can uncomment the next line to see the full error traceback
        # import traceback; traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Process complex antiSMASH GFF files to generate annotated CSV and FASTA files."
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the root directory containing GFF files.")
    parser.add_argument("--csv_dir", required=True, help="Path to the output directory for .csv files.")
    parser.add_argument("--faa_dir", required=True, help="Path to the output directory for .faa files.")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU cores to use for parallel processing.")
    args = parser.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.faa_dir, exist_ok=True)

    gff_files = [os.path.join(root, file) for root, _, files in os.walk(args.input_dir) for file in files if file.endswith(".gff") and not file.startswith('.')]
    
    if not gff_files:
        print(f"Error: No .gff files found in '{args.input_dir}'.")
        return

    print(f"Found {len(gff_files)} GFF files. Starting processing with {args.cpu} core(s)...")
    
    tasks = [(gff_path, args.csv_dir, args.faa_dir) for gff_path in gff_files]

    with multiprocessing.Pool(processes=args.cpu) as pool:
        results = pool.starmap(process_gff_file_final, tasks)

    success_count = sum(1 for res in results if res)
    print("\n--------------------------------------------------")
    print("Processing Summary:")
    print(f"  Total files found: {len(gff_files)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed or skipped: {len(gff_files) - success_count}")
    print(f"  CSV files are located in: {args.csv_dir}")
    print(f"  FASTA files are located in: {args.faa_dir}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()