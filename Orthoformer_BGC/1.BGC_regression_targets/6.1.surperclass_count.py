#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count superclass from BGC OG statistics CSV.
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-m', '--mapping', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    mapping = pd.read_csv(args.mapping)
    
    df['Sample_ID'] = df['Sample_ID'].str.replace('_OG', '', regex=False)
    
    all_samples = df['Sample_ID'].unique()
    
    df = df[df['product'] != '-'].copy()
    
    mapping_dict = dict(zip(mapping['product'].str.lower(), mapping['superclass']))
    
    unmatched = set()
    
    def convert_product(product):
        if ',' in product or '|' in product:
            return 'mixed'
        lower_product = product.lower()
        if lower_product in mapping_dict:
            return mapping_dict[lower_product]
        else:
            unmatched.add(product)
            return lower_product
    
    df['superclass'] = df['product'].apply(convert_product)
    
    if unmatched:
        print(f"Warning: {len(unmatched)} unmatched products found:")
        for p in sorted(unmatched):
            print(f"  - {p}")
        unmatch_file = args.output.replace('.csv', '_unmatched.txt')
        with open(unmatch_file, 'w') as f:
            for p in sorted(unmatched):
                f.write(f"{p}\n")
        print(f"Unmatched products saved to: {unmatch_file}")
    
    result = df.groupby(['Sample_ID', 'superclass']).size().unstack(fill_value=0)
    
    result = result.reindex(all_samples, fill_value=0)
    
    result.to_csv(args.output)

if __name__ == '__main__':
    main()