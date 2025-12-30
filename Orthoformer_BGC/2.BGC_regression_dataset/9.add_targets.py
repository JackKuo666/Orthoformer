#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add regression targets to tokenized HuggingFace datasets.
"""

import argparse
import pandas as pd
from datasets import load_from_disk, Dataset, Features, Sequence, Value


def main():
    """Add target values from CSV to an existing HuggingFace dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--targets_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ds = load_from_disk(args.dataset)
    targets_df = pd.read_csv(args.targets_csv)
    targets_df = targets_df.set_index("Sample_ID")
    target_cols = targets_df.columns.tolist()

    sample_names = ds["sample_name"]
    missing = set(sample_names) - set(targets_df.index)
    if missing:
        raise ValueError(f"Missing {len(missing)} samples in targets file: {list(missing)[:5]}...")

    targets_list = []
    for name in sample_names:
        targets_list.append(targets_df.loc[name].tolist())

    ds_dict = {
        "input_ids": ds["input_ids"],
        "sample_name": ds["sample_name"],
        "targets": targets_list
    }

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "sample_name": Value("string"),
        "targets": Sequence(Value("float32"))
    })

    new_ds = Dataset.from_dict(ds_dict, features=features)
    new_ds.save_to_disk(args.output)
    print(f"Saved to {args.output}")
    print(f"Samples: {len(new_ds)}, Targets: {len(target_cols)} {target_cols}")

if __name__ == "__main__":
    main()