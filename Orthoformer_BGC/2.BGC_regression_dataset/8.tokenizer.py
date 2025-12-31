#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulk Orthologs Tokenizer for converting gene expression data to BERT input format.
"""

import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value


class BulkOrthologsTokenizer:
    """
    Tokenizer for converting bulk ortholog expression data to integer sequences
    suitable for BERT model input.
    """

    def __init__(
        self,
        token_dictionary_file,
        chunk_size=1024,
        model_input_size=4096,
        special_token=True,
        output_dir="tokenized_output",
        label_file=None,
    ):
        """
        Initialize BulkOrthologsTokenizer.

        Args:
            token_dictionary_file: Path to token dictionary file (.pkl or .csv).
            chunk_size: Chunk size for reading CSV files.
            model_input_size: Maximum sequence length for model input.
            special_token: Whether to use special tokens (<cls>, <eos>).
            output_dir: Output directory for tokenized datasets.
            label_file: Optional path to label file for supervised learning.
        """
        # Load token dictionary
        if token_dictionary_file.endswith(".pkl"):
            with open(token_dictionary_file, "rb") as f:
                self.orthologs_token_dict = pickle.load(f)
        elif token_dictionary_file.endswith(".csv"):
            dx = pd.read_csv(token_dictionary_file, header=None)
            self.orthologs_token_dict = dict(zip(dx[0], dx[1]))

        # Load label file if provided
        if label_file is not None:
            df = pd.read_csv(label_file, header=None, names=["sample_name", "label"])
            self.sample2label = dict(zip(
                df["sample_name"].astype(str),
                df["label"].astype(str)
            ))
        else:
            self.sample2label = None

        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.special_token = special_token

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Special token IDs
        self.cls_token = self.orthologs_token_dict.get("<cls>", None)
        self.eos_token = self.orthologs_token_dict.get("<eos>", None)

    def tokenize_csv(self, csv_path, output_prefix="bulk"):
        """
        Tokenize input CSV file and save as HuggingFace Dataset.

        Args:
            csv_path: Path to input CSV file.
            output_prefix: Prefix for output dataset filename.
        """
        reader = pd.read_csv(csv_path, index_col=0, chunksize=self.chunk_size)
        chunk_idx = 0

        all_tokenized = []
        all_sample_names = []

        for chunk in tqdm(reader, desc="Tokenizing CSV by chunk"):
            tokenized, sample_names = self.process_chunk(chunk)
            all_tokenized.extend(tokenized)
            all_sample_names.extend(sample_names)
            chunk_idx += 1

        # Verify sequence lengths
        lengths = [len(x) for x in all_tokenized]
        expected_length = self.model_input_size

        if not all(length == expected_length for length in lengths):
            print(f"Warning: Some sequences have unexpected length. Expected: {expected_length}")
            print(f"Length distribution: {set(lengths)}")

        # Build dataset dictionary
        ds_dict = {
            "input_ids": all_tokenized,
            "sample_name": [str(x) for x in all_sample_names],
            "length": lengths
        }
        if self.sample2label is not None:
            ds_dict["labels"] = [self.sample2label[x] for x in all_sample_names]

        # Define dataset features
        fea_items = {
            "input_ids": Sequence(Value("int32")),
            "sample_name": Value("string"),
            "length": Value("int32")
        }
        if self.sample2label is not None:
            fea_items["labels"] = Value("string")

        features = Features(fea_items)

        # Build and save HuggingFace Dataset
        ds = Dataset.from_dict(ds_dict, features=features)
        ds = ds.cast(features)

        output_path = (self.output_dir / output_prefix).with_suffix(".dataset")
        ds.save_to_disk(str(output_path))
        print(f"Saved HuggingFace Dataset to: {output_path}")

    def process_chunk(self, df_chunk):
        """
        Process a data chunk and convert gene expression to token sequences.

        This method processes each sample's gene expression data:
        1. Filter genes present in the token dictionary
        2. Sort genes by expression level (descending)
        3. Convert genes to token IDs
        4. Add special tokens and truncate to model input size

        Args:
            df_chunk: DataFrame with samples as rows and genes as columns.

        Returns:
            tuple: (tokens_list, sample_names)
                - tokens_list: List of token ID sequences for each sample
                - sample_names: Array of sample names
        """
        cols_in_token_dict = [
            c for c in df_chunk.columns if c in self.orthologs_token_dict
        ]
        X = df_chunk[cols_in_token_dict].to_numpy()

        tokens_list = []
        for i, row in enumerate(X):
            nonzero_idx = np.where(row > 0)[0]
            token_idx = [
                self.orthologs_token_dict[cols_in_token_dict[j]]
                for j in nonzero_idx
            ]

            values = row[nonzero_idx]
            if len(values) > 0:
                rank_idx = np.argsort(-values, kind="stable")
                sorted_token_idx = [token_idx[j] for j in rank_idx]
            else:
                sorted_token_idx = []

            if self.special_token:
                # Calculate available space for gene tokens
                available_space = self.model_input_size
                if self.cls_token is not None:
                    available_space -= 1
                if self.eos_token is not None:
                    available_space -= 1

                # Truncate gene sequence
                sorted_token_idx = sorted_token_idx[:available_space]

                # Add special tokens
                if self.cls_token is not None:
                    sorted_token_idx = [self.cls_token] + sorted_token_idx
                if self.eos_token is not None:
                    sorted_token_idx = sorted_token_idx + [self.eos_token]
            else:
                sorted_token_idx = sorted_token_idx[:self.model_input_size]

            tokens_list.append(sorted_token_idx)

        sample_names = df_chunk.index.astype(str).to_numpy()
        return tokens_list, sample_names


def parameters():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bulk Orthologs Tokenizer")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("--token_dictionary_file", type=str, required=True,
                        help="Path to the orthologs token dictionary file (.pkl or .csv).")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to the label file (optional).")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for processing. Default: 100000")
    parser.add_argument("--model_input_size", type=int, default=4096,
                        help="Model input sequence size. Default: 4096")
    parser.add_argument("--special_token", action="store_true", default=False,
                        help="Whether to use special tokens (<cls>, <eos>).")
    parser.add_argument("--output_dir", type=str, default="tokenized_output",
                        help="Output directory. Default: tokenized_output")
    parser.add_argument("--output_prefix", type=str, default="my_bulk",
                        help="Output filename prefix. Default: my_bulk")
    return parser.parse_args()


def main():
    args = parameters()
    tokenizer = BulkOrthologsTokenizer(
        token_dictionary_file=args.token_dictionary_file,
        label_file=args.label_file,
        chunk_size=args.chunk_size,
        model_input_size=args.model_input_size,
        special_token=args.special_token,
        output_dir=args.output_dir
    )
    tokenizer.tokenize_csv(args.input_file, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
