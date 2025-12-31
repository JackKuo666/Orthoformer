#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple FAA files into a single file for eggNOG annotation.
Headers are modified to include the source filename for provenance tracking.
"""

import os
import argparse
from pathlib import Path
from typing import Iterator, Tuple


def parse_fasta(file_path: Path) -> Iterator[Tuple[str, str]]:
    """
    Parse a FASTA file and yield (header, sequence) tuples.

    Args:
        file_path: Path to the FASTA file.

    Yields:
        Tuple of (header, sequence) for each entry.
    """
    header = None
    sequence_parts = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(sequence_parts)
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)

        if header is not None:
            yield header, ''.join(sequence_parts)


def merge_faa_files(input_dir: str, output_file: str) -> None:
    """
    Merge all FAA files in the input directory into a single output file.
    Each header is prefixed with the source filename for provenance.

    Args:
        input_dir: Path to directory containing FAA files.
        output_file: Path to the merged output file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    faa_files = sorted(input_path.glob("*.faa"))

    if not faa_files:
        print(f"Error: No .faa files found in {input_dir}")
        return

    print(f"Found {len(faa_files)} FAA files in: {input_dir}")
    print(f"Output will be saved to: {output_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_sequences = 0

    with open(output_path, 'w') as out_f:
        for faa_file in faa_files:
            filename = faa_file.name
            file_sequences = 0

            for header, sequence in parse_fasta(faa_file):
                new_header = f">{filename}|{header}"
                out_f.write(f"{new_header}\n{sequence}\n")
                file_sequences += 1

            total_sequences += file_sequences
            print(f"  Processed {filename}: {file_sequences} sequences")

    print(f"\nMerge complete.")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple FAA files for eggNOG annotation."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to directory containing FAA files.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output merged FAA file.")
    args = parser.parse_args()

    merge_faa_files(args.input, args.output)


if __name__ == "__main__":
    main()
