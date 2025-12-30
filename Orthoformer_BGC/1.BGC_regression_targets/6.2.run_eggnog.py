#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run eggNOG-mapper on FAA files for functional annotation.
"""

import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


def run_eggnog_for_file(faa_file: Path, output_dir: Path, data_dir: str, cpus: int) -> str:
    """
    Run eggNOG-mapper on a single FAA file.

    Args:
        faa_file: Path to the input FAA file.
        output_dir: Path to the output directory.
        data_dir: Path to eggNOG database directory.
        cpus: Number of CPUs for eggNOG-mapper.

    Returns:
        Status message string.
    """
    base_name = faa_file.stem
    output_prefix = output_dir / base_name

    annotation_file = Path(f"{output_prefix}.emapper.annotations")
    if annotation_file.exists():
        return f"Skipped (exists): {faa_file.name}"

    print(f"Processing: {faa_file.name}")

    command = [
        "emapper.py",
        "-i", str(faa_file),
        "-m", "diamond",
        "--output", str(output_prefix),
        "--cpu", str(cpus),
        "--data_dir", data_dir
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        return f"Success: {faa_file.name}"
    except subprocess.CalledProcessError as e:
        return f"Failed: {faa_file.name} - {e.stderr[:200]}"
    except FileNotFoundError:
        return f"Error: emapper.py not found. Please install eggnog-mapper."


def main():
    parser = argparse.ArgumentParser(
        description="Run eggNOG-mapper on FAA files for functional annotation."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input directory containing FAA files or a single FAA file.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output directory for eggNOG results.")
    parser.add_argument("--data_dir", required=True,
                        help="Path to eggNOG database directory.")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of CPUs per eggNOG job (default: 4).")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel jobs (default: 1).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        faa_files = [input_path]
    elif input_path.is_dir():
        faa_files = sorted(input_path.glob("*.faa"))
    else:
        print(f"Error: Input path not found: {args.input}")
        return

    if not faa_files:
        print(f"Error: No FAA files found in {args.input}")
        return

    print(f"Found {len(faa_files)} FAA files to process.")
    print(f"Output directory: {args.output}")
    print(f"eggNOG database: {args.data_dir}")

    if args.jobs > 1:
        task_func = partial(
            run_eggnog_for_file,
            output_dir=output_path,
            data_dir=args.data_dir,
            cpus=args.cpu
        )
        with Pool(processes=args.jobs) as pool:
            results = pool.map(task_func, faa_files)
    else:
        results = []
        for faa_file in faa_files:
            result = run_eggnog_for_file(faa_file, output_path, args.data_dir, args.cpu)
            results.append(result)
            print(f"  {result}")

    print("\nProcessing complete.")
    success = sum(1 for r in results if r.startswith("Success"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    failed = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))

    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
