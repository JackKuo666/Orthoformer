#!/usr/bin/env python3
"""
Convert eggNOG-mapper output for one or more MAGs into a HuggingFace dataset
ready for Orthoformer embedding extraction.

Example (single MAG):
  python mag_preprocess.py \\
    --emapper MAG001.emapper.annotations \\
    --model_dir model/model_140k_2048_v18 \\
    --output_dir datasets/MAG001.dataset

Example (save intermediate OG count table):
  python mag_preprocess.py \\
    --emapper MAG001.emapper.annotations \\
    --model_dir model/model_140k_2048_v18 \\
    --output_dir datasets/MAG001.dataset \\
    --counts_tsv outputs/MAG001_og_counts.tsv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Sequence, Value


def load_vocab(model_dir: Path) -> dict[str, int]:
    vocab_path = model_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocab.txt not found in {model_dir}. Download a pretrained model first."
        )
    vocab: dict[str, int] = {}
    for idx, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()):
        token = line.strip()
        if token:
            vocab[token] = idx
    return vocab


def first_og(ogs_str: str) -> str:
    return ogs_str.split(",")[0].split("@")[0].strip()


def emapper_to_counts(
    emapper_path: Path,
    separator: str = "|",
    sample_idx: int = 0,
    mag_id: str | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Parse .emapper.annotations into a long OG count table."""
    reader = pd.read_csv(emapper_path, sep="\t", skiprows=4, chunksize=chunk_size, dtype=str)
    data: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "pfam_counter": Counter()})
    )

    for chunk in reader:
        chunk = chunk[chunk["eggNOG_OGs"].apply(lambda x: isinstance(x, str) and pd.notna(x))]
        if chunk.empty:
            continue

        queries = chunk["#query"].astype(str)
        ogs_col = chunk["eggNOG_OGs"].astype(str)
        pfams_col = chunk.get("PFAMs", pd.Series([""] * len(chunk)))

        for query, ogs_str, pfams_str in zip(queries, ogs_col, pfams_col):
            parts = query.split(separator)
            sample = mag_id or (parts[sample_idx] if len(parts) > sample_idx else parts[0])
            og = first_og(ogs_str)
            if not og:
                continue

            pfam_list: list[str] = []
            if isinstance(pfams_str, str) and pd.notna(pfams_str):
                pfam_list = [
                    p.strip() for p in pfams_str.split(",") if p and p.strip() and p.strip() != "-"
                ]

            cell = data[str(sample)][og]
            cell["count"] = int(cell["count"]) + 1
            if pfam_list:
                cell["pfam_counter"].update(pfam_list)

    rows: list[dict[str, object]] = []
    for sample, og_dict in data.items():
        for og, agg in og_dict.items():
            pfam_counter: Counter = agg["pfam_counter"]  # type: ignore[assignment]
            pfam_total = int(sum(pfam_counter.values()))
            max_pfam_count = int(max(pfam_counter.values())) if pfam_counter else 0
            if pfam_counter:
                sorted_pfams = sorted(pfam_counter.items(), key=lambda x: (-x[1], x[0]))
                pfams_str = ",".join(name for name, _ in sorted_pfams)
            else:
                pfams_str = ""
            rows.append(
                {
                    "Sample": sample,
                    "OG": og,
                    "Count": int(agg["count"]),
                    "PFAM_Count": pfam_total,
                    "max_PFAM_Count": max_pfam_count,
                    "PFAMs": pfams_str,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["Sample", "OG", "Count", "PFAM_Count", "max_PFAM_Count", "PFAMs"]
        )

    out_df = pd.DataFrame(rows)
    out_df.sort_values(
        by=["Sample", "Count", "PFAM_Count", "max_PFAM_Count", "PFAMs", "OG"],
        ascending=[True, False, False, False, True, True],
        inplace=True,
    )
    return out_df


def counts_to_dataset(
    counts_df: pd.DataFrame,
    vocab: dict[str, int],
    output_dir: Path,
    model_input_size: int = 2048,
    special_token: bool = True,
) -> Path:
    """Tokenize OG counts and save a HuggingFace dataset on disk."""
    if counts_df.empty:
        raise ValueError("No OG counts found. Check the emapper file and query ID format.")

    df = counts_df[counts_df["OG"].astype(str).isin(vocab.keys())].copy()
    if df.empty:
        raise ValueError("No OGs matched the model vocabulary. Check eggNOG annotation quality.")

    cls_id = vocab.get("<cls>")
    eos_id = vocab.get("<eos>")
    available = model_input_size
    if special_token:
        if cls_id is not None:
            available -= 1
        if eos_id is not None:
            available -= 1

    input_ids_list: list[list[int]] = []
    sample_names: list[str] = []
    lengths: list[int] = []

    for sample, grp in df.groupby("Sample", sort=False):
        grp_sorted = grp.sort_values(
            by=["Count", "PFAM_Count", "max_PFAM_Count", "PFAMs", "OG"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        token_ids = [vocab[str(og)] for og in grp_sorted["OG"].tolist()]
        token_ids = token_ids[:available]
        if special_token:
            if cls_id is not None:
                token_ids = [cls_id] + token_ids
            if eos_id is not None:
                token_ids = token_ids + [eos_id]
        else:
            token_ids = token_ids[:model_input_size]

        input_ids_list.append(token_ids)
        sample_names.append(str(sample))
        lengths.append(len(token_ids))

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "sample_name": Value("string"),
            "length": Value("int32"),
        }
    )
    ds = Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "sample_name": sample_names,
            "length": lengths,
        },
        features=features,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert eggNOG emapper output to an Orthoformer-ready dataset."
    )
    parser.add_argument("--emapper", type=str, required=True, help="Path to .emapper.annotations")
    parser.add_argument("--model_dir", type=str, required=True, help="Pretrained model folder (for vocab.txt)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output HuggingFace dataset directory (e.g. datasets/MAG001.dataset)",
    )
    parser.add_argument(
        "--mag_id",
        type=str,
        default=None,
        help="Force all proteins into one MAG ID (optional; otherwise parsed from query names)",
    )
    parser.add_argument(
        "--counts_tsv",
        type=str,
        default=None,
        help="Optional path to save the intermediate OG count table (TSV)",
    )
    parser.add_argument("--separator", type=str, default="|", help="Separator in #query names")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample field index in #query")
    parser.add_argument("--model_input_size", type=int, default=2048, help="Max OG tokens")
    parser.add_argument("--no_special_token", action="store_true", help="Do not add <cls>/<eos>")
    args = parser.parse_args()

    emapper_path = Path(args.emapper)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Orthoformer — MAG preprocessing")
    print("=" * 60)
    print(f"emapper:    {emapper_path}")
    print(f"model_dir:  {model_dir}")
    print(f"output_dir: {output_dir}")
    print("=" * 60)

    vocab = load_vocab(model_dir)
    counts_df = emapper_to_counts(
        emapper_path,
        separator=args.separator,
        sample_idx=args.sample_idx,
        mag_id=args.mag_id,
    )
    n_samples = counts_df["Sample"].nunique() if not counts_df.empty else 0
    n_ogs = len(counts_df)
    print(f"Parsed {n_ogs} OG rows across {n_samples} genome(s)")

    if args.counts_tsv:
        counts_path = Path(args.counts_tsv)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(counts_path, sep="\t", index=False)
        print(f"Saved OG counts: {counts_path}")

    saved = counts_to_dataset(
        counts_df,
        vocab,
        output_dir,
        model_input_size=args.model_input_size,
        special_token=not args.no_special_token,
    )

    print(f"Saved dataset: {saved}")
    print()
    print("Next step:")
    print("  python biologist_quickstart.py \\")
    print(f"    --model_dir {model_dir} \\")
    print(f"    --dataset_path {saved} \\")
    print("    --output_dir outputs/my_mag_embeddings \\")
    print("    --use_alibi")


if __name__ == "__main__":
    main()
