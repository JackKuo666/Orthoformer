#!/usr/bin/env python3
"""
Orthoformer quickstart for biologists.

Load a pretrained model, extract per-genome embedding vectors (mean pooling),
and save them as .npy files plus a summary CSV.

Example:
  python biologist_quickstart.py \\
    --model_dir model/model_140k_2048_v18 \\
    --dataset_path datasets/example \\
    --output_dir outputs/example_embeddings \\
    --use_alibi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import BertForMaskedLM, BertModel, BertTokenizer


def load_orthoformer(model_dir: str, use_alibi: bool, device: torch.device):
    """Load tokenizer and encoder; apply ALiBi if requested."""
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    if use_alibi:
        print("[1/4] Loading model with ALiBi positional encoding...")
        try:
            mlm_model = BertForMaskedLM.from_pretrained(model_dir)
            model = mlm_model.bert
        except Exception:
            model = BertModel.from_pretrained(model_dir)
        try:
            from orthoformer_model import OrthoformerSelfAttention

            for layer in model.encoder.layer:
                orig = layer.attention.self
                if not isinstance(orig, OrthoformerSelfAttention):
                    layer.attention.self = OrthoformerSelfAttention(
                        orig,
                        pos_kind="alibi",
                        max_position_embeddings=model.config.max_position_embeddings,
                    )
        except ImportError:
            print("  Warning: orthoformer_model not found; using standard attention.")
    else:
        print("[1/4] Loading model with standard positional encoding...")
        model = BertModel.from_pretrained(model_dir)

    model.to(device)
    model.eval()
    return tokenizer, model


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors over non-padding positions."""
    mask_exp = mask.unsqueeze(-1).expand(hidden.size()).float()
    summed = (hidden * mask_exp).sum(dim=1)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-9).float()
    return summed / counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract Orthoformer genome embeddings (biologist quickstart)."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Pretrained model folder")
    parser.add_argument("--dataset_path", type=str, required=True, help="HuggingFace dataset on disk")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write .npy embeddings")
    parser.add_argument("--use_alibi", action="store_true", help="Required for v8/v10/v18 models")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if omitted)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Orthoformer — Biologist Quickstart")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Model:      {args.model_dir}")
    print(f"Dataset:    {args.dataset_path}")
    print(f"Output:     {out_dir}")
    print("=" * 60)

    tokenizer, model = load_orthoformer(args.model_dir, args.use_alibi, device)
    hidden_size = model.config.hidden_size
    print(f"  Hidden size (embedding length): {hidden_size}")

    print("[2/4] Loading tokenized genomes...")
    dataset = load_from_disk(args.dataset_path)
    n = len(dataset)
    print(f"  Genomes in dataset: {n}")

    if "sample_name" in dataset.column_names:
        names = dataset["sample_name"]
    elif "Sample" in dataset.column_names:
        names = dataset["Sample"]
    else:
        names = [f"genome_{i}" for i in range(n)]

    print("[3/4] Extracting embeddings (mean pooling)...")
    summary_rows = []
    pad_id = tokenizer.pad_token_id

    for start in range(0, n, args.batch_size):
        batch = dataset[start : start + args.batch_size]
        ids_list = batch["input_ids"]
        max_len = min(max(len(x) for x in ids_list), model.config.max_position_embeddings)

        padded = []
        for seq in ids_list:
            seq = seq[:max_len]
            padded.append(seq + [pad_id] * (max_len - len(seq)))

        input_ids = torch.tensor(padded, device=device)
        attention_mask = (input_ids != pad_id).long()

        with torch.no_grad():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            emb = mean_pool(hidden, attention_mask)

        for i, vec in enumerate(emb.cpu().numpy()):
            idx = start + i
            sid = str(names[idx])
            safe_name = sid.replace("/", "_")
            np.save(out_dir / f"{safe_name}.npy", vec)
            summary_rows.append(
                {
                    "genome_id": sid,
                    "embedding_file": f"{safe_name}.npy",
                    "embedding_dim": hidden_size,
                    "sequence_length": int(attention_mask[i].sum().item()),
                }
            )
        done = min(start + args.batch_size, n)
        print(f"  Processed {done}/{n} genomes")

    print("[4/4] Writing summary table...")
    summary_path = out_dir / "embedding_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print()
    print("Done.")
    print(f"  Embeddings:  {out_dir}/*.npy  ({n} files)")
    print(f"  Summary:     {summary_path}")
    print()
    print("Next steps (see docs/Biologist_Tutorial.md or docs/MAG_Tutorial.md):")
    print("  • PCA/UMAP scatter plot for exploratory clustering")
    print("  • Orthoformer_Phylogeny/build_tree_from_embeddings.py for a phylogenetic tree")
    print("  • Orthoformer_Taxon/ for taxonomy inference")


if __name__ == "__main__":
    main()
