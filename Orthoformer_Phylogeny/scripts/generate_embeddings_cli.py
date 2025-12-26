"""
CLI tool to generate embeddings with configurable parameters.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertForMaskedLM, BertModel, BertTokenizer


def load_model_with_alibi(model_dir: str, device: torch.device, use_alibi: bool = True) -> BertModel:
    """
    Load a BERT model and optionally inject ALiBi positional encoding layers.
    """
    if use_alibi:
        print("\n[Info] Loading model with ALiBi positional encoding support...")
        try:
            mlm_model = BertForMaskedLM.from_pretrained(model_dir)
            model = mlm_model.bert
            print("[Info] Successfully loaded BertForMaskedLM, extracted BERT base model")
        except Exception as exc:
            print(f"[Warn] Failed to load as BertForMaskedLM: {exc}")
            print("[Info] Falling back to BertModel...")
            model = BertModel.from_pretrained(model_dir)

        try:
            from pretrainer import CustomBertSelfAttention  # local module

            print("[Info] Applying ALiBi positional encoding to attention layers...")
            num_layers_replaced = 0
            for layer in model.encoder.layer:
                orig_sa = layer.attention.self
                if isinstance(orig_sa, CustomBertSelfAttention):
                    print(f"[Info] Layer {num_layers_replaced + 1} already uses CustomBertSelfAttention")
                else:
                    layer.attention.self = CustomBertSelfAttention(
                        orig_sa,
                        pos_kind="alibi",
                        max_position_embeddings=model.config.max_position_embeddings,
                    )
                num_layers_replaced += 1
            print(f"[Info] Successfully applied ALiBi to {num_layers_replaced} attention layers")
        except ImportError as exc:
            print(f"[Warn] Failed to import CustomBertSelfAttention: {exc}")
            print("[Warn] Model will use standard attention (ALiBi functionality disabled)")
        except Exception as exc:
            print(f"[Warn] Failed to apply ALiBi: {exc}")
            print("[Warn] Model will use standard attention (ALiBi functionality disabled)")
    else:
        print("\n[Info] Loading model with standard positional encoding...")
        model = BertModel.from_pretrained(model_dir)

    return model.eval().to(device)


def load_sampled_ids(sample_file: Optional[str]) -> Optional[set]:
    """Return a set of sample IDs if a file is provided."""
    if not sample_file:
        return None
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample list file not found: {sample_file}")
    with open(sample_file, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def pad_samples(samples, tokenizer, device: torch.device, model_max_length: int) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Pad or truncate sample sequences to a uniform length."""
    if "input_ids" in samples:
        max_length = min(max(len(seq) for seq in samples["input_ids"]), model_max_length)
        padded_input_ids = []
        for seq in samples["input_ids"]:
            if len(seq) > max_length:
                padded_seq = seq[:max_length]
            else:
                padded_seq = seq + [tokenizer.pad_token_id] * (max_length - len(seq))
            padded_input_ids.append(padded_seq)
        input_ids = torch.tensor(padded_input_ids).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    else:
        encoded = tokenizer(
            samples["tokens"],
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            max_length=model_max_length,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

    return input_ids, attention_mask, samples["sample_name"]


def mean_pool_embeddings(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean-pooled embeddings from model outputs."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


def save_embeddings(embeddings, sample_names: list, outdir: str, mode: str) -> None:
    """Persist embeddings as .npy files."""
    os.makedirs(outdir, exist_ok=True)
    for idx, emb in enumerate(embeddings):
        output_path = os.path.join(outdir, f"{sample_names[idx]}.npy")
        if mode == "mean":
            np.save(output_path, emb.detach().cpu().numpy())
        else:  # tokens
            np.save(output_path, emb)


def token_level_embeddings(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[np.ndarray]:
    """Return per-token embeddings (trimmed to each sample's true length)."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, H]
        lengths = attention_mask.sum(dim=1).tolist()
        per_sample = []
        for idx, length in enumerate(lengths):
            length = max(int(length), 1)  # avoid zero-length slices
            per_sample.append(hidden_states[idx, :length].detach().cpu().numpy())
        return per_sample


def process_batches(dataset, tokenizer, model, device, outdir: str, batch_size: int, model_max_length: int, output_mode: str) -> None:
    """Iterate over dataset batches and save embeddings."""
    from tqdm import tqdm

    processed = 0
    for start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[start : start + batch_size]
        input_ids, attention_mask, sample_names = pad_samples(batch, tokenizer, device, model_max_length)
        if output_mode == "mean":
            genome_embeddings = mean_pool_embeddings(model, input_ids, attention_mask)
            save_embeddings(genome_embeddings, sample_names, outdir, mode="mean")
        else:
            tokens_embeddings = token_level_embeddings(model, input_ids, attention_mask)
            save_embeddings(tokens_embeddings, sample_names, outdir, mode="tokens")
        processed += len(sample_names)
    print(f"\nProcessed {processed} samples. Embeddings stored in {outdir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings using configurable parameters.")
    parser.add_argument("--model_dir", required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--dataset_path", required=True, help="Path to the HuggingFace dataset directory.")
    parser.add_argument("--sample_list", help="Optional text file listing sample_name values to keep.")
    parser.add_argument("--output_dir", required=True, help="Directory to store generated embeddings.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--model_max_length", type=int, default=2048, help="Max sequence length for the model.")
    parser.add_argument("--device", default="cuda:0", help="Device specifier, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--use_alibi", action="store_true", help="Enable ALiBi positional encoding replacement.")
    parser.add_argument(
        "--output_mode",
        choices=["mean", "tokens"],
        default="mean",
        help="Choose 'mean' for sentence embeddings or 'tokens' for per-token matrices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("=" * 80)
    print("EMBEDDING GENERATION (CLI)")
    print("=" * 80)
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        gpu_idx = device.index or 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_idx)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1024 ** 3:.1f} GB")

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = load_model_with_alibi(args.model_dir, device, use_alibi=args.use_alibi)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,} ({total_params / 1_000_000:.2f}M)")

    dataset = load_from_disk(args.dataset_path)
    print(f"\nLoaded dataset with {len(dataset)} samples.")

    sampled_ids = load_sampled_ids(args.sample_list)
    if sampled_ids is not None:
        print(f"Filtering dataset to {len(sampled_ids)} IDs listed in {args.sample_list}")
        filtered_indices = [idx for idx, sample in enumerate(dataset) if sample["sample_name"] in sampled_ids]
        dataset = dataset.select(filtered_indices)
        print(f"Filtered dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("No samples to process after filtering. Exiting.")
        sys.exit(0)

    process_batches(
        dataset,
        tokenizer,
        model,
        device,
        args.output_dir,
        args.batch_size,
        args.model_max_length,
        args.output_mode,
    )
    print("\nFinished embedding generation.")


if __name__ == "__main__":
    main()

## Example:

# python scripts/generate_embeddings_cli.py \
#   --model_dir /mnt/np1/Foundation_Model/gene_bert_output/model_2048_v18/final_model \
#   --dataset_path /mnt/np1/Orthoformer_eval/combine_dataset \
#   --sample_list /mnt/np1/Orthoformer_eval/sampled_genomes.all.txt \
#   --output_dir /mnt/np1/Orthoformer_eval/embeddings/model_2048_v18_tokens \
#   --batch_size 16 \
#   --model_max_length 2048 \
#   --device cuda:0 \
#   --use_alibi \
#   --output_mode tokens  ## or mean