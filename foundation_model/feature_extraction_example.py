import os
import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM
from datasets import load_from_disk
from safetensors.torch import load_file

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def main(model_dir, dataset_path, use_alibi):
    # Load trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Check training configuration: if ALiBi was used, need to manually replace attention layers
    # According to run_training_v18.sh, training used --pos_encoding alibi
    if use_alibi:
        print("\n[Info] Loading model with ALiBi positional encoding support...")
        # First load BertForMaskedLM (complete model saved during training)
        try:
            mlm_model = BertForMaskedLM.from_pretrained(model_dir)
            model = mlm_model.bert
            print("[Info] Successfully loaded BertForMaskedLM, extracted BERT base model")
        except Exception as e:
            print(f"[Warn] Failed to load as BertForMaskedLM: {e}")
            print("[Info] Falling back to BertModel...")
            model = BertModel.from_pretrained(model_dir)
        
        # Manually apply ALiBi positional encoding
        try:
            from orthoformer_model import OrthoformerSelfAttention
            print("[Info] Applying ALiBi positional encoding to attention layers...")
            num_layers_replaced = 0
            for layer in model.encoder.layer:
                orig_sa = layer.attention.self
                # Check if already OrthoformerSelfAttention
                if isinstance(orig_sa, OrthoformerSelfAttention):
                    print(f"[Info] Layer {num_layers_replaced + 1} already has OrthoformerSelfAttention")
                else:
                    layer.attention.self = OrthoformerSelfAttention(
                        orig_sa,
                        pos_kind="alibi",
                        max_position_embeddings=model.config.max_position_embeddings
                    )
                num_layers_replaced += 1
            print(f"[Info] Successfully applied ALiBi to {num_layers_replaced} attention layers")
        except ImportError as e:
            print(f"[Warn] Failed to import OrthoformerSelfAttention: {e}")
            print("[Warn] Model will use standard attention (ALiBi functionality disabled)")
        except Exception as e:
            print(f"[Warn] Failed to apply ALiBi: {e}")
            print("[Warn] Model will use standard attention (ALiBi functionality disabled)")
    else:
        print("\n[Info] Loading model with standard positional encoding...")
        model = BertModel.from_pretrained(model_dir)

    model.to(device)  # Move model to GPU
    model.eval()

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Params: {total_params:,} ({total_params/1_000_000:.2f}M)")
    print(f"Trainable Model Params: {trainable_params:,} ({trainable_params/1_000_000:.2f}M)")

    # Print model architecture information
    print("\n" + "="*50)
    print("Orthoformer Model Architecture:")
    print("="*50)
    print(f"Model Type: {type(model).__name__}")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Number of Layers: {model.config.num_hidden_layers}")
    print(f"Number of Attention Heads: {model.config.num_attention_heads}")
    print(f"Intermediate Size: {model.config.intermediate_size}")
    print(f"Max Position Embeddings: {model.config.max_position_embeddings}")
    print(f"Vocabulary Size: {model.config.vocab_size}")
    print(f"Hidden Dropout: {model.config.hidden_dropout_prob}")
    print(f"Attention Dropout: {model.config.attention_probs_dropout_prob}")

    # Print embedding layer information
    print("\nEmbedding Layers:")
    print("-" * 30)
    print(f"Word Embeddings: {model.embeddings.word_embeddings.weight.shape}")
    print(f"Position Embeddings: {model.embeddings.position_embeddings.weight.shape}")
    print(f"Token Type Embeddings: {model.embeddings.token_type_embeddings.weight.shape}")
    print(f"Layer Norm: {model.embeddings.LayerNorm}")

    # Print transformer layer information
    print(f"\nTransformer Layers: {model.config.num_hidden_layers}")
    print("-" * 30)
    for i, layer in enumerate(model.encoder.layer):
        print(f"Layer {i+1}:")
        print(f"  - Attention: {layer.attention}")
        print(f"  - Intermediate: {layer.intermediate}")
        print(f"  - Output: {layer.output}")

    # Print pooler layer information
    print(f"\nPooler:")
    print("-" * 30)
    print(f"Pooler: {model.pooler}")
    print("="*50)

    # Manually calculate parameter count
    print("\n" + "="*50)
    print("Parameter Calculation Breakdown:")
    print("="*50)

    config = model.config
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    max_position_embeddings = config.max_position_embeddings

    # 1. Embedding layer parameters
    word_embeddings = vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    # Note: From output, Token Type Embeddings is actually [1, 512], not [2, 512]
    token_type_embeddings = 1 * hidden_size  # Actually only 1 token type
    embedding_layer_norm = hidden_size * 2  # weight + bias
    embedding_total = word_embeddings + position_embeddings + token_type_embeddings + embedding_layer_norm

    print(f"1. Embedding Layers:")
    print(f"   Word Embeddings: {vocab_size:,} × {hidden_size} = {word_embeddings:,}")
    print(f"   Position Embeddings: {max_position_embeddings:,} × {hidden_size} = {position_embeddings:,}")
    print(f"   Token Type Embeddings: 1 × {hidden_size} = {token_type_embeddings:,}")
    print(f"   Layer Norm: {hidden_size} × 2 = {embedding_layer_norm:,}")
    print(f"   Embedding Total: {embedding_total:,}")

    # 2. Parameters for each Transformer layer
    # Attention layer
    attention_query = hidden_size * hidden_size  # Q matrix
    attention_key = hidden_size * hidden_size    # K matrix  
    attention_value = hidden_size * hidden_size  # V matrix
    attention_output = hidden_size * hidden_size # Output projection
    attention_bias = hidden_size * 4  # Bias for Q, K, V, output
    attention_total = attention_query + attention_key + attention_value + attention_output + attention_bias

    # Feed Forward layer
    ff_intermediate = hidden_size * intermediate_size  # First linear layer
    ff_output = intermediate_size * hidden_size        # Second linear layer
    ff_bias = intermediate_size + hidden_size          # Two biases
    ff_total = ff_intermediate + ff_output + ff_bias

    # Layer Norm (2 layers)
    layer_norm_params = hidden_size * 2 * 2  # 2 layer norms, each with weight and bias

    # Total parameters for a single Transformer layer
    single_layer_total = attention_total + ff_total + layer_norm_params
    all_layers_total = single_layer_total * num_layers

    print(f"\n2. Transformer Layers ({num_layers} layers):")
    print(f"   Per Layer:")
    print(f"     Attention:")
    print(f"       Query: {hidden_size} × {hidden_size} = {attention_query:,}")
    print(f"       Key: {hidden_size} × {hidden_size} = {attention_key:,}")
    print(f"       Value: {hidden_size} × {hidden_size} = {attention_value:,}")
    print(f"       Output: {hidden_size} × {hidden_size} = {attention_output:,}")
    print(f"       Bias: {hidden_size} × 4 = {attention_bias:,}")
    print(f"       Attention Total: {attention_total:,}")
    print(f"     Feed Forward:")
    print(f"       Intermediate: {hidden_size} × {intermediate_size} = {ff_intermediate:,}")
    print(f"       Output: {intermediate_size} × {hidden_size} = {ff_output:,}")
    print(f"       Bias: {intermediate_size} + {hidden_size} = {ff_bias:,}")
    print(f"       FF Total: {ff_total:,}")
    print(f"     Layer Norms: {hidden_size} × 2 × 2 = {layer_norm_params:,}")
    print(f"   Single Layer Total: {single_layer_total:,}")
    print(f"   All Layers Total: {single_layer_total:,} × {num_layers} = {all_layers_total:,}")

    # 3. Pooler layer parameters
    pooler_dense = hidden_size * hidden_size
    pooler_bias = hidden_size
    pooler_total = pooler_dense + pooler_bias

    print(f"\n3. Pooler Layer:")
    print(f"   Dense: {hidden_size} × {hidden_size} = {pooler_dense:,}")
    print(f"   Bias: {hidden_size} = {pooler_bias:,}")
    print(f"   Pooler Total: {pooler_total:,}")

    # Total calculation
    calculated_total = embedding_total + all_layers_total + pooler_total
    print(f"\n4. Total Calculated: {calculated_total:,} ({calculated_total/1_000_000:.2f}M)")
    print(f"   Actual Model Total: {total_params:,} ({total_params/1_000_000:.2f}M)")
    print(f"   Difference: {abs(calculated_total - total_params):,}")
    print("="*50)

    # Load training dataset
    dataset_path = "datasets/example"
    dataset = load_from_disk(dataset_path)

    # Take first two samples
    samples = dataset[:2]  # Take first two rows

    # Handle inconsistent sequence lengths
    if "input_ids" in samples:
        # Get maximum sequence length
        max_length = max(len(seq) for seq in samples["input_ids"])

        model_max_length = model.config.max_length
        # Limit maximum length to avoid memory issues
        max_length = min(max_length, model_max_length)
        
        # Pad or truncate sequences
        padded_input_ids = []
        for seq in samples["input_ids"]:
            if len(seq) > max_length:
                # Truncate
                padded_seq = seq[:max_length]
            else:
                # Pad
                padded_seq = seq + [tokenizer.pad_token_id] * (max_length - len(seq))
            padded_input_ids.append(padded_seq)
        
        input_ids = torch.tensor(padded_input_ids).to(device)  # Move to GPU
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    else:
        # Assume 'tokens' field exists
        inputs = tokenizer(
            samples["tokens"],
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            max_length=4000,
            truncation=True
        )
        input_ids = inputs["input_ids"].to(device)  # Move to GPU
        attention_mask = inputs["attention_mask"].to(device)  # Move to GPU

    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Forward inference to get embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Method 1: CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)
        print("\n[CLS] embedding shape:", cls_embeddings.shape)
        print("[CLS] embedding for first sample:")
        print(cls_embeddings[0][:10].cpu())  # Move to CPU for display
        
        # Method 2: Mean pooling embedding
        # last_hidden_state: (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        # Expand attention_mask to hidden_size dimension
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        # Sum over valid tokens
        sum_hidden = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        # Count valid tokens
        sum_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # Calculate mean
        mean_embeddings = sum_hidden / sum_mask
        
        print("\nMean pooling embedding shape:", mean_embeddings.shape)
        print("Mean pooling embedding for first sample:")
        print(mean_embeddings[0][:10].cpu())  # Move to CPU for display

        # Method 3: Attention pooling embedding (using trained classification head attention vector)
        attn_head_path = os.path.join(model_dir, "classification_head", "model.safetensors")
        if os.path.exists(attn_head_path):
            try:
                state = load_file(attn_head_path)
                if "attn_vec" in state:
                    attn_vec = state["attn_vec"].to(device)
                    # Normalize attention vector
                    a = attn_vec / (attn_vec.norm() + 1e-6)
                    # Calculate scores for each token and apply mask
                    scores = torch.einsum("blh,h->bl", hidden_states, a)
                    scores = scores.masked_fill(attention_mask == 0, float("-inf"))
                    weights = torch.softmax(scores, dim=1)
                    # Weighted sum to get sentence vector
                    attn_embeddings = torch.einsum("bl,blh->bh", weights, hidden_states)
                    print("\nAttention pooling embedding shape:", attn_embeddings.shape)
                    print("Attention pooling embedding for first sample:")
                    print(attn_embeddings[0][:10].cpu())

                    # Optional: Apply L2 normalization to sentence vectors for more stable and comparable retrieval/clustering
                    attn_embeddings_norm = torch.nn.functional.normalize(attn_embeddings, dim=-1)
                    print("\n[Info] Applied L2 normalization to attention embeddings (recommended for retrieval/clustering).")
                    print("Normalized attention embedding for first sample:")
                    print(attn_embeddings_norm[0][:10].cpu())

                    # Brief comparison and usage recommendations for the two sentence vector construction methods
                    print("\n[Guide] How to choose sentence embedding:")
                    print("  - Attention pooling (recommended): focuses on informative tokens; better for long/sparse signals and downstream classification/retrieval.")
                    print("  - Mean pooling: simple and stable baseline; good as a neutral representation and for sanity checks.")
                    print("  - Tip: For retrieval/clustering, prefer L2-normalized attention embeddings.")
                    print("  - L2 on: retrieval, clustering, visualization, cross-batch comparison.")
                    print("  - L2 off: feeding a trainable linear/MLP head that uses raw magnitudes, or when vector norm encodes confidence/strength for thresholding/ranking.")
                else:
                    print("\n[Warn] attn_vec not found in classification head weights, skipping attention pooling example.")
            except Exception as e:
                print(f"\n[Warn] Failed to load classification head attention vector: {e}")
        else:
            print("\n[Info] Classification head weight file not found, path: ", attn_head_path)

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Feature Extraction Example")
    parser.add_argument("--model_dir", type=str, default="model/model_3M_2048_v8", help="Model directory")
    parser.add_argument("--dataset_path", type=str, default="datasets/example", help="Dataset directory")
    parser.add_argument("--use_alibi", action="store_true", help="Use ALiBi positional encoding (default: auto-detect from model_dir, v8=True, v5=False)")
    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()
    model_dir = args.model_dir
    dataset_path = args.dataset_path
    # Auto-detect use_alibi from model_dir if not explicitly set
    if args.use_alibi:
        use_alibi = True
    else:
        # Default: v8 uses ALiBi, v5 uses standard encoding
        use_alibi = "v8" in model_dir
    main(model_dir, dataset_path, use_alibi)