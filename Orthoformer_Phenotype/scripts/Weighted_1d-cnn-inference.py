import torch
import pandas as pd
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,precision_score,balanced_accuracy_score,recall_score

from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from typing import List, Optional, Tuple
from transformers import BertForMaskedLM, BertModel, BertTokenizer

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch.nn.functional as F

def collate_fn_test(batch):
    """
    batch: list of (X, y, length, og_seq) from TestSeqDataset.__getitem__
    Goals:
      - X, y, length normally stack into tensors
      - og_seq keeps a list of "one sequence per sample", no zip transpose
    """
    Xs, ys, lens, ogs,names = zip(*batch)  # length = batch_size

    Xs = torch.stack(Xs, dim=0)             # [B, MAX_LEN, D]
    ys = torch.stack(ys, dim=0)             # [B]
    lens = torch.stack(lens, dim=0)         # [B]
    ogs = list(ogs)                         # [B], each element is one OG sequence (list or tuple)
    names = list(names)
    return Xs, ys, lens, ogs , names

# ==================== Loss: FocalLoss ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits input.
    logits: [B]
    targets: [B], float 0/1
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B], targets: [B]
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # [B]

        # p_t = p if y=1 else (1-p)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # [B]

        focal_weight = self.alpha * (1 - p_t) ** self.gamma  # [B]
        loss = focal_weight * bce_loss                       # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# ==================== Global Parameters ====================

D = 1024
MAX_LEN = 2048

# ==================== Padding Function ====================

def padding_to_maxlen(sequences, labels, MAX_LEN=2048):
    """
    sequences: list of np.array, each shape = (length_i, D)
    labels: list or array, len = N
    Returns:
       X_padded: [N, MAX_LEN, D]
       y:        [N]
       lengths:  [N] valid lengths
    """
    N = len(sequences)
    X_padded = np.zeros((N, MAX_LEN, D), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)

    for i, seq in enumerate(sequences):
        L = seq.shape[0]
        eff_L = min(L, MAX_LEN)
        lengths[i] = eff_L
        if L >= MAX_LEN:
            X_padded[i] = seq[:MAX_LEN, :] ###
        else:
            X_padded[i, :L, :] = seq
    # Don't modify label
    #labels[-1] = 1 ### for test
    y = np.asarray(labels).astype(np.int64)
    return X_padded, y, lengths

# ==================== Dataset ====================

class SeqDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.from_numpy(X)          # [N, MAX_LEN, D]
        self.y = torch.from_numpy(y)          # [N]
        self.lengths = torch.from_numpy(lengths)  # [N]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

class TestSeqDataset(Dataset):
    def __init__(self, X, y, lengths, og_seqs,sample_names):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.lengths = torch.from_numpy(lengths)
        self.og_seqs = og_seqs  # list of OG lists
        self.sample_names = sample_names
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx], self.og_seqs[idx] , self.sample_names[idx]

# ==================== 1D CNN Model ====================

class CNN1DClassifier(nn.Module):
    def __init__(self, in_dim=1024, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 256, kernel_size=1, padding=0)
        self.bn1   = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, padding=0)
        self.bn2   = nn.BatchNorm1d(256)
        self.relu  = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # -> [B, C, 1]
        self.fc    = nn.Linear(256, num_classes)

    def forward_features(self, x):
        # x: [B, L, D]
        x = x.permute(0, 2, 1)  # -> [B, D, L]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        # [B, 256, L]
        return x

    def forward(self, x):
        x = self.forward_features(x)   # [B, 256, L]
        x = self.global_pool(x)        # [B, 256, 1]
        x = x.squeeze(-1)              # [B, 256]
        logits = self.fc(x).squeeze(-1)  # [B]
        return logits                  # BCEWithLogits / FocalLoss

# ==================== CSV Load Labels ====================

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    label_counts = df['Label'].value_counts().to_dict()
    positive_ratio = float(df['Label'].mean())

    print("Basic data information:")
    print(f"  Total samples: {total_samples}")
    print(f"  Class distribution: {label_counts}")
    print(f"  Positive ratio: {positive_ratio:.3f}")

    label_dict = df.set_index('genome_name')['Label'].to_dict()
    return label_dict

# ==================== Class for Reading .pt Files ====================

class COGAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.data = []
        self.sample_names = []
        self.labels = []
        self.OG_seqs = []

    def load_pt_files_with_samples(self, samples_list):
        pt_files = list(self.folder_path.glob("*.pt"))
        print(f"Found {len(pt_files)} pt files")
        filter_pt_files = [filename for filename in pt_files if filename.stem in samples_list.keys()]

        sample_count = 0
        for pt_file in filter_pt_files:
            try:
                file_data = torch.load(pt_file, weights_only=False)
                sample_name = pt_file.stem
                og_seq = file_data["OG_seq"]          # list of OG IDs
                embedding = file_data["embedding"]    # tensor [length_i, D]

                self.OG_seqs.append(og_seq)
                self.data.append(
                    embedding.numpy() if torch.is_tensor(embedding) else embedding
                )
                self.sample_names.append(sample_name)
                self.labels.append(samples_list[sample_name])
                sample_count += 1
                #print(f"Successfully loaded: {sample_name}, contains {len(og_seq)} COGs")
            except Exception as e:
                print(f"Error loading file {pt_file}: {e}")
            #if sample_count >= 40:
            #    break

# ==================== Saliency Computation (with OG Mapping) ====================

def compute_saliency_and_map_to_sequences(model, dataloader, device, max_batches=None):
    model.eval()

    all_saliency_trimmed = []
    all_lengths = []
    all_original_sequences = []
    all_sample_names = []

    for b_idx, (X_batch, y_batch, len_batch, og_seq_batch, name_batch) in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        X_batch = X_batch.to(device)           # [B, MAX_LEN, D]
        len_batch = len_batch.to(device)       # [B]
        X_batch.requires_grad_(True)

        logits = model(X_batch)                # [B]
        loss = logits.sum()
        model.zero_grad()
        loss.backward()

        grads = X_batch.grad.detach()          # [B, MAX_LEN, D]
        inputs = X_batch.detach()
        saliency = (grads * inputs).abs().sum(dim=2)  # [B, MAX_LEN]

        saliency_np = saliency.cpu().numpy()
        len_np = len_batch.cpu().numpy()

        B = saliency_np.shape[0]
        for i in range(B):
            L_i = int(len_np[i])
            sal_map_i = saliency_np[i, :L_i].copy()

            og_seq_i = og_seq_batch[i]
            if isinstance(og_seq_i, tuple):
                og_seq_i = list(og_seq_i)

            if len(og_seq_i) != L_i:
                print(f"[WARN] len(og_seq)={len(og_seq_i)} != L_i={L_i}, will crop to minimum length")
                L_eff = min(len(og_seq_i), L_i)
                og_seq_i = og_seq_i[:L_eff]
                sal_map_i = sal_map_i[:L_eff]
                L_i = L_eff

            sample_name_i = name_batch[i]

            all_saliency_trimmed.append(sal_map_i)
            all_lengths.append(L_i)
            all_original_sequences.append(og_seq_i)
            all_sample_names.append(sample_name_i)

    return all_saliency_trimmed, all_lengths, all_original_sequences, all_sample_names

# ==================== Find Top OG for Each Sample from Saliency ====================
def get_important_OG(saliency_list, original_seq_list, sample_name_list, top_k=10):
    """
    Find top_k most contributing OGs based on each sample's saliency.

    Parameters
    ----
    saliency_list : list of np.ndarray
        Saliency vector for each sample (length = number of valid positions).
    original_seq_list : list of list[str]
        OG sequence corresponding to each sample (aligned with saliency length).
    sample_name_list : list of str
        Name of each sample (for output labeling).
    top_k : int
        Top K positions to output for each sample (sorted by saliency descending).

    Returns
    ----
    all_top_positions : list[list[int]]
        List of top_k position indices for each sample (sorted by saliency descending).
    all_top_scores : list[list[float]]
        List of top_k saliency scores for each sample.
    all_top_ogs : list[list[str]]
        List of top_k OG names for each sample.
    records : list[dict]
        Expanded records for saving to CSV, each row is (sample_name, rank, position, score, og, context_ogs).
    """
    all_top_positions = []
    all_top_scores = []
    all_top_ogs = []
    records = []

    for sal_i, og_list, name in zip(saliency_list, original_seq_list, sample_name_list):
        sal_i = np.asarray(sal_i)
        L_sal = len(sal_i)
        L_og = len(og_list)
        L_eff = min(L_sal, L_og)

        if L_eff == 0:
            print(f"Sample {name}: empty effective length, skip.")
            continue

        sal_eff = sal_i[:L_eff]
        og_eff = og_list[:L_eff]

        # Take top_k positions, sorted by saliency descending
        k = min(top_k, L_eff)
        sorted_idx = np.argsort(-sal_eff)[:k]  # descending order

        sample_positions = []
        sample_scores = []
        sample_ogs = []

        for rank, pos in enumerate(sorted_idx, start=1):
            pos = int(pos)
            score = float(sal_eff[pos])
            og = og_eff[pos]

            window = 2
            start = max(0, pos - window)
            end = min(L_eff, pos + window + 1)
            fragment = og_eff[start:end]

            # Only print top1 for each sample to avoid log explosion
            if rank == 1:
                print(f"Sample {name}:")
                print(f"  Top position: {pos}")
                print(f"  Top OG:       {og}")
                print(f"  Saliency:     {score:.6f}")
                print(f"  Fragment:     {fragment}")
                print("-" * 60)

            sample_positions.append(pos)
            sample_scores.append(score)
            sample_ogs.append(og)

            records.append({
                "sample_name": name,
                "rank": rank,
                "position": pos,
                "saliency_score": score,
                "L_eff": L_eff,
                "og": og,
                "context_ogs": ";".join(fragment),
            })

        all_top_positions.append(sample_positions)
        all_top_scores.append(sample_scores)
        all_top_ogs.append(sample_ogs)

    return all_top_positions, all_top_scores, all_top_ogs, records


# ==================== Visualization Functions ====================

def plot_sequence_saliency_heatmap(
    sequence,
    saliency,
    max_len=None,
    step=None,
    figsize=(12, 2),
    cmap="RdBu_r",
    title=None,
    show_colorbar=True
):
    """
    sequence: list(str) or str, length L (here is OG sequence)
    saliency: 1D array-like, length L
    """
    if isinstance(sequence, str):
        seq_list = list(sequence)
    else:
        seq_list = list(sequence)

    saliency = np.asarray(saliency, dtype=float)

    assert len(seq_list) == len(saliency), \
        f"sequence length {len(seq_list)} != saliency length {len(saliency)}"

    L = len(seq_list)

    if max_len is not None and L > max_len:
        L = max_len
        seq_list = seq_list[:L]
        saliency = saliency[:L]

    if np.all(saliency == 0):
        norm_sal = np.zeros_like(saliency)
    else:
        min_v = saliency.min()
        max_v = saliency.max()
        if max_v == min_v:
            norm_sal = np.zeros_like(saliency)
        else:
            norm_sal = (saliency - min_v) / (max_v - min_v)

    heat = norm_sal[np.newaxis, :]  # [1, L]

    if step is None:
        if L <= 50:
            step = 1
        elif L <= 200:
            step = 5
        else:
            step = 10

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        heat,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=1.0)
    )

    ax.set_xticks(np.arange(0, L))
    #ax.set_xticklabels([f"{seq_list[i]}\n{i}" for i in range(0, L, step)], fontsize=6)
    
    saliency_sorted = np.argsort(saliency)[::-1]
    show_positions = saliency_sorted[0:10]

    labels = []
    for i in range(L):
        if i in show_positions:
            labels.append(f"{seq_list[i]}\n{i}")
        else:
            labels.append("")

    ax.set_xticklabels(labels, fontsize=6)
    ax.tick_params(bottom=False)
    ax.set_yticks([])
    ax.set_ylabel("Saliency", fontsize=10)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, 0.5)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Normalized saliency", fontsize=10)

    plt.tight_layout()
    return fig, ax

def show_ith_sample_bar(sample_name_list,saliency_list, original_seq_list, i=0):
    sal_i = saliency_list[i]
    seq_i = original_seq_list[i]
    fig, ax = plot_sequence_saliency_heatmap(
        sequence=seq_i,
        saliency=sal_i,
        max_len=len(sal_i),
        title=f"{sample_name_list[i]} saliency",
    )
    plt.savefig('saliency_heatmap.pdf', dpi=300, bbox_inches='tight')

# ==================== Main Training Function ====================

def train_1dcnn(args):

    task_path = Path(args.label)

    genome2label_dict = load_and_preprocess_data(args.label)
    analyzer = COGAnalyzer(args.embedding_path)
    analyzer.load_pt_files_with_samples(genome2label_dict)

    X_padded, y, lengths = padding_to_maxlen(analyzer.data, analyzer.labels, MAX_LEN=args.model_max_length)
    
    X_train, X_test, y_train, y_test, len_train, len_test, OG_train, OG_test, name_train, name_test = train_test_split(
    X_padded, y, lengths, analyzer.OG_seqs, analyzer.sample_names,test_size=0.2, random_state=42, stratify=y
    )


    train_ds = SeqDataset(X_train, y_train, len_train)
    test_ds  = TestSeqDataset(X_test,  y_test,  len_test, OG_test,name_test)

    ### previous sample
    #train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    # ===== Use WeightedRandomSampler for class-balanced sampling =====
    y_train_np = y_train.astype(np.int64)
    class_sample_count = np.bincount(y_train_np)
    print("Train class count:", class_sample_count)
    if class_sample_count.size < 2:
        # Extreme case: only one class in training set, fallback to normal shuffle
        print("Warning: Only one class in training set, falling back to normal random sampling.")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    else:
        num_neg, num_pos = class_sample_count[0], class_sample_count[1]
        # Give minority class larger sampling weight
        weights = np.where(y_train_np == 1, num_neg / max(num_pos, 1), 1.0).astype(np.float64)
        weights = torch.from_numpy(weights)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)

    # ===== Build test DataLoader using custom collate_fn_test function =====
    test_loader  = DataLoader(test_ds,  batch_size=2, shuffle=False, collate_fn=collate_fn_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DClassifier(in_dim=D).to(device)

    # ===== Loss: Use FocalLoss (imbalance adaptation) =====
    pos_frac = y.mean()
    alpha = 1.0 - pos_frac
    print("Positive fraction:", pos_frac, " -> alpha for focal loss:", alpha)
    criterion = FocalLoss(alpha=float(alpha), gamma=2.0, reduction="mean")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30

    # ====== Early stopping & best checkpoint ======
    best_auc = -1.0
    best_epoch = -1
    best_state_dict = None

    patience = 5      # Number of consecutive epochs without improvement allowed
    min_delta = 1e-3  # AUC must improve by at least this much to be considered "progress"
    no_improve_count = 0
    # ==============================================

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch, len_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch, len_batch, _ , _ in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                all_logits.append(logits.cpu())
                all_labels.append(y_batch.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        probs = torch.sigmoid(all_logits).numpy()

        # Accuracy at default 0.5 threshold
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(all_labels.numpy(), preds)

        # AUC does not depend on threshold
        weighted_f1  = None
        try:
            auc = roc_auc_score(all_labels.numpy(), probs)
            weighted_f1 = f1_score(all_labels.numpy(), preds, average='weighted')
        except ValueError:
            auc = None

        # ===== Scan threshold on validation set, compute balanced accuracy to mitigate extreme imbalance =====
        y_true = all_labels.numpy()
        best_thr = 0.5
        best_bal_acc = None

        if np.unique(y_true).size > 1:
            thresholds = np.linspace(0.05, 0.95, 19)
            best_bal_acc = -1.0
            for thr in thresholds:
                preds_thr = (probs >= thr).astype(int)
                bal_acc = balanced_accuracy_score(y_true, preds_thr)
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_thr = thr
        
        # Log output: provide AUC, acc@0.5 and balanced accuracy at best threshold
        if auc is not None:
            if best_bal_acc is not None and best_bal_acc >= 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Test loss: {avg_loss:.4f}, "
                    f"acc@0.5: {acc:.4f}, auc: {auc:.4f}, best_thr: {best_thr:.2f}, "
                    f"bal_acc: {best_bal_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Test loss: {avg_loss:.4f}, "
                    f"acc@0.5: {acc:.4f}, auc: {auc:.4f}"
                )
        else:
            if best_bal_acc is not None and best_bal_acc >= 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Test loss: {avg_loss:.4f}, "
                    f"acc@0.5: {acc:.4f}, best_thr: {best_thr:.2f}, "
                    f"bal_acc: {best_bal_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Test loss: {avg_loss:.4f}, "
                    f"acc@0.5: {acc:.4f}"
                )
        # ----------- Early stopping & checkpoint -----------
        metric = auc if auc is not None else acc
        if metric is None:
            continue
        
        if metric > best_auc + min_delta:
            # Significant improvement: update best
            best_auc = metric
            best_epoch = epoch + 1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
            print(f"  -> New best model at epoch {epoch+1}, metric={metric:.4f}")
        else:
            # No improvement: increment count
            no_improve_count += 1
            print(f"  -> No improvement for {no_improve_count} epoch(s) (best_auc={best_auc:.4f})")
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch}, best_auc={best_auc:.4f}")
                break
    
    # ===== Use best checkpoint to overwrite model parameters =====
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch} with AUC={best_auc:.4f}")
    else:
        print("Warning: best_state_dict is None, using last epoch model.")
    
    # Optional: save model to disk
    Path("cnn_models").mkdir(parents=True, exist_ok=True)
    save_path = f"cnn_models/{task_path.stem}_best_cnn1d_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "in_dim": D,
        "max_len": MAX_LEN,
        "best_weighted_f1": weighted_f1,
    }, save_path)
    print(f"Best model saved to {save_path}")

    # ===== After training, compute saliency + OG mapping for each sample =====
    saliency_list, length_list, original_seq_list, sample_name_list = compute_saliency_and_map_to_sequences(
        model, test_loader, device
    )

    # Simple check
    for i, saliency in enumerate(saliency_list[:5]):
        print(f"[Check] Sample {sample_name_list[i]}: saliency_len={len(saliency)}, "
              f"length={length_list[i]}, OG_len={len(original_seq_list[i])}")

    # Output top-K OG for each sample (K controlled by --top_k parameter)
    top_positions, top_scores, top_ogs, records = get_important_OG(
        saliency_list, original_seq_list, sample_name_list, top_k=args.top_k
    )

    # Save top-K results for all samples as CSV for subsequent analysis
    if len(records) > 0:
        df_top = pd.DataFrame.from_records(records)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{task_path.stem}_top{args.top_k}_salient_OGs.csv"
        df_top.to_csv(csv_path, index=False)
        print(f"Saved top-{args.top_k} OG saliency results to {csv_path}")
    else:
        print("Warning: no saliency records to save (empty records list).")

    # If you want to see heatmap for a specific sample:
    # show_ith_sample_bar(saliency_list, original_seq_list, i=0)
    
    model.eval()
    all_test_names = []
    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []

    with torch.no_grad():
        for X_batch, y_batch, len_batch, og_batch, name_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            #preds = (probs >= best_thr).astype(int)

            all_test_names.extend(name_batch)
            all_true_labels.extend(y_batch.cpu().numpy())
            all_pred_probs.extend(probs)
            all_pred_labels.extend(preds)

    # Create prediction results DataFrame
    test_results_df = pd.DataFrame({
        'sample_name': all_test_names,
        'true_label': all_true_labels,
        'predicted_prob': all_pred_probs,
        'predicted_label': all_pred_labels
    })

    # Compute evaluation metrics
    try:
        auc = roc_auc_score(all_true_labels, all_pred_probs)
        weighted_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)

        # Add evaluation metrics to CSV header or save separately
        metrics_summary = {
            'AUC': auc,
            'Weighted_F1': weighted_f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        }

        print("\n" + "="*50)
        print("Model Evaluation Metrics:")
        print("="*50)
        for metric, value in metrics_summary.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)

    except Exception as e:
        print(f"Error computing evaluation metrics: {e}")
        metrics_summary = {}

    test_results_path = output_dir / f"{task_path.stem}_test_predictions.csv"
    test_results_df.to_csv(test_results_path, index=False)
    print(f"Saved test set predictions to {test_results_path}")

    # 3. Save evaluation metrics (optional)
    if metrics_summary:
        metrics_df = pd.DataFrame([metrics_summary])
        metrics_path = output_dir / f"{task_path.stem}_evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved evaluation metrics to {metrics_path}")

# ==================== Argument Parsing & main ====================

def pred_1dcnn(args):
    ### 
    genome2label_dict = load_and_preprocess_data(args.label)
    analyzer = COGAnalyzer(args.embedding_path)
    analyzer.load_pt_files_with_samples(genome2label_dict)
    
    X_padded, y, lengths = padding_to_maxlen(analyzer.data, analyzer.labels, MAX_LEN=args.model_max_length)
    
    #X_train, X_test, y_train, y_test, len_train, len_test, OG_train, OG_test, name_train, name_test = train_test_split(X_padded, y, lengths, analyzer.OG_seqs, analyzer.sample_names,test_size=1, random_state=42, stratify=y)
    X_test,y_test,len_test,OG_test,name_test = X_padded, y, lengths, analyzer.OG_seqs, analyzer.sample_names
    ### all data is test data
    ### only need data for testing 
    test_ds  = TestSeqDataset(X_test,  y_test,  len_test, OG_test,name_test)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn_test)

    ### load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DClassifier(in_dim=D).to(device)
    
    ### load pt
    parameters = torch.load(args.model_path)
    model.load_state_dict(parameters["model_state_dict"])
    
    
    saliency_list, length_list, original_seq_list, sample_name_list = compute_saliency_and_map_to_sequences(
        model, test_loader, device
    )

    # Simple check
    #for i, saliency in enumerate(saliency_list[:5]):
    #    print(f"[Check] Sample {sample_name_list[i]}: saliency_len={len(saliency)}, "
    #          f"length={length_list[i]}, OG_len={len(original_seq_list[i])}")
    
    # Output top-K OG for each sample (K controlled by --top_k parameter)
    top_positions, top_scores, top_ogs, records = get_important_OG(
        saliency_list, original_seq_list, sample_name_list, top_k=args.top_k
    )
    
    show_ith_sample_bar(sample_name_list,saliency_list, original_seq_list, i=0)
    
    print(f"{sample_name_list[0]}_saliency_score.txt")
    with open(f"{sample_name_list[0]}_saliency_score.txt",'w') as f:
        f.write(f"Rank,saliency_score,OG\n")
        for i,og in enumerate(saliency_list[0]):
            f.write(f"{i},{saliency_list[0][i]},{original_seq_list[0][i]}\n")

def parse_argv():
    parser = argparse.ArgumentParser(description="dealing with OG embedding")
    parser.add_argument("--model_path", type=str, default="/mnt/MAG/jinfang/deaminase_perturbation_pipelines/model_3M_2048_v10")
    parser.add_argument("--dataset_path", type=str, default="/mnt/MAG/jinfang/phenotype/COG3209.dataset")
    parser.add_argument("--output_dir", type=str, default="1dcnn_results")
    parser.add_argument("--embedding_path", type=str, default="/mnt/MAG/jinfang/phenotype/all_phentypes.samples.model_3M_2048_v10.embedding")
    parser.add_argument("--og", type=str, default="COG3209")
    parser.add_argument("--label", type=str, default="/mnt/MAG/jinfang/phenotype/classified_output/binary/madin_carbsubs_Tween_20.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top OGs per sample to output based on saliency"
    )
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--use_alibi", action="store_true", help="Enable ALiBi positional encoding replacement.")
    parser.add_argument("--sample_list", default="/mnt/MAG/jinfang/phenotype/classified_output/binary/carbsubs_genomeIDs/madin_carbsubs_Tween_20.genomeID",help="Optional text file listing sample_name values to keep.")
    parser.add_argument("funtion")
    args = parser.parse_args()
    return args

def main():
    args = parse_argv()
    ### python3 1d-cnn-loss.py 1dcnn
    if args.funtion == "1dcnn":
        train_1dcnn(args)
    ### python3 1d-cnn-loss.py 1dcnn_pred --label raffinose.csv --model_path madin_carbsubs_raffinose_best_cnn1d_model.pt ### only test data
    if args.funtion == "1dcnn_pred":
        pred_1dcnn(args)

if __name__ == "__main__":
    main()
