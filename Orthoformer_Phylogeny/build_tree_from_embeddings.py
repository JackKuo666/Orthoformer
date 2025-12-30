import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import re
import io

from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo

try:
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from skbio import DistanceMatrix as SkbioDistanceMatrix
    from skbio.tree import nj as skbio_nj
    HAS_SKBIO = True
except Exception:
    HAS_SKBIO = False

try:
    from ete3 import Tree as EteTree
    HAS_ETE3 = True
except Exception:
    HAS_ETE3 = False

import hashlib


def _stable_hex_color(text: str) -> str:
    """Map an arbitrary string to a stable hex color (#RRGGBB).
    Uses md5 hash -> H in [0,360), fixed S/L, and converts HSL->RGB.
    """
    if not text:
        return "#808080"  # grey fallback
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 360
    # fixed saturation/lightness for good readability
    s, l = 0.65, 0.50
    # HSL to RGB
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h/60) % 2 - 1))
    m = l - c/2
    if   0 <= h < 60:   r_, g_, b_ = c, x, 0
    elif 60 <= h < 120: r_, g_, b_ = x, c, 0
    elif 120 <= h < 180:r_, g_, b_ = 0, c, x
    elif 180 <= h < 240:r_, g_, b_ = 0, x, c
    elif 240 <= h < 300:r_, g_, b_ = x, 0, c
    else:               r_, g_, b_ = c, 0, x
    r = int(round((r_ + m) * 255))
    g = int(round((g_ + m) * 255))
    b = int(round((b_ + m) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def _pick_text_style(_: str) -> str:
    """Return a valid iTOL text style. Keep it simple for now."""
    return "normal"  # or "bold"/"italic"


def _read_input_samples(path: str):
    """Read a file containing one sample name per line; return a set of names.
    Ignores empty lines and lines starting with '#'.
    """
    names = set()
    with open(path, 'r') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            names.add(s)
    return names


def load_embeddings_from_dir(data_dir, include_names: set | None = None):
    """Load per-sample embeddings from a directory.
    Supported formats:
      - .npy : 2D array [L+2, D]; we take mean over 1:-1 -> [D]
      - .pt  : 1D tensor [D] (already pooled); loaded via torch.load
    If include_names is provided, filter by file stem.
    Returns (X, names) with names in the same order as loaded files.
    """
    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    pt_files  = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    if npy_files and pt_files:
        raise ValueError(f"Found mixed formats in {data_dir}: both .npy and .pt. Please keep one format per directory.")

    if npy_files:
        fmt = "npy"
        files = npy_files
    elif pt_files:
        fmt = "pt"
        files = pt_files
    else:
        raise FileNotFoundError(f"No .npy or .pt files found under {data_dir}")

    # Filter by include_names (stems)
    if include_names is not None:
        before = len(files)
        files = [p for p in files if os.path.splitext(os.path.basename(p))[0] in include_names]
        found_names = {os.path.splitext(os.path.basename(p))[0] for p in files}
        missing = sorted(list(include_names - found_names))
        print(f"Filtering by --input_samples: {len(files)}/{before} files matched.")
        if missing:
            print(f"Warning: {len(missing)} requested samples not found in {data_dir}. Example: {missing[:5]}")
        if not files:
            raise FileNotFoundError("No matching files after applying --input_samples filter.")

    feats, names = [], []
    desc = "Loading and averaging" if fmt == "npy" else "Loading .pt vectors"
    for path in tqdm(files, desc=desc):
        stem = os.path.splitext(os.path.basename(path))[0]
        if fmt == "npy":
            x = np.load(path)  # [L+2, D]
            if x.ndim != 2:
                raise ValueError(f"{path} has shape {x.shape}, expected 2D array.")
            if x.shape[0] < 2:
                raise ValueError(f"{path} has too few rows to drop <cls>/<eos>.")
            vec = x[1:-1, :].mean(axis=0)  # [D]
        else:  # fmt == "pt"
            t = torch.load(path, map_location="cpu")
            # Accept tensor, numpy array, list; coerce to 1D numpy vector
            if hasattr(t, "detach"):
                t = t.detach().cpu().numpy()
            elif isinstance(t, np.ndarray):
                pass
            else:
                t = np.asarray(t)
            if t.ndim == 2 and t.shape[0] == 1:
                t = t.reshape(-1)
            if t.ndim != 1:
                raise ValueError(f"{path} expected 1D vector [D], got shape {t.shape}")
            vec = t.astype(np.float32, copy=False)
        feats.append(vec)
        names.append(stem)

    X = np.vstack(feats)
    return X, names


def save_matrix_and_csv(X, names, out_prefix, save_csv=False):
    np.save(f"{out_prefix}.features.npy", X)
    with open(f"{out_prefix}.sample_names.txt", "w") as f:
        for s in names:
            f.write(s + "\n")
    if save_csv:
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df.insert(0, "sample_name", names)
        df.to_csv(f"{out_prefix}.features.csv", index=False)
    print(f"Saved features to {out_prefix}.features.npy and sample names to {out_prefix}.sample_names.txt")


def l2_normalize(X):
    return normalize(X, norm="l2", axis=1)


def compute_distance_matrix(X_norm, metric="euclidean"):
    D = pairwise_distances(X_norm.astype(np.float32, copy=False), metric=metric)
    return D.astype(np.float32, copy=False)


def _to_biopython_distance_matrix(square_D, names):
    N = len(names)
    for i in range(N):
        square_D[i, i] = 0.0
    matrix_list = []
    for i in range(N):
        row = [float(square_D[i, j]) for j in range(i)] + [0.0]
        matrix_list.append(row)
    return DistanceMatrix(names, matrix_list)


# ---------- Fast UPGMA via SciPy ----------

def _linkage_to_newick(Z, names):
    """Convert SciPy linkage (average/UPGMA) to Newick with ultrametric branch lengths.
    Uses cluster heights: branch = parent.dist - child.dist.
    """
    root = to_tree(Z, rd=False)

    def build(node, parent_height):
        if node.is_leaf():
            name = names[node.id]
            blen = max(parent_height - node.dist, 0.0)
            return f"{name}:{blen:.10f}"
        left = build(node.left, node.dist)
        right = build(node.right, node.dist)
        blen = max(parent_height - node.dist, 0.0)
        return f"({left},{right}):{blen:.10f}"

    return build(root, root.dist) + ";"


def upgma_scipy(D, names, out_newick):
    if not HAS_SCIPY:
        raise RuntimeError("SciPy not available for fast UPGMA.")
    # condensed vector is faster/smaller
    D = np.asarray(D, dtype=np.float64)
    Y = squareform(D, checks=False)  # len = N*(N-1)/2
    Z = linkage(Y, method="average", optimal_ordering=False)
    newick = _linkage_to_newick(Z, names)
    with open(out_newick, "w") as f:
        f.write(newick)
    print(f"Saved tree (UPGMA-SciPy) to {out_newick}")
    return out_newick


# ---------- Fast NJ via scikit-bio ----------

def nj_skbio(D, names, out_newick):
    if not HAS_SKBIO:
        raise RuntimeError("scikit-bio not available for fast NJ.")
    dm = SkbioDistanceMatrix(np.asarray(D, dtype=float), names)
    tree = skbio_nj(dm)
    # Get Newick string first
    try:
        newick = tree.to_newick()
    except AttributeError:
        buf = io.StringIO()
        tree.write(buf)
        newick = buf.getvalue()
    # Normalize: strip single quotes around simple labels (letters/digits/_ . -)
    # This keeps quotes for labels that truly need them (e.g., those containing spaces or colons)
    newick = re.sub(r"'([A-Za-z0-9_.-]+)'", r"\1", newick)
    with open(out_newick, "w") as f:
        f.write(newick)
    print(f"Saved tree (NJ-skbio) to {out_newick}")
    return out_newick


def build_tree(D, names, method="nj", out_newick="tree.nwk"):
    """
    由距离矩阵构树，method ∈ {
        'nj', 'upgma',          # Bio.Phylo (原版)
        'nj_skbio',             # 更快的 NJ（如安装 scikit-bio）
        'upgma_scipy'           # 更快的 UPGMA（SciPy linkage）
    }
    输出 Newick 到 out_newick。
    """
    method = method.lower()
    print(f"constructing tree with method: {method}")
    
    if method == "upgma_scipy" and HAS_SCIPY:
        return upgma_scipy(D, names, out_newick)
    if method == "nj_skbio" and HAS_SKBIO:
        return nj_skbio(D, names, out_newick)

    dm = _to_biopython_distance_matrix(np.asarray(D, dtype=float).copy(), names)
    constructor = DistanceTreeConstructor()

    if method == "nj":
        tree = constructor.nj(dm)
    elif method == "upgma":
        tree = constructor.upgma(dm)
    else:
        raise ValueError("method must be one of: 'nj', 'upgma', 'nj_skbio', 'upgma_scipy'")

    Phylo.write(tree, out_newick, "newick")
    print(f"Saved tree ({method.upper()}) to {out_newick}")
    return out_newick


def load_taxonomy_map(taxonomy_map_tsv):
    """
    读取 taxonomy 映射：两列 TSV：sample_name \t taxonomy_string
    返回 dict: name -> taxonomy
    """
    mapping = {}
    with open(taxonomy_map_tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def write_itol_text_labels(names, taxonomy_map, out_file="itol_ranges.txt"):
    """
    生成 iTOL 的 DATASET_RANGE 文件（高亮单叶/克莱德）。
    这里按 *range* 示例的格式输出（SEPARATOR COMMA），每个样本一行：
      START_NODE_ID,END_NODE_ID,FILL_COLOR
    我们将 START=END=样本名；颜色由 taxonomy 文本经稳定哈希得到。
    参考: iTOL range dataset 示例。
    """
    rows = 0
    with open(out_file, "w") as f:
        # Header (modeled after example; COMMA separator)
        f.write("DATASET_RANGE\n")
        f.write("SEPARATOR COMMA\n")
        f.write("DATASET_LABEL,range\n")
        f.write("COLOR,#ffff00\n")
        # Reasonable defaults matching the shared example
        f.write("RANGE_TYPE,box\n")
        f.write("RANGE_COVER,clade\n")
        f.write("UNROOTED_SMOOTH,simplify\n")
        f.write("COVER_LABELS,1\n")
        f.write("COVER_DATASETS,0\n")
        f.write("FIT_LABELS,0\n")
        f.write("SHOW_LABELS,1\n")
        f.write("LABEL_POSITION,bottom-right\n")
        f.write("LABELS_VERTICAL,0\n")
        f.write("STRAIGHT_LABELS,0\n")
        f.write("LABEL_ROTATION,0\n")
        f.write("LABEL_SHIFT_X,0\n")
        f.write("LABEL_SHIFT_Y,0\n")
        f.write("LABEL_SIZE_FACTOR,1\n")
        f.write("DATA\n")
        # Data rows: START_NODE_ID,END_NODE_ID,FILL_COLOR[,GRADIENT_COLOR,LINE_COLOR,LINE_STYLE,LINE_WIDTH,LABEL_TEXT,LABEL_COLOR,LABEL_SIZE_FACTOR,LABEL_STYLE]
        for n in names:
            label = taxonomy_map.get(n, None)
            if label is None or str(label).strip() == "":
                continue
            color = _stable_hex_color(str(label))
            # 最简三列: start,end,fill_color
            f.write(f"{n},{n},{color}\n")
            rows += 1
    print(f"Saved iTOL RANGE dataset to {out_file} (rows: {rows})")
    return out_file


def compute_nrf(pred_tree_newick, ref_tree_newick, out_pruned_pred=None, out_pruned_ref=None):
    """
    计算 normalized RF 距离：
      - 将两棵树裁剪到共有叶子集合
      - 计算 RF 与 nRF
    需要 ete3，可选保存裁剪后的 newick。
    """
    if not HAS_ETE3:
        print("ete3 not installed; skip nRF. (pip install ete3)")
        return None

    print(f"computing nRF with ete3")
    t1 = EteTree(pred_tree_newick, format=1)
    t2 = EteTree(ref_tree_newick, format=1)

    leaves1 = set(t1.get_leaf_names())
    leaves2 = set(t2.get_leaf_names())
    common = leaves1 & leaves2
    print(f"common leaves: {len(common)}")
    
    if len(common) < 3:
        print("Too few shared leaves (<3) for meaningful RF.")
        return None

    # 裁剪到共同叶子
    t1.prune(common, preserve_branch_length=True)
    t2.prune(common, preserve_branch_length=True)

    if out_pruned_pred:
        t1.write(format=1, outfile=out_pruned_pred)
        print(f"Saved pruned predicted tree to {out_pruned_pred}")
    if out_pruned_ref:
        t2.write(format=1, outfile=out_pruned_ref)
        print(f"Saved pruned reference tree to {out_pruned_ref}")

    rf, max_rf, *_ = t1.robinson_foulds(t2, unrooted_trees=True)
    nrf = rf / max_rf if max_rf > 0 else 0.0
    print(f"RF = {rf}, maxRF = {max_rf}, nRF = {nrf:.6f}")
    return {"RF": rf, "maxRF": max_rf, "nRF": nrf, "shared_leaves": len(common)}


def main():
    parser = argparse.ArgumentParser(description="Build tree from per-sample .npy embeddings")
    parser.add_argument("--data_dir", type=str, default="dataset", help="dir with *.npy (one per sample)")
    parser.add_argument("--out_prefix", type=str, default="out", help="prefix for outputs")
    parser.add_argument("--save_csv", action="store_true", help="also write a big CSV with features")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine", "manhattan"], help="distance metric")
    parser.add_argument("--no_l2", action="store_true", help="disable L2 normalization before distance computation")
    parser.add_argument("--method", type=str, default="nj_skbio", choices=["nj_skbio", "upgma_scipy", "nj", "upgma"], help="tree method")
    parser.add_argument("--taxonomy_map", type=str, default=None, help="TSV: sample_name\\ttaxonomy_string")
    parser.add_argument("--itol_labels", type=str, default=None, help="output iTOL text dataset file (if provided, will generate)")
    parser.add_argument("--ref_tree", type=str, default=None, help="reference tree (newick) for nRF")
    parser.add_argument("--pruned_pred", type=str, default=None, help="save pruned predicted tree newick (optional)")
    parser.add_argument("--pruned_ref", type=str, default=None, help="save pruned reference tree newick (optional)")
    parser.add_argument("--input_samples", type=str, default=None, help="Optional file with target sample names (one per line). If provided, only these samples will be used to build the tree.")
    args = parser.parse_args()

    # 1) 加载并取均值得到 [N, D]
    include_names = _read_input_samples(args.input_samples) if args.input_samples else None
    X, names = load_embeddings_from_dir(args.data_dir, include_names=include_names)
    save_matrix_and_csv(X, names, args.out_prefix, save_csv=args.save_csv)

    # 2) 可选 L2 归一化（按行）
    if args.no_l2:
        print("Skipping L2 normalization (--no_l2)")
        X_used = X
    else:
        print("Applying L2 normalization")
        X_used = l2_normalize(X)

    # 3) 距离矩阵
    D = compute_distance_matrix(X_used, metric=args.metric)
    np.save(f"{args.out_prefix}.dist.npy", D)
    print(f"Saved distance matrix to {args.out_prefix}.dist.npy")

    # 4) NJ / UPGMA 构树
    newick_path = f"{args.out_prefix}.{args.method}.nwk"
    build_tree(D, names, method=args.method, out_newick=newick_path)

    # 5) taxonomy 装饰（iTOL 文本数据集）
    if args.taxonomy_map and args.itol_labels:
        tx = load_taxonomy_map(args.taxonomy_map)
        write_itol_text_labels(names, tx, out_file=args.itol_labels)

    # 6) nRF 与参考树比较（可选）
    if args.ref_tree:
        if not os.path.isfile(args.ref_tree):
            print(f"Reference tree not found: {args.ref_tree}")
        else:
            compute_nrf(
                pred_tree_newick=newick_path,
                ref_tree_newick=args.ref_tree,
                out_pruned_pred=args.pruned_pred,
                out_pruned_ref=args.pruned_ref
            )


if __name__ == "__main__":
    main()
    
"""
## Example: taxonomy_map.tsv
GB_GCA_001303465.1    d__Bacteria;p__Firmicutes;...
GB_GCA_019116455.1    d__Bacteria;p__Proteobacteria;...

## Usage
python build_tree_from_embeddings.py \
  --data_dir dataset \  # 存放每个样本的.npy文件
  --out_prefix result \  # 输出文件前缀
  --method nj \  # 构树方法
  --metric euclidean \  # 距离度量
  --save_csv \  # 保存CSV文件（可选）
  --taxonomy_map taxonomy_map.tsv \  # 物种分类映射文件（可选）
  --itol_labels itol_labels.txt \  # 输出iTOL文本标注文件（可选）
  --ref_tree reference.nwk \     # 参考树文件（可选）
  --pruned_pred result.pruned_pred.nwk \  # 裁剪后的预测树文件（可选）
  --pruned_ref result.pruned_ref.nwk \  # 裁剪后的参考树文件（可选）
"""