# Orthoformer Practical Tutorial for Biologists

**Who is this for?** Microbiologists, ecologists, and bioinformaticians who want to use Orthoformer **without** deep machine-learning expertise.

**What you will learn (≈30–60 min):**
1. What Orthoformer does—and what it does **not** do  
2. How to install the software and download a pretrained model  
3. How to turn your genome annotations into model input  
4. How to extract a **genome embedding** (a numeric “functional fingerprint”)  
5. Three example analyses you can run on those embeddings  

**Repository:** [https://github.com/JackKuo666/Orthoformer](https://github.com/JackKuo666/Orthoformer)  
**Models & datasets:** [Hugging Face – jackkuo/Orthoformer](https://huggingface.co/jackkuo/Orthoformer)

---

## 1. What Orthoformer does (in plain language)

### The biological question

A bacterial or archaeal genome is not just a string of A/T/G/C. Biologically, it is a **collection of protein families** (orthologous groups, OGs) with different **copy numbers**. Two closely related species share similar OG **composition** and **abundance patterns**.

**Orthoformer** reads this functional profile and compresses it into a fixed-length vector (an **embedding**, typically 512 numbers). That vector captures genome-wide functional organization learned from millions of genomes.

### Analogy

| Traditional approach | Orthoformer |
|---------------------|-------------|
| Compare genomes by nucleotide similarity (ANI) or a few marker genes | Compare genomes by **full OG abundance profiles** |
| Hand-crafted distance metrics | **Learned** representation from pretraining |
| Limited to sequence alignment | Works on **OG count tables** from eggNOG / emapper |

### Input → Model → Output

```
eggNOG annotations  →  OG count table per genome  →  ranked OG token sequence  →  Orthoformer  →  embedding vector
     (.emapper)            (which OGs, how many)         (top 2048 OGs)              (BERT)         (512-dim)
```

### What Orthoformer is **good** for

- Comparing many genomes in a **functional** space (clustering, ordination)  
- Taxonomic / phylogenetic analyses based on OG profiles (see `Orthoformer_Phylogeny/`, `Orthoformer_Taxon/`)  
- Feeding embeddings into downstream predictors (phenotype, CRISPR, BGC; see respective folders)  
- Exploring whether genomes group by ecology, habitat, or disease status  

### What it is **not**

- It does **not** replace eggNOG: you still need OG annotation first  
- It does **not** read raw FASTA directly in the basic tutorial (protein/DNA sequences must be annotated to OGs first)  
- It is **not** a genome assembler or gene caller  

---

## 2. Before you start: checklist

| Item | Requirement | Notes |
|------|-------------|--------|
| **Input** | eggNOG-mapper (emapper) output per genome, or a pre-built OG count matrix | See Section 4 |
| **Compute** | Linux/macOS; **GPU recommended** (≥8 GB VRAM) | CPU works but is slow for many genomes |
| **Software** | Python ≥3.10, conda or venv | Python 3.12 tested in repo |
| **Disk** | ~2–5 GB per pretrained model | Download once from Hugging Face |
| **Skills** | Basic command line (`cd`, `python`, `pip`) | No PyTorch coding required for this tutorial |

---

## 3. Installation (step by step)

### 3.1 Clone the repository

```bash
git clone https://github.com/JackKuo666/Orthoformer.git
cd Orthoformer
```

### 3.2 Create a conda environment (recommended)

```bash
conda create -n orthoformer python=3.12 -y
conda activate orthoformer
```

### 3.3 Install Python dependencies

```bash
cd foundation_model
pip install -r requirements.txt
```

### 3.4 Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If `CUDA available: True`, a GPU will be used automatically.

---

## 4. Prepare your data

### 4.1 Biological meaning of the input

Each genome is represented as a list of **OG tokens** sorted by **descending copy number** (most abundant OG first). Only the top **2,048** OGs are kept (plus special tokens `<cls>` / `<eos>`).

This matches how Orthoformer was pretrained: a “sentence” of OGs ranked by abundance.

### 4.2 Typical workflow from a new genome

1. **Predict proteins** from your assembly (Prodigal, etc.)  
2. **Run eggNOG-mapper** (`emapper.py`) on the `.faa` file  
3. **Aggregate** OG counts per genome (one row = one genome)  
4. **Map** OG names to the Orthoformer vocabulary (140K OGs)  
5. **Tokenize** → save as a Hugging Face `Dataset` on disk  

> **Shortcut for this tutorial:** use the bundled example dataset (already tokenized):  
> `foundation_model/datasets/example/`

### 4.3 Minimum columns for a count table

| Column | Meaning |
|--------|---------|
| `Sample` or genome ID | Unique identifier (e.g. `GCA_00012345.1`) |
| `OG` | eggNOG OG identifier (e.g. `2EUSZ`, `COG0001`) |
| `Count` | Number of genes assigned to that OG in the genome |

The pretrained vocabulary is in the model folder (`vocab.txt`) or at [Hugging Face](https://huggingface.co/jackkuo/Orthoformer).

### 4.4 OGs not in the vocabulary

Unknown OGs are skipped. Missing OGs are treated as absent (count = 0). This is normal for novel or rare OGs.

---

## 5. Download a pretrained model

All official models use **ALiBi** positional encoding and **span-masked language modeling** (see main [README](https://github.com/JackKuo666/Orthoformer/tree/main)).

| Model | Training genomes | Best for |
|-------|------------------|----------|
| `model_140k_2048_v18` | ~140K (JGI isolates) | **Start here** — fast, good for isolate genomes |
| `model_3M_2048_v8` | ~3M | Broad diversity, compact (512-dim, 6 layers) |
| `model_3M_2048_v10` | ~3M | Highest capacity (1024-dim, 12 layers); downstream fine-tuning |

```bash
cd foundation_model
pip install huggingface-hub

# Download the recommended starter model (~1–2 GB)
huggingface-cli download jackkuo/Orthoformer/model_140k_2048_v18 \
  --local-dir ./model/model_140k_2048_v18
```

Detailed download options: [`foundation_model/model/readme.md`](../foundation_model/model/readme.md).

---

## 6. Extract embeddings (main hands-on exercise)

### 6.1 Run the biologist-friendly quickstart script

We provide a simplified script with clear progress messages (no model-architecture dump):

```bash
cd foundation_model

# Use bundled example data + downloaded model
python biologist_quickstart.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/example \
  --output_dir outputs/example_embeddings \
  --use_alibi
```

**Expected output:**
- One `.npy` file per genome in `outputs/example_embeddings/`  
- Each file is a 1D vector of length **512** (for v18 / v8)  
- A summary table `embedding_summary.csv`  

### 6.2 Alternative: full-featured example script

The repository also includes [`feature_extraction_example.py`](../foundation_model/feature_extraction_example.py) with detailed model inspection:

```bash
python feature_extraction_example.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/example \
  --use_alibi
```

### 6.3 Which embedding type should I use?

| Method | When to use |
|--------|-------------|
| **Mean pooling** (default in quickstart) | General-purpose genome vector; stable baseline |
| **CLS token** | Alternative single-vector summary |
| **Attention pooling** | When a classification head is bundled with the model |

For clustering, PCA/UMAP, and distance-based trees, **mean pooling** is the standard choice in Orthoformer benchmarks (`Orthoformer_eval/`).

---

## 7. Example downstream analyses (what to do with embeddings)

Below are three analyses a biologist commonly wants, with pointers to repository modules.

### Example A — Visualize genomes in 2D (clustering / ordination)

**Biological goal:** Do my isolates group by taxonomy or habitat?

**Steps:**
1. Collect all `.npy` embedding files into a matrix (samples × 512)  
2. Run PCA or UMAP (`pip install umap-learn`)  
3. Color points by metadata (phylum, source, patient, etc.)  

Minimal sketch (after running quickstart):

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

emb_dir = Path("outputs/example_embeddings")
rows = []
for f in sorted(emb_dir.glob("*.npy")):
    rows.append({"sample": f.stem, "vector": np.load(f)})
X = np.vstack([r["vector"] for r in rows])
coords = PCA(n_components=2).fit_transform(X)

plt.scatter(coords[:, 0], coords[:, 1])
for i, r in enumerate(rows):
    plt.annotate(r["sample"], (coords[i, 0], coords[i, 1]), fontsize=8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Orthoformer genome embedding PCA")
plt.tight_layout(); plt.savefig("outputs/example_pca.png", dpi=150)
print("Saved outputs/example_pca.png")
```

**Full benchmark pipeline:** [`Orthoformer_eval/readme.md`](../Orthoformer_eval/readme.md)

---

### Example B — Build a phylogenetic tree from embeddings

**Biological goal:** How related are these genomes functionally?

Orthoformer embeddings can be converted to a distance matrix and passed to neighbor-joining (NJ):

**Repository tool:** [`Orthoformer_Phylogeny/build_tree_from_embeddings.py`](../Orthoformer_Phylogeny/build_tree_from_embeddings.py)

```bash
cd Orthoformer_Phylogeny
python build_tree_from_embeddings.py \
  --embedding_dir ../foundation_model/outputs/example_embeddings \
  --output_tree outputs/example.nwk \
  --method nj
```

Open `outputs/example.nwk` in [iTOL](https://itol.embl.de/) or FigTree.

**Reference evaluation (GTDB archaea):** [`Orthoformer_Phylogeny/ar53_r226/readme.md`](../Orthoformer_Phylogeny/ar53_r226/readme.md)

---

### Example C — Taxonomic placement from embeddings

**Biological goal:** What genus/species is this MAG closest to?

Use pretrained embeddings + the CLEAN-based taxonomy module:

1. Generate embeddings (Section 6)  
2. Follow [`Orthoformer_Taxon/readme.md`](../Orthoformer_Taxon/readme.md)  
   - `demo_train.py` — embedding distances  
   - `train-triplet.py` — train classifier  
   - `inference.py` — predict taxon at chosen rank  

---

### Other applications in this repository

| Folder | Biological application |
|--------|------------------------|
| [`Orthoformer_Phenotype/`](../Orthoformer_Phenotype/) | Predict growth phenotypes from embeddings |
| [`Orthoformer_CRISPR/`](../Orthoformer_CRISPR/) | CRISPR-associated protein token classification |
| [`Orthoformer_BGC/`](../Orthoformer_BGC/) | Biosynthetic gene cluster abundance regression |
| [`Orthoformer_eval/`](../Orthoformer_eval/) | Compare Orthoformer vs. other foundation models on taxonomy benchmarks |

---

## 8. Batch processing many genomes (CNGBdb-scale)

For hundreds or thousands of genomes, use the evaluation CLI (same logic as the manuscript):

```bash
cd Orthoformer_eval

python scripts/s1_generate_embeddings_cli.py \
  --model_dir /path/to/model_140k_2048_v18 \
  --dataset_path /path/to/your_tokenized.dataset \
  --sample_list my_genome_list.txt \
  --output_dir embeddings/my_project \
  --batch_size 16 \
  --model_max_length 2048 \
  --device cuda:0 \
  --use_alibi \
  --output_mode mean
```

- `my_genome_list.txt`: one genome ID per line  
- Already-finished genomes are skipped (safe to restart)  
- Output: one `GENOME_ID.npy` per genome  

---

## 9. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` | Batch too large or sequence too long | Reduce `--batch_size` to 4 or 8 |
| Very slow on CPU | No GPU | Use GPU node; or process fewer genomes first |
| `UNEXPECTED` / `MISSING` keys when loading | Loading ALiBi model without `--use_alibi` | Add `--use_alibi` for v8, v10, v18 |
| All embeddings look similar | Wrong model path or empty input | Check `dataset` has non-zero `input_ids` lengths |
| OG not found in vocab | OG outside 140K training dictionary | Expected; annotate with eggNOG standard OGs |
| Download interrupted | Network | Re-run `huggingface-cli download` (resumes automatically) |

---

## 10. Quick reference card

```bash
# --- One-time setup ---
git clone https://github.com/JackKuo666/Orthoformer.git && cd Orthoformer
conda create -n orthoformer python=3.12 -y && conda activate orthoformer
pip install -r foundation_model/requirements.txt
huggingface-cli download jackkuo/Orthoformer/model_140k_2048_v18 \
  --local-dir foundation_model/model/model_140k_2048_v18

# --- Extract embeddings (example data) ---
cd foundation_model
python biologist_quickstart.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/example \
  --output_dir outputs/my_run \
  --use_alibi

# --- Next: PCA / tree / taxonomy (Sections 7A–7C) ---
```

---

## 11. How to cite

If you use Orthoformer, please cite our manuscript (see main [README](https://github.com/JackKuo666/Orthoformer/blob/main/README.md)) and acknowledge:

- **eggNOG-mapper** for OG annotation  
- **Hugging Face Transformers** for model infrastructure  

---

## 12. Getting help

- **GitHub Issues:** [https://github.com/JackKuo666/Orthoformer/issues](https://github.com/JackKuo666/Orthoformer/issues)  
- **Foundation model details:** [`foundation_model/README.md`](../foundation_model/README.md)  
- **Model files:** [https://huggingface.co/jackkuo/Orthoformer](https://huggingface.co/jackkuo/Orthoformer)  

---

*This tutorial was written to address usability feedback from reviewers and to lower the barrier for biologists adopting functional genome embeddings in routine analyses.*
