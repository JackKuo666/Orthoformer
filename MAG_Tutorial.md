# MAG Workflow Tutorial: From Genes to Applications

**Who is this for?** Researchers working with **metagenome-assembled genomes (MAGs)** who want a complete, practical path from protein-coding genes to Orthoformer embeddings and downstream analyses.

**What you will learn (≈1–2 hours hands-on):**
1. How MAGs differ from isolate genomes in the Orthoformer pipeline  
2. Gene calling → eggNOG annotation → OG count table → tokenized dataset  
3. Extracting a MAG embedding with `mag_preprocess.py` + `biologist_quickstart.py`  
4. Three real applications: compare with reference MAGs, build a tree, taxonomic placement  

**Prerequisites:** Read the general [Biologist Tutorial](Biologist_Tutorial.md) for background on what Orthoformer does.

**Repository:** [https://github.com/JackKuo666/Orthoformer](https://github.com/JackKuo666/Orthoformer)  
**Models:** [Hugging Face – jackkuo/Orthoformer](https://huggingface.co/jackkuo/Orthoformer)

---

## 1. End-to-end pipeline (overview)

A MAG is treated like any other genome in Orthoformer: **one functional profile → one embedding vector**.

```
MAG assembly (.fa)
    ↓  Prodigal (meta mode)
Protein sequences (.faa)
    ↓  eggNOG-mapper (emapper.py)
OG annotations (.emapper.annotations)
    ↓  mag_preprocess.py
Tokenized dataset (.dataset)
    ↓  biologist_quickstart.py
Embedding vector (.npy)
    ↓  PCA / tree / taxonomy / phenotype modules
Biological interpretation
```

### MAG-specific considerations

| Topic | What to know |
|-------|----------------|
| **Incomplete genomes** | Missing OGs are treated as absent (count = 0). Orthoformer still works, but very small MAGs (< ~50 genes) give weak embeddings. |
| **Contamination** | Run CheckM2 / GUNC **before** Orthoformer. Contaminant genes change the OG profile. |
| **Gene calling** | Use `prodigal -p meta` on MAG assemblies. Do not use `-p single` (isolate mode). |
| **Query ID format** | Name proteins as `MAG_ID\|gene_001` so one emapper file can hold one or many MAGs. |
| **Model choice** | Start with `model_140k_2048_v18` for single MAGs; use `model_3M_2048_v8` when comparing across phyla. |

---

## 2. Before you start: checklist

| Item | Requirement | Notes |
|------|-------------|--------|
| **Starting point** | MAG assembly (`.fa`) **or** predicted proteins (`.faa`) | Skip gene calling if you already have `.faa` |
| **QC** | CheckM2 / GUNC recommended | Not bundled in Orthoformer |
| **Annotation** | eggNOG-mapper ≥ 2.1 | [eggNOG database](http://eggnog5.embl.de/) |
| **Orthoformer** | Python ≥ 3.10, GPU ≥ 8 GB VRAM recommended | Same setup as Biologist Tutorial |
| **Model** | `model_140k_2048_v18` downloaded | ~1–2 GB from Hugging Face |

---

## 3. One-time setup

Follow [Biologist Tutorial Section 3](Biologist_Tutorial.md#3-installation-step-by-step), then download the model:

```bash
git clone https://github.com/JackKuo666/Orthoformer.git
cd Orthoformer
conda create -n orthoformer python=3.12 -y
conda activate orthoformer
pip install -r foundation_model/requirements.txt

cd foundation_model
pip install huggingface-hub
huggingface-cli download jackkuo/Orthoformer/model_140k_2048_v18 \
  --local-dir ./model/model_140k_2048_v18
```

### External tools (install separately)

```bash
# Gene calling
conda install -c bioconda prodigal

# OG annotation (eggNOG-mapper)
conda create -n eggnog -c bioconda -c defaults eggnog-mapper python=3.10 -y
conda activate eggnog
download_eggnog_data.py -y
```

---

## 4. Phase 1 — From MAG assembly to proteins

### 4.1 Quality control (recommended)

Before gene calling, check completeness and contamination:

```bash
# Example with CheckM2
checkm2 predict -i MAG001.fa -o checkm2_MAG001 -x fa --threads 8
```

Typical filters for downstream analysis:
- Completeness ≥ 50% (higher is better for taxonomy)
- Contamination ≤ 5–10%

Orthoformer does **not** enforce these thresholds; they are biological best practice.

### 4.2 Predict genes with Prodigal (meta mode)

```bash
prodigal -i MAG001.fa -a MAG001.faa -d MAG001.fna -p meta -f gff -o MAG001.gff
```

**Important:** `-p meta` is required for MAGs and metagenomes.

Check output:

```bash
grep -c "^>" MAG001.faa    # number of predicted proteins
```

A MAG with fewer than ~100 proteins may still run, but embedding quality will be limited.

### 4.3 Name proteins for downstream parsing

Orthoformer preprocessing expects protein IDs like:

```
MAG001|gene_00001
MAG001|gene_00002
```

If your `.faa` headers are different, rename them before emapper:

```bash
awk '/^>/{print ">MAG001|"$0; next} {print}' MAG001.faa > MAG001.renamed.faa
# Or use seqkit rename / custom script for batch renaming
```

---

## 5. Phase 2 — eggNOG annotation

### 5.1 Run eggNOG-mapper

```bash
conda activate eggnog

emapper.py -i MAG001.faa -o MAG001 --cpu 8 -m diamond
```

**Output file:** `MAG001.emapper.annotations`

### 5.2 Quick sanity check

```bash
# Annotated proteins (excluding header lines)
tail -n +5 MAG001.emapper.annotations | wc -l

# Example OG assignments
tail -n +5 MAG001.emapper.annotations | cut -f1,5 | head
```

You should see:
- Column 1 (`#query`): protein names with your MAG ID  
- Column 5 (`eggNOG_OGs`): OG identifiers like `COG0001@2|Bacteria,...`

### 5.3 Multiple MAGs in one emapper file

If one `.faa` contains proteins from several MAGs (`MAG002|...`, `MAG003|...`), a single emapper run is fine. `mag_preprocess.py` will create **one dataset row per MAG** automatically.

To force a single MAG ID regardless of headers:

```bash
python mag_preprocess.py ... --mag_id MAG001
```

---

## 6. Phase 3 — Build the Orthoformer input dataset

We provide `mag_preprocess.py` in `foundation_model/` to convert emapper output into a HuggingFace dataset using the model vocabulary (`vocab.txt`).

```bash
cd foundation_model

python mag_preprocess.py \
  --emapper /path/to/MAG001.emapper.annotations \
  --model_dir model/model_140k_2048_v18 \
  --output_dir datasets/MAG001.dataset \
  --counts_tsv outputs/MAG001_og_counts.tsv
```

**What this does:**
1. Parses `.emapper.annotations` (skips first 4 header lines)  
2. Counts OGs per MAG (first OG in `eggNOG_OGs` column)  
3. Ranks OGs by abundance (same rule as training data)  
4. Maps OGs to token IDs from `vocab.txt`  
5. Keeps top 2048 OGs + `<cls>` / `<eos>` special tokens  
6. Saves `datasets/MAG001.dataset/` (HuggingFace format)

**Inspect the intermediate table** (`MAG001_og_counts.tsv`):

```
Sample    OG      Count   PFAM_Count   max_PFAM_Count   PFAMs
MAG001    2EUSZ   12      8            3                PF00001,PF00002
MAG001    COG0001  5      2            1                PF12345
```

---

## 7. Phase 4 — Extract the MAG embedding

```bash
python biologist_quickstart.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/MAG001.dataset \
  --output_dir outputs/MAG001_embedding \
  --use_alibi \
  --batch_size 4
```

**Outputs:**
- `outputs/MAG001_embedding/MAG001.npy` — 512-dimensional embedding (for v18)  
- `outputs/MAG001_embedding/embedding_summary.csv` — metadata table  

Verify:

```python
import numpy as np
vec = np.load("outputs/MAG001_embedding/MAG001.npy")
print(vec.shape)   # (512,)
print(vec[:5])     # first five dimensions
```

---

## 8. Phase 5 — Applications

### Application A — Compare your MAG with reference genomes

**Biological goal:** Where does my MAG sit relative to known isolates or other MAGs?

**Steps:**
1. Process reference genomes the same way (emapper → dataset → embeddings)  
2. Collect all `.npy` files into one folder  
3. Run PCA or UMAP and color by metadata  

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

emb_dir = Path("outputs/all_embeddings")
meta = pd.read_csv("metadata.tsv", sep="\t")  # columns: genome_id, source, phylum, ...

rows = []
for f in sorted(emb_dir.glob("*.npy")):
    rows.append({"genome_id": f.stem, "vector": np.load(f)})
X = np.vstack([r["vector"] for r in rows])
ids = [r["genome_id"] for r in rows]

coords = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(8, 6))
for i, gid in enumerate(ids):
  color = "red" if gid == "MAG001" else "steelblue"
  size = 80 if gid == "MAG001" else 30
  plt.scatter(coords[i, 0], coords[i, 1], c=color, s=size)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("MAG001 vs reference genomes")
plt.tight_layout(); plt.savefig("outputs/MAG001_pca.png", dpi=150)
```

**Tip:** Include GTDB or JGI reference MAGs in the same embedding space for interpretable ordination.

---

### Application B — Phylogenetic placement (functional tree)

**Biological goal:** How is my MAG related to a set of reference genomes functionally?

```bash
cd ../Orthoformer_Phylogeny

python build_tree_from_embeddings.py \
  --data_dir ../foundation_model/outputs/all_embeddings \
  --out_prefix outputs/MAG001_with_refs \
  --method nj_skbio \
  --metric euclidean
```

**Outputs:**
- `outputs/MAG001_with_refs.nwk` — Newick tree  
- Distance matrix files (if enabled)

Open the `.nwk` file in [iTOL](https://itol.embl.de/) or FigTree. Highlight `MAG001` to see its position.

**Reference:** [`Orthoformer_Phylogeny/readme.md`](../Orthoformer_Phylogeny/readme.md)

---

### Application C — Taxonomic placement

**Biological goal:** What genus or species is this MAG closest to?

1. Generate embeddings for your MAG **and** a labeled reference set (GTDB isolates, JGI isolates, etc.)  
2. Follow [`Orthoformer_Taxon/readme.md`](../Orthoformer_Taxon/readme.md):

```bash
cd ../Orthoformer_Taxon

# Build embedding distance matrix (demo)
python demo_train.py \
  --embedding_dir ../foundation_model/outputs/all_embeddings \
  --taxonomy_file reference_taxonomy.tsv

# Train triplet classifier at desired rank (e.g. genus)
python train-triplet.py --rank genus ...

# Predict taxonomy for MAG001
python inference.py --query MAG001 --rank genus ...
```

**Note:** Taxonomy accuracy depends on reference coverage at the target rank. For incomplete MAGs, genus-level placement is more reliable than species-level.

---

### Application D — Phenotype prediction (optional)

If you have a pretrained phenotype model (e.g. carbon source utilization):

- See [`Orthoformer_Phenotype/`](../Orthoformer_Phenotype/)  
- Input: MAG embedding or tokenized dataset  
- Requires a task-specific trained classifier on top of Orthoformer embeddings  

This is advanced and task-dependent; see phenotype module README for available models.

---

## 9. Batch processing many MAGs

For dozens or hundreds of MAGs:

### Step 1 — Batch emapper (per batch of `.faa` files)

```bash
for faa in mag_proteins/*.faa; do
  base=$(basename "$faa" .faa)
  emapper.py -i "$faa" -o "emapper_out/$base" --cpu 8 -m diamond
done
```

### Step 2 — Combine into one dataset

Process each emapper file, then merge datasets:

```bash
python mag_preprocess.py \
  --emapper emapper_out/MAG001.emapper.annotations \
  --model_dir model/model_140k_2048_v18 \
  --output_dir datasets/batch_part1/MAG001.dataset

# Repeat for each MAG, or write a simple loop
```

For large-scale projects, use the evaluation CLI after building a combined `.dataset`:

```bash
cd ../Orthoformer_eval
python scripts/s1_generate_embeddings_cli.py \
  --model_dir /path/to/model_140k_2048_v18 \
  --dataset_path /path/to/combined_MAGs.dataset \
  --sample_list mag_list.txt \
  --output_dir embeddings/my_MAG_project \
  --batch_size 16 \
  --model_max_length 2048 \
  --use_alibi \
  --output_mode mean
```

---

## 10. Worked example (minimal test)

If you already have an emapper file from a public MAG:

```bash
cd foundation_model

# 1) Preprocess
python mag_preprocess.py \
  --emapper /path/to/MAG001.emapper.annotations \
  --model_dir model/model_140k_2048_v18 \
  --output_dir datasets/MAG001.dataset

# 2) Embed
python biologist_quickstart.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/MAG001.dataset \
  --output_dir outputs/MAG001_embedding \
  --use_alibi

# 3) Quick PCA with example data (optional)
python biologist_quickstart.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/example \
  --output_dir outputs/example_embedding \
  --use_alibi
# Then combine outputs/MAG001_embedding/ and outputs/example_embedding/ for PCA (Section 8A)
```

---

## 11. Troubleshooting (MAG-specific)

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `No OG counts found` | Empty or malformed emapper file | Re-run emapper; check `tail -n +5 file \| wc -l` |
| `No OGs matched the model vocabulary` | Wrong OG namespace or failed annotation | Ensure eggNOG OGs are assigned; update eggNOG DB |
| Very short `sequence_length` in summary | MAG has few annotated genes | Check Prodigal output; verify CheckM completeness |
| MAG clusters with wrong phylum | Contamination or incomplete assembly | Run CheckM2/GUNC; remove contaminant contigs |
| Multiple rows in dataset from one MAG | Protein headers use different sample prefixes | Use `--mag_id MAG001` to collapse |
| `UNEXPECTED` / `MISSING` keys when loading | ALiBi flag missing | Add `--use_alibi` for v8/v10/v18 |
| Embedding similar to all references | Empty or near-empty OG profile | Inspect `MAG001_og_counts.tsv` |

---

## 12. Quick reference card

```bash
# --- One-time setup ---
conda activate orthoformer
cd Orthoformer/foundation_model

# --- Gene calling (if starting from assembly) ---
prodigal -i MAG001.fa -a MAG001.faa -p meta

# --- eggNOG (separate conda env) ---
conda activate eggnog
emapper.py -i MAG001.faa -o MAG001 --cpu 8 -m diamond

# --- Orthoformer preprocessing + embedding ---
conda activate orthoformer
python mag_preprocess.py \
  --emapper MAG001.emapper.annotations \
  --model_dir model/model_140k_2048_v18 \
  --output_dir datasets/MAG001.dataset

python biologist_quickstart.py \
  --model_dir model/model_140k_2048_v18 \
  --dataset_path datasets/MAG001.dataset \
  --output_dir outputs/MAG001_embedding \
  --use_alibi

# --- Downstream: tree / taxonomy (see Sections 8B–8C) ---
```

---

## 13. Related documentation

| Document | Content |
|----------|---------|
| [Biologist_Tutorial.md](Biologist_Tutorial.md) | General Orthoformer intro, installation, embedding concepts |
| [foundation_model/README.md](../foundation_model/README.md) | Model details, feature extraction options |
| [Orthoformer_Phylogeny/readme.md](../Orthoformer_Phylogeny/readme.md) | Tree building from embeddings |
| [Orthoformer_Taxon/readme.md](../Orthoformer_Taxon/readme.md) | CLEAN-based taxonomy inference |
| [Orthoformer_eval/readme.md](../Orthoformer_eval/readme.md) | Large-scale embedding benchmarks |

---

## 14. How to cite

Please cite the Orthoformer manuscript (see main [README](https://github.com/JackKuo666/Orthoformer/blob/main/README.md)) and acknowledge:

- **Prodigal** for gene calling  
- **eggNOG-mapper** for OG annotation  
- **CheckM2** / **GUNC** if used for MAG QC  

---

*This tutorial complements the general Biologist Tutorial with a MAG-focused workflow from gene sequences to applied analyses.*
