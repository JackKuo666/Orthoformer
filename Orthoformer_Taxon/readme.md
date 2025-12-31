This implementation builds upon the CLEAN framework (https://github.com/tttianhao/CLEAN), which we have adapted for taxonomy classification tasks. We gratefully acknowledge the contributions of the CLEAN authors.

## Environment Setup
Please configure the environment according to CLEAN's requirements. Refer to the original repository for detailed dependency specifications.

**Usage Instructions:**
1. **Embedding Distance Calculation:** Run `demo_train.py` to compute pairwise distances between all input embeddings. These embeddings should be pretrained using Orthoformer.

2. **Model Training:** Execute `train-triplet.py` to train the CLEAN model on your taxonomy classification task.

3. **Taxonomy Inference:** Use `inference.py` to predict the taxonomic ID for a given pretrained embedding at a specific taxonomic level (e.g., genus as shown in the example).

### 1. MODEL TRAINING
[ Step 1 ] Embedding Distance Calculation

Script to run:
demo_train.py

Purpose:
Compute pairwise distances between all pretrained embeddings.

Input Files:
(A) Metadata file:
./data/train_aug2_genus_s2000_taxid.csv

(B) Embedding directory:
    ./data/datasets/pretrained_embeddings

    Note:
    - Update pretrained embedding file paths in:
        1) load_esm() in ./src/CLEAN/utils.py
        2) __getitem__() in Triplet_dataset_with_mine_EC
    - Embeddings must be generated from Orthoformer.


Output:
Pairwise distances stored in:
./data/distance_map/

[ Step 2 ] Triplet Model Training

Script to run:
train-triplet.py

Purpose:
Train a CLEAN-based model for taxonomy classification.

Inputs:
(A) Metadata file:
./data/train_aug2_genus_s2000_taxid.csv

(B) Embedding directory:
    ./data/datasets/pretrained_embeddings

(C) Distance maps:
    ./data/distance_map/


Output:
Trained model saved in:
./data/model/

### 2. TAXONOMY INFERENCE
[ Step 3 ] Predict Taxonomic ID

Script to run:
inference.py

Purpose:
Predict the taxonomic ID for a pretrained embedding (e.g., genus level).

Input Files:
(A) Metadata file:
./data/UHGG2_5genus.csv

(B) Embedding directory:
    ./data/datasets/pretrained_embeddings

    Note:
    - Update pretrained embedding path in:
        load_esm() in ./src/CLEAN/utils.py
    - Embeddings must be generated from Orthoformer.


Output:
Prediction results written to:
./results/ (CSV files)
