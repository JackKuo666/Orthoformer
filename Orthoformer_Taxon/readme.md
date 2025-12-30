This implementation builds upon the CLEAN framework (https://github.com/tttianhao/CLEAN), which we have adapted for taxonomy classification tasks. We gratefully acknowledge the contributions of the CLEAN authors.

## Environment Setup
Please configure the environment according to CLEAN's requirements. Refer to the original repository for detailed dependency specifications.

**Usage Instructions:**
1. **Embedding Distance Calculation:** Run `demo_train.py` to compute pairwise distances between all input embeddings. These embeddings should be pretrained using Orthoformer.

2. **Model Training:** Execute `train-triplet.py` to train the CLEAN model on your taxonomy classification task.

3. **Taxonomy Inference:** Use `inference.py` to predict the taxonomic ID for a given pretrained embedding at a specific taxonomic level (e.g., genus as shown in the example).