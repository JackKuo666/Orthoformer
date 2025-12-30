#!/bin/bash

### run xgboost using embeding from Orthoformer
python3 scripts/bacformer_xgb_binary.py -i bacformer_embedding/gideon_Bacillus_or_coccobacillus.csv -o binary -e  embeddings_4000_v2 --output_dir embeddings_4000_v2_Bacillus_or_coccobacillus_prediction

### run xgboost using embeding from Bacformer
python3 scripts/bacformer_xgb_binary.py -i bacformer_embedding/gideon_Bacillus_or_coccobacillus.csv --force -o binary -e embeddings_4000_v2 --output_dir Bacformer_Bacillus_or_coccobacillus_prediction

### run 1d-cnn to identify marker genes
python3 scripts/Weighted_1d-cnn-topk.py 1dcnn --label bacformer_embedding/gideon_Bacillus_or_coccobacillus.csv

### show saliency score
cd salient_plot
python3 Weighted_1d-cnn-inference.py 1dcnn_pred --label raffinose.csv --model_path madin_carbsubs_raffinose_best_cnn1d_model.pt
