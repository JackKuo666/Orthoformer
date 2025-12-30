import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.CLEAN.infer import infer_maxsep, infer_get_embedding, infer_maxsep_from_embeddings
target = 'genus'
# train_data = f"train_aug2_{target}_s2000_taxid_v7_2048"
train_data = f"train_aug2_genus_s2000_taxid"
# test_data = "new"
# test_data = "UHGG2_1genus5speciese"
test_data = "UHGG2_5genus"
# test_data = "UHGG2_ecoli_strain_50"
# test_data = f"val_aug2_{target}_s2000_taxid"
# test_data = f"union_accession_3tools_3417_{target}"
# test_data = f"common_train_test_samples_aug2_{target}_s2000_taxid"
# test_data = f"test_only_samples_aug2_{target}_s2000_taxid"
# test_data = f"4toolsUnion_accession_9433_species"
# test_data = f"test_samples_aug2_species_s20000_taxid"
# test_data = f"train_aug2_species_s20000_taxid"
infer_maxsep(train_data, test_data, report_metrics=True, pretrained=False,model_name=train_data)
# infer_maxsep_from_embeddings(train_data, test_data, report_metrics=True, pretrained=False,model_name=train_data)
# infer_get_embedding(train_data, test_data, report_metrics=True, pretrained=False,model_name=train_data,output_dir='/root/project/data/UHGG2/UHGG2_5genus/embeddings_3m_v10_ft/')