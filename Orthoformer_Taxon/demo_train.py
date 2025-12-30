


# from src.CLEAN.utils import mutate_single_seq_ECs, retrive_esm1b_embedding, compute_esm_distance

# # train_file = "genus_sampled_fold1_train"
# train_file = "train_aug2_genus_s2000_taxid_v7_2048"

# # train_fasta_file_genus = mutate_single_seq_ECs(train_file)

# # retrive_esm1b_embedding(train_fasta_file)
 
# compute_esm_distance(train_file)

import argparse
from src.CLEAN.utils import mutate_single_seq_ECs, retrive_esm1b_embedding, compute_esm_distance

def main():
    parser = argparse.ArgumentParser(description='处理训练文件的ESM相关操作')
    parser.add_argument('--train_file', type=str, 
                       default='train_aug2_genus_s2000_taxid_v20_2048',
                       help='训练文件名（默认：train_aug2_genus_s2000_taxid_v20_2048）')
    
    args = parser.parse_args()
    train_file = args.train_file
    
    # 使用参数执行相应的函数
    # train_fasta_file_genus = mutate_single_seq_ECs(train_file)
    # retrive_esm1b_embedding(train_fasta_file)
    compute_esm_distance(train_file)

if __name__ == "__main__":
    main()