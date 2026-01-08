


# from src.CLEAN.utils import mutate_single_seq_ECs, retrive_esm1b_embedding, compute_esm_distance

# # train_file = "genus_sampled_fold1_train"
# train_file = "train_aug2_genus_s2000_taxid_v7_2048"

# # train_fasta_file_genus = mutate_single_seq_ECs(train_file)

# # retrive_esm1b_embedding(train_fasta_file)
 
# compute_esm_distance(train_file)

import argparse
from src.CLEAN.utils import mutate_single_seq_ECs, retrive_esm1b_embedding, compute_esm_distance

def main():
    parser = argparse.ArgumentParser(description='Process ESM-related operations for training files')
    parser.add_argument('--train_file', type=str, 
                       default='train_aug2_genus_s2000_taxid_v20_2048',
                       help='Training file name (default: train_aug2_genus_s2000_taxid_v20_2048)')
    
    args = parser.parse_args()
    train_file = args.train_file
    
    # Execute corresponding function with parameters
    # train_fasta_file_genus = mutate_single_seq_ECs(train_file)
    # retrive_esm1b_embedding(train_fasta_file)
    compute_esm_distance(train_file)

if __name__ == "__main__":
    main()