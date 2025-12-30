import torch
import random
from .utils import format_esm
from tqdm import tqdm

# def find_first_non_zero_distance(data):
#     for index, (name, distance) in enumerate(data):
#         if distance != 0:
#             return index
#     return None 

def mine_hard_negative(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    print("Mining hard negatives:")
    for _, target in tqdm(enumerate(ecs), total=len(ecs)):
        sorted_orders = sorted(dist_map[target].items(), key=lambda x: x[1], reverse=False)
        assert sorted_orders != None, "all clusters have zero distances!"
        neg_ecs_start_index = find_first_non_zero_distance(sorted_orders)
        closest_negatives = sorted_orders[neg_ecs_start_index:neg_ecs_start_index + knn]
        freq = [1/i[1] for i in closest_negatives]
        neg_ecs = [i[0] for i in closest_negatives]        
        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative

def find_first_non_zero_distance(sorted_orders):
    """Helper function to find the first non-zero distance"""
    for idx, (_, dist) in enumerate(sorted_orders):
        if dist > 0:
            return idx
    return 0  # If all distances are zero

def mine_hard_negative_cuda(dist_map, knn=10, device='cuda'):
    # Create a mapping from EC names (strings) to integer indices
    ecs = list(dist_map.keys())
    ec_to_index = {ec: idx for idx, ec in enumerate(ecs)}
    index_to_ec = {idx: ec for idx, ec in enumerate(ecs)}  # Reverse mapping from index to EC
    
    # Prepare lists for ECs and their corresponding distances
    ecs_list = []
    distances_list = []

    for target in ecs:
        target_ec = target
        distances = dist_map[target]
        
        # Extract distances and ECs and sort them
        sorted_orders = sorted(distances.items(), key=lambda x: x[1], reverse=False)
        
        ecs_list.append([ec_to_index[i[0]] for i in sorted_orders])  # Convert ECs to indices
        distances_list.append([i[1] for i in sorted_orders])  # Distances as numerical values

    # Convert lists into tensors
    ecs_tensor = torch.tensor(ecs_list, dtype=torch.int32, device=device)
    distances_tensor = torch.tensor(distances_list, dtype=torch.float32, device=device)

    negative = {}
    print("Mining hard negatives (CUDA):")

    # Iterate over the targets (ecs) to mine hard negatives
    for idx, target in tqdm(enumerate(ecs), total=len(ecs)):
        target_ec = ecs_tensor[idx]  # ECs for the current target
        target_distances = distances_tensor[idx]  # Corresponding distances

        # Find the first non-zero distance index
        neg_ecs_start_index = (target_distances > 0).nonzero(as_tuple=True)[0][0]

        # Get the closest `knn` negatives
        closest_negatives_indices = target_ec[neg_ecs_start_index:neg_ecs_start_index + knn]
        closest_negatives_distances = target_distances[neg_ecs_start_index:neg_ecs_start_index + knn]

        # Frequency is the inverse of distance, higher distance => lower frequency
        freq = 1 / closest_negatives_distances

        # Normalize frequencies using GPU
        freq_tensor = torch.tensor(freq, dtype=torch.float32, device=device)
        normalized_freq = freq_tensor / freq_tensor.sum()

        # Convert indices back to ECs
        closest_negatives = [index_to_ec[idx] for idx in closest_negatives_indices.cpu().numpy()]

        # Store the results
        negative[target] = {
            'weights': normalized_freq.cpu().numpy(),  # Move the result back to CPU
            'negative': closest_negatives
        }

    return negative



def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights']
    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    # pos = id
    # if len(ec_id[pos_ec]) == 1:
    #     return pos + '_' + str(random.randint(0, 9))
    # while pos == id:
    pos = random.choice(ec_id[pos_ec])
    return pos


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        # a = torch.load('./data/esm_data/' + anchor + '.pt')
        # p = torch.load('./data/esm_data/' + pos + '.pt')
        # n = torch.load('./data/esm_data/' + neg + '.pt')
        a = torch.load('/mnt/zzb/default/Workspace/liup/clean_data/embeddings_2048_v18_a2/' + anchor + '.pt')
        p = torch.load('/mnt/zzb/default/Workspace/liup/clean_data/embeddings_2048_v18_a2/' + pos + '.pt')
        n = torch.load('/mnt/zzb/default/Workspace/liup/clean_data/embeddings_2048_v18_a2/' + neg + '.pt')
        return format_esm(a), format_esm(p), format_esm(n)


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load('./data/esm_data/' +
                       anchor + '.pt')).unsqueeze(0)
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = format_esm(torch.load('./data/esm_data/' +
                           pos + '.pt')).unsqueeze(0)
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = format_esm(torch.load('./data/esm_data/' +
                           neg + '.pt')).unsqueeze(0)
            data.append(n)
        return torch.cat(data)
