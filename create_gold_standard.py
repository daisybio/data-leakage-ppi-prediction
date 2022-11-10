import pandas as pd
from algorithms.Custom.load_datasets import make_swissprot_to_dict

def read_kahip():
    path_to_list_input = 'network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt'
    path_to_partition = 'network_data/SIMAP2/human_networks/only_human_partition_3_bitscore.txt'
    print('Reading input ... ')
    uniprot_ids = pd.read_csv(path_to_list_input, sep='\t')['Node']
    partition_map = dict()
    print('Parsing KaHIP output ...')
    node = 0
    with open(path_to_partition, 'r') as f:
        for line in f:
            partition_map[uniprot_ids[node]] = int(line.strip())
            node += 1
    return partition_map


def sort_hippie(partition_map):
    all_ppis = set()
    intra_0 = set()
    intra_1 = set()
    intra_2 = set()
    inter_01 = set()
    inter_02 = set()
    inter_12 = set()
    with open('Datasets_PPIs/Hippiev2.3/hippie_PPIs.tsv') as pos_ppis:
        for line in pos_ppis:
            if line.startswith('ID_A'):
                continue
            else:
                id_a, id_b, confidence = line.strip().split('\t')
                all_ppis.add((id_a, id_b))
                block_a = partition_map.get(id_a)
                block_b = partition_map.get(id_b)
                if block_a == 0 and block_b == 0:
                    intra_0.add((id_a, id_b))
                elif block_a == 1 and block_b == 1:
                    intra_1.add((id_a, id_b))
                elif block_a == 2 and block_b == 2:
                    intra_2.add((id_a, id_b))
                elif (block_a == 0 and block_b == 1) or (block_a == 1 and block_b == 0):
                    inter_01.add((id_a, id_b))
                elif (block_a == 0 and block_b == 2) or (block_a == 2 and block_b == 0):
                    inter_02.add((id_a, id_b))
                elif (block_a == 1 and block_b == 2) or (block_a == 2 and block_b == 1):
                    inter_12.add((id_a, id_b))
    print(f'Size Intra-0: {len(intra_0)}\nSize Intra-1: {len(intra_1)}\nSize Intra-2: {len(intra_2)}')
    print(f'Size Inter-01: {len(inter_01)}\nSize Inter-02: {len(inter_02)}\nSize Inter-12: {len(inter_12)}')
    return all_ppis, intra_0, intra_1, intra_2, inter_01, inter_02, inter_12


def sort_negatome(partition_map):
    intra_0 = set()
    intra_1 = set()
    intra_2 = set()
    unique_prots = set()
    with open('Datasets_PPIs/Negatomev2.0/combined_stringent.txt') as neg_ppis:
        for line in neg_ppis:
            id_a, id_b = line.strip().split('\t')
            block_a = partition_map.get(id_a)
            block_b = partition_map.get(id_b)
            unique_prots.add(id_a)
            unique_prots.add(id_b)
            if block_a == 0 and block_b == 0:
                intra_0.add((id_a, id_b))
            elif block_a == 1 and block_b == 1:
                intra_1.add((id_a, id_b))
            elif block_a == 2 and block_b == 2:
                intra_2.add((id_a, id_b))
    print(f'Size Intra-0: {len(intra_0)}\nSize Intra-1: {len(intra_1)}\nSize Intra-2: {len(intra_2)}')
    prefix_dict, seq_dict = make_swissprot_to_dict('Datasets_PPIs/SwissProt/human_swissprot.fasta')
    print(f'{len(unique_prots)} unique proteins in the Negatome')
    with open('Datasets_PPIs/Negatomev2.0/negatome.fasta', 'w') as output:
        for prot in unique_prots:
            if seq_dict.get(prot) is not None:
                output.write(f'>{prot}\n')
                output.write(seq_dict.get(prot)+'\n')
    return intra_0, intra_1, intra_2


def export_fasta(block, seq_dict, filename):
    unique_proteins = set([ppi[0] for ppi in block])
    unique_proteins = unique_proteins.union([ppi[1] for ppi in block])
    with open(filename, 'w') as output:
        for prot in unique_proteins:
            if seq_dict.get(prot) is not None:
                output.write(f'>{prot}\n')
                output.write(seq_dict.get(prot)+'\n')


def export_block(block, filename):
    with open(filename, 'w') as output:
        for ppi in block:
            output.write(f'{ppi[0]} {ppi[1]}\n')


def sample_negatives(block, all_ppis):
    import random
    size = 0
    neg_ppis = set()
    # should lead to power law distribution
    candidates = [ppi[0] for ppi in block]
    candidates.extend([ppi[1] for ppi in block])
    while size < len(block):
        prot1 = random.choice(tuple(candidates))
        prot2 = random.choice(tuple(candidates))
        while prot1 == prot2 or (prot1, prot2) in all_ppis or (prot2, prot1) in all_ppis or (prot2, prot1) in neg_ppis:
            prot2 = random.choice(tuple(candidates))
        neg_ppis.add((prot1, prot2))
        if size % 1000 == 0:
            print(size)
        size = len(neg_ppis)
    return neg_ppis

if __name__ == '__main__':
    partition_map = read_kahip()
    all_ppis, intra_0, intra_1, intra_2, inter_01, inter_02, inter_12 = sort_hippie(partition_map)
    neg_intra0 = sample_negatives(intra_0, all_ppis)
    export_block(intra_0, 'Datasets_PPIs/Hippiev2.3/Intra0_pos.txt')
    export_block(neg_intra0, 'Datasets_PPIs/Hippiev2.3/Intra0_neg.txt')
    neg_intra_1 = sample_negatives(intra_1, all_ppis)
    export_block(intra_1, 'Datasets_PPIs/Hippiev2.3/Intra1_pos.txt')
    export_block(neg_intra_1, 'Datasets_PPIs/Hippiev2.3/Intra1_neg.txt')
    neg_intra_2 = sample_negatives(intra_2, all_ppis)
    export_block(intra_2, 'Datasets_PPIs/Hippiev2.3/Intra2_pos.txt')
    export_block(neg_intra_2, 'Datasets_PPIs/Hippiev2.3/Intra2_neg.txt')
    #prefix_dict, seq_dict = make_swissprot_to_dict('Datasets_PPIs/SwissProt/human_swissprot.fasta')
    #export_fasta(intra_0, seq_dict, 'Datasets_PPIs/Hippiev2.3/Intra_0.fasta')
    #export_fasta(intra_1, seq_dict, 'Datasets_PPIs/Hippiev2.3/Intra_1.fasta')
    #export_fasta(intra_2, seq_dict, 'Datasets_PPIs/Hippiev2.3/Intra_2.fasta')

