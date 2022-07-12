

def partition_pairs(simap_dict, pairs):
    only_partition_0_pos = []
    only_partition_1_pos = []
    both_partitions_pos = []
    only_partition_0_neg = []
    only_partition_1_neg = []
    both_partitions_neg = []
    for pair in pairs:
        if pair[0] in simap_dict.keys() and pair[1] in simap_dict.keys():
            if simap_dict[pair[0]] == 0 and simap_dict[pair[1]] == 0:
                if pair[2] == '1':
                    only_partition_0_pos.append(pair)
                else:
                    only_partition_0_neg.append(pair)
            elif simap_dict[pair[0]] == 1 and simap_dict[pair[1]] == 1:
                if pair[2] == '1':
                    only_partition_1_pos.append(pair)
                else:
                    only_partition_1_neg.append(pair)
            else:
                if pair[2] == '1':
                    both_partitions_pos.append(pair)
                else:
                    both_partitions_neg.append(pair)
    return only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg


def adapt_sizes(pos, neg, partition_dict, partition, all_pairs):
    import random
    pos_len = len(pos)
    neg_len = len(neg)
    if neg_len > pos_len:
        # randomly drop some negative samples
        print(f'randomly dropping negatives ({pos_len} positives, {neg_len} negatives)... ')
        to_delete = set(random.sample(range(len(neg)), neg_len - pos_len))
        neg = [x for i, x in enumerate(neg) if not i in to_delete]
    elif pos_len > neg_len:
        # sample neg negatives
        print(f'sampling more negatives ({pos_len} positives, {neg_len} negatives)...')
        if partition == 0:
            candidates = [key for key, value in partition_dict.items() if value == 0]
        elif partition == 1:
            candidates = [key for key, value in partition_dict.items() if value == 1]
        else:
            candidates = list(partition_dict.keys())
        while pos_len > neg_len:
            prot1 = random.choice(tuple(candidates))
            prot1_list = [pair[0] for pair in all_pairs if pair[1] == prot1] + [pair[1] for pair in all_pairs if
                                                                                pair[0] == prot1]
            while len(prot1_list) == 0:
                prot1 = random.choice(tuple(candidates))
                prot1_list = [pair[0] for pair in all_pairs if pair[1] == prot1] + [pair[1] for pair in all_pairs if
                                                                                    pair[0] == prot1]
            prot2 = random.choice(tuple(candidates))
            prot2_list = [pair[0] for pair in all_pairs if pair[1] == prot2] + [pair[1] for pair in all_pairs if
                                                                                pair[0] == prot2]
            while prot1 == prot2 or prot2 in prot1_list or len(prot2_list) == 0:
                prot2 = random.choice(tuple(candidates))
                prot2_list = [pair[0] for pair in all_pairs if pair[1] == prot2] + [pair[1] for pair in all_pairs if
                                                                                    pair[0] == prot2]
            neg.append([prot1, prot2, '0'])
            neg_len += 1
    return pos, neg


def rearrange_deepFE_dataset(yeast=True, sprint=False):
    import numpy as np
    import pandas as pd
    from rewrite_utils_DeepFE import read_NO, write_deepFE
    from rewrite_utils_SPRINT import write_SPRINT
    if yeast:
        file_1 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/11188/positive/Protein_A.txt'
        file_2 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/11188/positive/Protein_B.txt'
        file_3 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/11188/negative/Protein_A.txt'
        file_4 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/11188/negative/Protein_B.txt'
    else:
        file_1 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/human/positive/Protein_A.txt'
        file_2 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/human/positive/Protein_B.txt'
        file_3 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/human/negative/Protein_A.txt'
        file_4 = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/human/negative/Protein_B.txt'
    #  index for protein
    header_dict = {}
    seq_dict = {}
    pos_NO_A, header_dict_tmp, seq_dict_tmp = read_NO(file_1)
    header_dict.update(header_dict_tmp)
    seq_dict.update(seq_dict_tmp)
    pos_NO_B, header_dict_tmp, seq_dict_tmp = read_NO(file_2)
    header_dict.update(header_dict_tmp)
    seq_dict.update(seq_dict_tmp)
    neg_NO_A, header_dict_tmp, seq_dict_tmp = read_NO(file_3)
    header_dict.update(header_dict_tmp)
    seq_dict.update(seq_dict_tmp)
    neg_NO_B, header_dict_tmp, seq_dict_tmp = read_NO(file_4)
    header_dict.update(header_dict_tmp)
    seq_dict.update(seq_dict_tmp)

    # all pairs
    pairs = []
    for i in range(len(pos_NO_A)):
        pairs.append([pos_NO_A[i], pos_NO_B[i], 1])
    for i in range(len(neg_NO_A)):
        pairs.append([neg_NO_A[i], neg_NO_B[i], 0])
    pairs = np.array(pairs)

    if yeast:
        simap_dict = pd.read_csv(
            'network_data/SIMAP2/yeast_networks/only_yeast_partition_nodelist.txt',
            index_col=0, squeeze=True, sep='\t').to_dict()
    else:
        simap_dict = pd.read_csv(
            'network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt',
            index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, pairs)
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(only_partition_0_pos, only_partition_0_neg, simap_dict, 0,
                                                             pairs)
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             pairs)
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           pairs)
    if not sprint:
        print('writing positive files: only partition 0 ...')
        if yeast:
            folder = '11188'
        else:
            folder = 'human'
        write_deepFE(
            pathA=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/positive/Protein_A_0.txt',
            pathB=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/positive/Protein_B_0.txt',
            data=only_partition_0_pos, header_dict=header_dict, seq_dict=seq_dict)
        print('writing negative files: only partition 0 ...')
        write_deepFE(
            pathA=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/negative/Protein_A_0.txt',
            pathB=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/negative/Protein_B_0.txt',
            data=only_partition_0_neg, header_dict=header_dict, seq_dict=seq_dict)
        print('writing positive files: only partition 1 ...')
        write_deepFE(
            pathA=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/positive/Protein_A_1.txt',
            pathB=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/positive/Protein_B_1.txt',
            data=only_partition_1_pos, header_dict=header_dict, seq_dict=seq_dict)
        print('writing negative files: only partition 1 ...')
        write_deepFE(
            pathA=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/negative/Protein_A_1.txt',
            pathB=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/negative/Protein_B_1.txt',
            data=only_partition_1_neg, header_dict=header_dict, seq_dict=seq_dict)
        print('writing positive files: both partitions ...')
        write_deepFE(
            pathA=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/positive/Protein_A_both.txt',
            pathB=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/positive/Protein_B_both.txt',
            data=both_partitions_pos, header_dict=header_dict, seq_dict=seq_dict)
        print('writing negative files: both partitions ...')
        write_deepFE(
            pathA=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/negative/Protein_A_both.txt',
            pathB=f'/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/{folder}/negative/Protein_B_both.txt',
            data=both_partitions_neg, header_dict=header_dict, seq_dict=seq_dict)
    else:
        if yeast:
            prefix='guo'
        else:
            prefix='huang'
        print(f'writing positive files: only partition 0: {len(only_partition_0_pos)} proteins...')
        write_SPRINT(path=f'algorithms/SPRINT/data/{prefix}_partition_0_pos.txt',
            data=only_partition_0_pos)
        print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins...')
        write_SPRINT(path=f'algorithms/SPRINT/data/{prefix}_partition_0_neg.txt',
                                 data=only_partition_0_neg)
        print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/{prefix}_partition_1_pos.txt',
                                 data=only_partition_1_pos)
        print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/{prefix}_partition_1_neg.txt',
                                 data=only_partition_1_neg)
        print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/{prefix}_partition_both_pos.txt',
                                 data=both_partitions_pos)
        print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/{prefix}_partition_both_neg.txt',
                                 data=both_partitions_neg)


def rearrange_deepPPI_dataset(sprint=False):
    from rewrite_utils_DeepPPI import write_deepPPI
    import pandas as pd
    from rewrite_utils_SPRINT import write_SPRINT
    seq_dict = dict()
    pairs = []
    file = open('algorithms/DeepPPI/data/full_data.txt', 'r')
    for line in file:
        all_info = line.rstrip('\n').split(' ')
        id_1 = all_info[0]
        id_2 = all_info[1]
        seq_1 = all_info[2]
        seq_2 = all_info[3]
        label = all_info[4]
        if len(seq_1) < 1166 and len(seq_2) < 1166:
            seq_dict[id_1] = seq_1
            seq_dict[id_2] = seq_2
            pairs.append([id_1, id_2, label])
    file.close()
    print(f'{len(pairs)} PPIs!')
    simap_dict = pd.read_csv(
        'network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt',
        index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, pairs)

    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg, partition_dict=simap_dict, partition=0,
                                                             all_pairs=pairs)
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(pos=only_partition_1_pos, neg=only_partition_1_neg,
                                                             partition_dict=simap_dict, partition=1,
                                                             all_pairs=pairs)
    both_partitions_pos, both_partitions_neg = adapt_sizes(pos=both_partitions_pos, neg=both_partitions_neg,
                                                           partition_dict=simap_dict, partition=-1,
                                                           all_pairs=pairs)
    if not sprint:
        only_partition_0_pos.extend(only_partition_0_neg)
        write_deepPPI(path='algorithms/DeepPPI/data/partition0.txt', all_pairs=only_partition_0_pos, seq_dict=seq_dict)

        only_partition_1_pos.extend(only_partition_1_neg)
        write_deepPPI(path='algorithms/DeepPPI/data/partition1.txt',
                      all_pairs=only_partition_1_pos, seq_dict=seq_dict)

        both_partitions_pos.extend(both_partitions_neg)
        write_deepPPI(path='algorithms/DeepPPI/data/both_partitions.txt',
                      all_pairs=both_partitions_pos, seq_dict=seq_dict)
    else:
        print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/richoux_partition_0_pos.txt',
                     data=only_partition_0_pos)
        print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/richoux_partition_0_neg.txt',
                     data=only_partition_0_neg)
        print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/richoux_partition_1_pos.txt',
                     data=only_partition_1_pos)
        print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/richoux_partition_1_neg.txt',
                     data=only_partition_1_neg)
        print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/richoux_partition_both_pos.txt',
                     data=both_partitions_pos)
        print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/richoux_partition_both_neg.txt',
                     data=both_partitions_neg)


def rearrange_pipr():
    import pandas as pd
    from rewrite_utils_PIPR import write_pipr
    directory = 'algorithms/seq_ppi/yeast/preprocessed/'
    pairs = []
    file = open(directory+'protein.actions.tsv', 'r')
    for line in file:
        all_info = line.rstrip('\n').split('\t')
        id_1 = all_info[0]
        id_2 = all_info[1]
        label = all_info[2]
        pairs.append([id_1, id_2, label])
    file.close()
    print(f'{len(pairs)} PPIs!')
    simap_dict = pd.read_csv(
        'network_data/SIMAP2/yeast_networks/only_yeast_partition_nodelist.txt',
        index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, pairs)

    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg,
                                                             partition_dict=simap_dict, partition=0,
                                                             all_pairs=pairs)
    only_partition_0_pos.extend(only_partition_0_neg)
    write_pipr(path=directory+'protein.actions_partition0.tsv', all_pairs=only_partition_0_pos)

    only_partition_1_pos, only_partition_1_neg = adapt_sizes(pos=only_partition_1_pos, neg=only_partition_1_neg,
                                                             partition_dict=simap_dict, partition=1,
                                                             all_pairs=pairs)
    only_partition_1_pos.extend(only_partition_1_neg)
    write_pipr(path=directory + 'protein.actions_partition1.tsv', all_pairs=only_partition_1_pos)

    both_partitions_pos, both_partitions_neg = adapt_sizes(pos=both_partitions_pos, neg=both_partitions_neg,
                                                             partition_dict=simap_dict, partition=-1,
                                                             all_pairs=pairs)
    both_partitions_pos.extend(both_partitions_neg)
    write_pipr(path=directory + 'protein.actions_both_partitions.tsv', all_pairs=both_partitions_pos)


def rearrange_pan_dataset(sprint=True):
    import pandas as pd
    from rewrite_utils_SPRINT import write_SPRINT
    from algorithms.Custom.load_datasets import make_swissprot_to_dict, iterate_pan
    from tqdm import tqdm
    prefix_dict, seq_dict = make_swissprot_to_dict('network_data/Swissprot/human_swissprot.fasta')
    print('Mapping Protein IDs ...')
    mapping_dict = iterate_pan(prefix_dict, seq_dict, 'Datasets_PPIs/Pan_human_HPRD/SEQ-Supp-ABCD.tsv')
    pairs = []
    lines = open('Datasets_PPIs/Pan_human_HPRD/Supp-AB.tsv', 'r').readlines()
    for line in tqdm(lines):
        if line.startswith('v1'):
            # header
            continue
        else:
            line_split_pan = line.strip().split('\t')
            id0_pan = line_split_pan[0]
            id1_pan = line_split_pan[1]
            label = line_split_pan[2]
            if id0_pan in mapping_dict.keys() and id1_pan in mapping_dict.keys():
                uniprot_id0 = mapping_dict[id0_pan]
                uniprot_id1 = mapping_dict[id1_pan]
                if uniprot_id0 != '' and uniprot_id1 != '':
                    pairs.append([uniprot_id0, uniprot_id1, label])
    print(f'{len(pairs)} PPIs!')
    simap_dict = pd.read_csv(
        'network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt',
        index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, pairs)
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg,
                                                             partition_dict=simap_dict, partition=0,
                                                             all_pairs=pairs)
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             pairs)
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           pairs)
    if not sprint:
        pass
    else:
        print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/pan_partition_0_pos.txt',
                     data=only_partition_0_pos)
        print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/pan_partition_0_neg.txt',
                     data=only_partition_0_neg)
        print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/pan_partition_1_pos.txt',
                     data=only_partition_1_pos)
        print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/pan_partition_1_neg.txt',
                     data=only_partition_1_neg)
        print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/pan_partition_both_pos.txt',
                     data=both_partitions_pos)
        print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/pan_partition_both_neg.txt',
                     data=both_partitions_neg)


def rearrange_du_dataset(sprint=True):
    from rewrite_utils_SPRINT import write_SPRINT
    from tqdm import tqdm
    import pandas as pd
    f = open('Datasets_PPIs/Du_yeast_DIP/SupplementaryS1.csv').readlines()
    ppis = list()
    for line in tqdm(f):
        if line.startswith('proteinA'):
            # header
            continue
        line_split = line.strip().split(',')
        ppis.append(line_split)
    print(f'{len(ppis)} PPIs!')
    simap_dict = pd.read_csv(
        'network_data/SIMAP2/yeast_networks/only_yeast_partition_nodelist.txt',
        index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, ppis)
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg,
                                                             partition_dict=simap_dict, partition=0,
                                                             all_pairs=ppis)
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             ppis)
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           ppis)
    if not sprint:
        pass
    else:
        print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/du_partition_0_pos.txt',
                     data=only_partition_0_pos)
        print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/du_partition_0_neg.txt',
                     data=only_partition_0_neg)
        print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/du_partition_1_pos.txt',
                     data=only_partition_1_pos)
        print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/du_partition_1_neg.txt',
                     data=only_partition_1_neg)
        print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/du_partition_both_pos.txt',
                     data=both_partitions_pos)
        print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
        write_SPRINT(path=f'algorithms/SPRINT/data/du_partition_both_neg.txt',
                     data=both_partitions_neg)

if __name__ == '__main__':
    # rearrange_deepFE_dataset(yeast=False, sprint=True)
    # rearrange_deepPPI_dataset(sprint=True)
    # rearrange_pipr()
    # rearrange_pan_dataset(sprint=True)
    rearrange_du_dataset(sprint=True)
