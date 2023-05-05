from rewrite_utils_SPRINT import write_SPRINT


def partition_pairs(simap_dict, pairs):
    only_partition_0_pos = set()
    only_partition_1_pos = set()
    both_partitions_pos = set()
    only_partition_0_neg = set()
    only_partition_1_neg = set()
    both_partitions_neg = set()
    for pair in pairs:
        if pair[0] in simap_dict.keys() and pair[1] in simap_dict.keys():
            if simap_dict[pair[0]] == 0 and simap_dict[pair[1]] == 0:
                if pair[2] == '1':
                    only_partition_0_pos.add((pair[0], pair[1]))
                else:
                    only_partition_0_neg.add((pair[0], pair[1]))
            elif simap_dict[pair[0]] == 1 and simap_dict[pair[1]] == 1:
                if pair[2] == '1':
                    only_partition_1_pos.add((pair[0], pair[1]))
                else:
                    only_partition_1_neg.add((pair[0], pair[1]))
            else:
                if pair[2] == '1':
                    both_partitions_pos.add((pair[0], pair[1]))
                else:
                    both_partitions_neg.add((pair[0], pair[1]))
    return only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg


def adapt_sizes(pos, neg, partition_dict, partition, all_pairs, factor=1):
    import random
    all_pairs = set((ppi[0], ppi[1], ppi[2]) for ppi in all_pairs)
    # no overlaps between positives and negatives
    pos_ppis = pos.copy()
    pos_ppis = pos_ppis.union({(ppi[1], ppi[0]) for ppi in pos_ppis})
    neg_ppis = neg.copy()
    neg_ppis = neg_ppis.union({(ppi[1], ppi[0]) for ppi in neg_ppis})
    intersect_ppis = pos_ppis.intersection(neg_ppis)
    print(f'Number of overlaps between pos and neg: {len(intersect_ppis)}')
    pos = pos - intersect_ppis
    neg = neg - intersect_ppis
    pos = set((ppi[0], ppi[1], 1) for ppi in pos)
    neg = set((ppi[0], ppi[1], 0) for ppi in neg)
    pos_len = len(pos)
    neg_len = len(neg)
    if neg_len > (pos_len * factor):
        # randomly drop some negative samples
        print(f'randomly dropping negatives ({pos_len} positives, {neg_len} negatives)... ')
        to_delete = set(random.sample(range(len(neg)), neg_len - (factor * pos_len)))
        neg = set(x for i, x in enumerate(neg) if not i in to_delete)
    elif (pos_len * factor) > neg_len:
        # sample neg negatives
        print(f'sampling more negatives ({pos_len} positives, {neg_len} negatives)...')
        if partition == 0:
            candidates = [key for key, value in partition_dict.items() if value == 0]
        elif partition == 1:
            candidates = [key for key, value in partition_dict.items() if value == 1]
        else:
            candidates = list(partition_dict.keys())
        to_generate = (factor * pos_len) - neg_len
        while (pos_len * factor) > neg_len:
            if to_generate % 100 == 0:
                print(f'Still {to_generate} proteins left to generate!')
            prot1 = random.choice(tuple(candidates))
            prot2 = random.choice(tuple(candidates))
            while prot1 == prot2 or (prot1, prot2) in all_pairs or (prot2, prot1) in all_pairs or (prot2, prot1) in neg:
                prot2 = random.choice(tuple(candidates))
            neg.add((prot1, prot2, 0))
            neg_len = len(neg)
            to_generate = (factor * pos_len) - neg_len
    return pos, neg


def rearrange_guo_huang_dataset(guo=True):
    import numpy as np
    import pandas as pd
    from rewrite_utils_DeepFE import read_NO
    if guo:
        file_1 = 'algorithms/DeepFE-PPI/dataset/11188/positive/Protein_A.txt'
        file_2 = 'algorithms/DeepFE-PPI/dataset/11188/positive/Protein_B.txt'
        file_3 = 'algorithms/DeepFE-PPI/dataset/11188/negative/Protein_A.txt'
        file_4 = 'algorithms/DeepFE-PPI/dataset/11188/negative/Protein_B.txt'
    else:
        file_1 = 'algorithms/DeepFE-PPI/dataset/human/positive/Protein_A.txt'
        file_2 = 'algorithms/DeepFE-PPI/dataset/human/positive/Protein_B.txt'
        file_3 = 'algorithms/DeepFE-PPI/dataset/human/negative/Protein_A.txt'
        file_4 = 'algorithms/DeepFE-PPI/dataset/human/negative/Protein_B.txt'
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

    if guo:
        simap_dict = pd.read_csv(
            'network_data/SIMAP2/yeast_networks/only_yeast_partition_nodelist.txt',
            index_col=0, squeeze=True, sep='\t').to_dict()
    else:
        simap_dict = pd.read_csv(
            'network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt',
            index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, pairs)
    print('Cleaning and balancing partition 0 ...')
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(only_partition_0_pos, only_partition_0_neg, simap_dict, 0,
                                                             pairs)
    print('Cleaning and balancing partition 1 ...')
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             pairs)
    print('Cleaning and balancing partition both ...')
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           pairs)
    if guo:
        prefix='guo'
    else:
        prefix='huang'
    print(f'writing positive files: only partition 0: {len(only_partition_0_pos)} proteins...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/{prefix}_partition_0_pos.txt',
        data=only_partition_0_pos)
    print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/{prefix}_partition_0_neg.txt',
                             data=only_partition_0_neg)
    print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/{prefix}_partition_1_pos.txt',
                             data=only_partition_1_pos)
    print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/{prefix}_partition_1_neg.txt',
                             data=only_partition_1_neg)
    print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/{prefix}_partition_both_pos.txt',
                             data=both_partitions_pos)
    print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/{prefix}_partition_both_neg.txt',
                             data=both_partitions_neg)


def rearrange_richoux_dataset():
    import pandas as pd
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
    print('Cleaning and balancing partition 0 ...')
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg, partition_dict=simap_dict, partition=0,
                                                             all_pairs=pairs)
    print('Cleaning and balancing partition 1 ...')
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(pos=only_partition_1_pos, neg=only_partition_1_neg,
                                                             partition_dict=simap_dict, partition=1,
                                                             all_pairs=pairs)
    print('Cleaning and balancing partition both ...')
    both_partitions_pos, both_partitions_neg = adapt_sizes(pos=both_partitions_pos, neg=both_partitions_neg,
                                                           partition_dict=simap_dict, partition=-1,
                                                           all_pairs=pairs)
    print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/richoux_partition_0_pos.txt',
                 data=only_partition_0_pos)
    print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/richoux_partition_0_neg.txt',
                 data=only_partition_0_neg)
    print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/richoux_partition_1_pos.txt',
                 data=only_partition_1_pos)
    print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/richoux_partition_1_neg.txt',
                 data=only_partition_1_neg)
    print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/richoux_partition_both_pos.txt',
                 data=both_partitions_pos)
    print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/richoux_partition_both_neg.txt',
                 data=both_partitions_neg)


def rearrange_pan_dataset():
    import pandas as pd
    from algorithms.Custom.load_datasets import make_swissprot_to_dict
    from algorithms.SPRINT.create_SPRINT_datasets import iterate_pan
    from tqdm import tqdm
    prefix_dict, seq_dict = make_swissprot_to_dict('Datasets_PPIs/SwissProt/human_swissprot.fasta')
    print('Mapping Protein IDs ...')
    mapping_dict = iterate_pan(prefix_dict, seq_dict, 'algorithms/seq_ppi/sun/preprocessed/SEQ-Supp-ABCD.tsv')
    pairs = []
    lines = open('algorithms/seq_ppi/sun/preprocessed/Supp-AB.tsv', 'r').readlines()
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
    print('Cleaning and balancing partition 0 ...')
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg,
                                                             partition_dict=simap_dict, partition=0,
                                                             all_pairs=pairs)
    print('Cleaning and balancing partition 1 ...')
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             pairs)
    print('Cleaning and balancing partition both ...')
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           pairs)
    print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/pan_partition_0_pos.txt',
                 data=only_partition_0_pos)
    print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/pan_partition_0_neg.txt',
                 data=only_partition_0_neg)
    print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/pan_partition_1_pos.txt',
                 data=only_partition_1_pos)
    print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/pan_partition_1_neg.txt',
                 data=only_partition_1_neg)
    print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/pan_partition_both_pos.txt',
                 data=both_partitions_pos)
    print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/pan_partition_both_neg.txt',
                 data=both_partitions_neg)


def rearrange_du_dataset():
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
    print('Cleaning and balancing partition 0 ...')
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg,
                                                             partition_dict=simap_dict, partition=0,
                                                             all_pairs=ppis)
    print('Cleaning and balancing partition 1 ...')
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             ppis)
    print('Cleaning and balancing partition both ...')
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           ppis)

    print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/du_partition_0_pos.txt',
                 data=only_partition_0_pos)
    print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/du_partition_0_neg.txt',
                 data=only_partition_0_neg)
    print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/du_partition_1_pos.txt',
                 data=only_partition_1_pos)
    print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/du_partition_1_neg.txt',
                 data=only_partition_1_neg)
    print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/du_partition_both_pos.txt',
                 data=both_partitions_pos)
    print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/du_partition_both_neg.txt',
                 data=both_partitions_neg)


def rearrange_dscript_dataset():
    import pandas as pd
    from algorithms.Custom.load_datasets import make_swissprot_to_dict
    from algorithms.SPRINT.create_SPRINT_datasets import iterate_fasta, process_dscript
    prefix_dict, seq_dict = make_swissprot_to_dict('Datasets_PPIs/SwissProt/human_swissprot.fasta')
    print('Mapping Protein IDs ...')
    mapping_dict = iterate_fasta(prefix_dict, seq_dict, 'algorithms/D-SCRIPT-main/dscript-data/seqs/human.fasta')
    pairs = process_dscript('algorithms/D-SCRIPT-main/dscript-data/pairs/human_train.tsv', mapping_dict)
    ppis_test = process_dscript('algorithms/D-SCRIPT-main/dscript-data/pairs/human_test.tsv', mapping_dict)
    pairs.extend(ppis_test)
    print(f'{len(pairs)} PPIs!')
    simap_dict = pd.read_csv(
        'network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt',
        index_col=0, squeeze=True, sep='\t').to_dict()

    only_partition_0_pos, only_partition_1_pos, both_partitions_pos, only_partition_0_neg, only_partition_1_neg, both_partitions_neg = partition_pairs(
        simap_dict, pairs)
    print('Cleaning and balancing partition 0 ...')
    only_partition_0_pos, only_partition_0_neg = adapt_sizes(pos=only_partition_0_pos, neg=only_partition_0_neg,
                                                             partition_dict=simap_dict, partition=0,
                                                             all_pairs=pairs, factor=10)
    print('Cleaning and balancing partition 1 ...')
    only_partition_1_pos, only_partition_1_neg = adapt_sizes(only_partition_1_pos, only_partition_1_neg, simap_dict, 1,
                                                             pairs, factor=10)
    print('Cleaning and balancing partition both ...')
    both_partitions_pos, both_partitions_neg = adapt_sizes(both_partitions_pos, both_partitions_neg, simap_dict, -1,
                                                           pairs, factor=10)
    print(f'writing positive files for SPRINT: only partition 0: {len(only_partition_0_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/dscript_partition_0_pos.txt',
                 data=only_partition_0_pos)
    print(f'writing negative files: only partition 0: {len(only_partition_0_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/dscript_partition_0_neg.txt',
                 data=only_partition_0_neg)
    print(f'writing positive files: only partition 1: {len(only_partition_1_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/dscript_partition_1_pos.txt',
                 data=only_partition_1_pos)
    print(f'writing negative files: only partition 1: {len(only_partition_1_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/dscript_partition_1_neg.txt',
                 data=only_partition_1_neg)
    print(f'writing positive files: both partitions: {len(both_partitions_pos)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/dscript_partition_both_pos.txt',
                 data=both_partitions_pos)
    print(f'writing negative files: both partitions: {len(both_partitions_neg)} proteins ...')
    write_SPRINT(path=f'algorithms/SPRINT/data/partitions/dscript_partition_both_neg.txt',
                 data=both_partitions_neg)



if __name__ == '__main__':
    print('############################ GUO DATASET ############################')
    rearrange_guo_huang_dataset(guo=True)
    print('############################ HUANG DATASET ############################')
    rearrange_guo_huang_dataset(guo=False)
    print('############################ RICHOUX DATASET ############################')
    rearrange_richoux_dataset()
    print('############################ PAN DATASET ############################')
    rearrange_pan_dataset()
    print('############################ DU DATASET ############################')
    rearrange_du_dataset()
    print('############################ DSCRIPT DATASET ############################')
    rearrange_dscript_dataset()
