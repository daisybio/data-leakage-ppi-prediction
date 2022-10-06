import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def process_dataset(path, prots):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            prots.add(line[0])
            prots.add(line[1])
    return prots


def load_dataset(folder, dataset):
    prots_tr = process_dataset(f'../algorithms/SPRINT/data/{folder}/{dataset}_train_pos.txt', set())
    prots_tr = process_dataset(f'../algorithms/SPRINT/data/{folder}/{dataset}_train_neg.txt', prots_tr)
    prots_te = process_dataset(f'../algorithms/SPRINT/data/{folder}/{dataset}_test_pos.txt', set())
    prots_te = process_dataset(f'../algorithms/SPRINT/data/{folder}/{dataset}_test_neg.txt', prots_te)
    return prots_tr, prots_te

def load_partition_dataset(dataset, partition_tr, partition_te):
    prots_tr = process_dataset(f'../algorithms/SPRINT/data/partitions/{dataset}_partition_{partition_tr}_pos.txt', set())
    prots_tr = process_dataset(f'../algorithms/SPRINT/data/partitions/{dataset}_partition_{partition_tr}_neg.txt', prots_tr)
    prots_te = process_dataset(f'../algorithms/SPRINT/data/partitions/{dataset}_partition_{partition_te}_pos.txt', set())
    prots_te = process_dataset(f'../algorithms/SPRINT/data/partitions/{dataset}_partition_{partition_te}_neg.txt', prots_te)
    return prots_tr, prots_te


def calculate_bitscores(prots_tr, prots_te, bitscores):
    mat = np.ones(shape=(len(prots_tr), len(prots_te)))
    i = 0
    for tr_prot in prots_tr:
        if i % 1000 == 0:
            print(i)
        j = 0
        for te_prot in prots_te:
            if bitscores.get(f'{tr_prot}_{te_prot}') is not None:
                mat[i, j] = bitscores.get(f'{tr_prot}_{te_prot}')
            elif tr_prot == te_prot:
                mat[i, j] = 1000
            else:
                mat[i, j] = 0.0
            j += 1
        i += 1
    print(f'Non-zero bitscores: {np.count_nonzero(mat)}/{len(prots_tr) * len(prots_te)} = {100 * np.count_nonzero(mat) / (len(prots_tr) * len(prots_te))}%')
    mat = np.hstack((np.array(list(prots_tr))[:, np.newaxis], mat))
    mat = np.vstack((np.array(['#'] + list(prots_te))[np.newaxis, :], mat))
    return mat

#node_lines = open('../algorithms/Custom/data/yeast.nodelist', 'r').readlines()
#node_dict = {x[0]: x[1] for x in (line.strip().split('\t') for line in node_lines)}
#lines = open('../algorithms/Custom/data/yeast.edgelist', 'r').readlines()
#yeast_bitscores = {f'{node_dict[x[0]]}_{node_dict[x[1]]}': x[2] for x in (line.strip().split(' ') for line in lines)}
#yeast_bitscores_inv = {f'{node_dict[x[1]]}_{node_dict[x[0]]}': x[2] for x in (line.strip().split(' ') for line in lines)}
#yeast_bitscores.update(yeast_bitscores_inv)
#for dataset in ['guo', 'du']:
#    print(dataset)
#    prots_tr, prots_te = load_dataset('original', dataset)
#    print(f'Unique proteins in training: {len(prots_tr)}, testing: {len(prots_te)}')
#    print(f'Overlap between training and testing: {len(prots_tr.intersection(prots_te))}')
#    print(f'Unique proteins in whole set: {len(prots_tr.union(prots_te))}')
#    yeast_mat = calculate_bitscores(prots_tr, prots_te, yeast_bitscores)
#    np.savetxt(f'../network_data/SIMAP2/datasets/{dataset}.tsv', yeast_mat, delimiter='\t', fmt='%s')

#node_lines = open('../algorithms/Custom/data/human.nodelist', 'r').readlines()
#node_dict = {x[0]: x[1] for x in (line.strip().split('\t') for line in node_lines)}
#lines = open('../algorithms/Custom/data/human.edgelist', 'r').readlines()
#human_bitscores = {f'{node_dict[x[0]]}_{node_dict[x[1]]}': x[2] for x in (line.strip().split(' ') for line in lines)}
#human_bitscores_inv = {f'{node_dict[x[1]]}_{node_dict[x[0]]}': x[2] for x in (line.strip().split(' ') for line in lines)}
#human_bitscores.update(human_bitscores_inv)

for test in ['original', 'rewired']:
    print(f'############ {test} ############')
    for dataset in ['huang', 'guo', 'du' ,'pan', 'richoux_regular', 'richoux_strict']:
        print(f'############ {dataset} ############')
        prots_tr, prots_te = load_dataset(test, dataset)
        len_tr = len(prots_tr)
        len_te = len(prots_te)
        whole_set = len(prots_tr.union(prots_te))
        intersect_set = len(prots_tr.intersection(prots_te))
        print(f'Unique proteins in whole set: {whole_set}')
        print(f'Unique proteins in training: {len_tr}={round(100*len_tr/whole_set, 1)}%, testing: {len_te}={round(100*len_te/whole_set, 1)}%')
        print(f'Overlap between training and testing: {intersect_set}={round(100*intersect_set/len_tr, 1)}% of training, {round(100*intersect_set/len_te, 1)}% of testing')
        #human_mat = calculate_bitscores(prots_tr, prots_te, human_bitscores)
        #np.savetxt(f'../network_data/SIMAP2/datasets/{dataset}.tsv', human_mat, delimiter='\t', fmt='%s')

for dataset in ['huang', 'guo', 'du' ,'pan', 'richoux']:
    print(f'############ {dataset} ############')
    for partition in ['both_0', 'both_1', '0_1']:
        print(f'############ Partition {partition} ############')
        partition = partition.split('_')
        prots_tr, prots_te = load_partition_dataset(dataset, partition[0], partition[1])
        len_tr = len(prots_tr)
        len_te = len(prots_te)
        whole_set = len(prots_tr.union(prots_te))
        intersect_set = len(prots_tr.intersection(prots_te))
        print(f'Unique proteins in whole set: {whole_set}')
        print(
            f'Unique proteins in training: {len_tr}={round(100 * len_tr / whole_set, 1)}%, testing: {len_te}={round(100 * len_te / whole_set, 1)}%')
        print(
            f'Overlap between training and testing: {intersect_set}={round(100 * intersect_set / len_tr, 1)}% of training, {round(100 * intersect_set / len_te, 1)}% of testing')
