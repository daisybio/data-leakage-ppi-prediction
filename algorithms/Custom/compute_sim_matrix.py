import numpy as np
from tqdm import tqdm


def parse_proteins(path_to_simap):
    all_yeast_prots = list()
    all_human_prots = list()
    score_dict = dict()
    f = open(path_to_simap, 'r').readlines()
    for line in tqdm(f):
        line_split = line.strip().split('\t')
        id0 = line_split[0].split('|')[1]
        id1 = line_split[1].split('|')[1]
        if 'HUMAN' in line_split[0] and id0 not in all_human_prots:
            all_human_prots.append(id0)
        elif 'YEAST' in line_split[0] and id0 not in all_yeast_prots:
            all_yeast_prots.append(id0)
        if 'HUMAN' in line_split[1] and id1 not in all_human_prots:
            all_human_prots.append(id1)
        elif 'YEAST' in line_split[1] and id1 not in all_yeast_prots:
            all_yeast_prots.append(id1)
        if ('HUMAN' in line_split[0] and 'HUMAN' in line_split[1]) or (
                'YEAST' in line_split[0] and 'YEAST' in line_split[1]):
            score_dict[f'{id0}_{id1}'] = line_split[3]
    return score_dict, all_yeast_prots, all_human_prots


def compute_sim_matrix(score_dict, all_prots):
    sim_matrix = np.zeros(shape=(len(all_prots), len(all_prots)))
    idx_p0 = 0
    for p0 in tqdm(all_prots):
        idx_p1 = 0
        for p1 in all_prots:
            if f'{p0}_{p1}' in score_dict.keys():
                sim_matrix[idx_p0][idx_p1] = score_dict[f'{p0}_{p1}']
            idx_p1 += 1
        idx_p0 += 1
    return sim_matrix


def write_list(path, prot_list):
    with open(path, 'w') as f:
        for item in prot_list:
            f.write(f'{item}\n')


def parse_simap2(path_to_simap):
    score_dict, all_yeast_prots, all_human_prots = parse_proteins(path_to_simap)
    print('Computing yeast similarity matrix ...')
    sim_matrix_yeast = compute_sim_matrix(score_dict, all_yeast_prots)
    print('Saving yeast similarity matrix ...')
    write_list('../../network_data/SIMAP2/matrices/sim_matrix_yeast_colnames.txt',
               all_yeast_prots)
    np.save('../../network_data/SIMAP2/matrices/sim_matrix_yeast.npy',
            sim_matrix_yeast)
    print('Computing human similarity matrix ...')
    sim_matrix_human = compute_sim_matrix(score_dict, all_human_prots)
    print('Saving human similarity matrix ...')
    write_list('../../network_data/SIMAP2/matrices/sim_matrix_human_colnames.txt',
               all_human_prots)
    np.save('../../network_data/SIMAP2/matrices/sim_matrix_human.npy',
            sim_matrix_human)
    return sim_matrix_yeast, sim_matrix_human


def binarize_matrices(sim_matrix_yeast, sim_matrix_human):
    binary_yeast = np.where(sim_matrix_yeast > 0, 1, 0)
    np.save('../../network_data/SIMAP2/matrices/sim_matrix_yeast_binary.npy',
            binary_yeast)
    binary_human = np.where(sim_matrix_human > 0, 1, 0)
    np.save('../../network_data/SIMAP2/matrices/sim_matrix_human_binary.npy',
            binary_human)


if __name__ == "__main__":
    path_to_submatrix = '../../network_data/SIMAP2/submatrix.tsv'
    sim_matrix_yeast, sim_matrix_human = parse_simap2(path_to_submatrix)
    binarize_matrices(sim_matrix_yeast, sim_matrix_human)
