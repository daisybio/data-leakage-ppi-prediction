import sys
import random
from sklearn.model_selection import train_test_split


def clean_dataset(ppis):
    '''
    Cleans duplicates and overlaps between positive and negative dataset,
    balances dataset
    :param ppis:
    :return: cleaned up ppis
    '''
    n_ppis = len(ppis)
    print(f'Original number of PPIs: {n_ppis}'
          f'({len([ppi for ppi in ppis if ppi[2] == "1"])} pos/{len([ppi for ppi in ppis if ppi[2] == "0"])} neg)')

    ppis = {(ppi[0], ppi[1], ppi[2]) for ppi in ppis}
    print(f'Number of duplicates: {n_ppis-len(ppis)}')

    pos_ppis = {ppi for ppi in ppis if ppi[2] == '1'}
    pos_ppis = pos_ppis.union({(ppi[1], ppi[0], ppi[2]) for ppi in pos_ppis})
    neg_ppis = {ppi for ppi in ppis if ppi[2] == '0'}
    neg_ppis = neg_ppis.union({(ppi[1], ppi[0], ppi[2]) for ppi in neg_ppis})
    intersect_ppis = pos_ppis.intersection(neg_ppis)
    print(f'Number of overlaps between pos and neg: {len(intersect_ppis)}')
    ppis = [[ppi[0], ppi[1], ppi[2]] for ppi in ppis if ppi not in intersect_ppis]
    return ppis


def balance_ppis(ppis_train, ppis_test, organism):
    print(f'Current status: n={len(ppis_train+ppis_test)},'
          f'n_train={len(ppis_train)} ({len([ppi for ppi in ppis_train if ppi[2] == "1"])} pos/{len([ppi for ppi in ppis_train if ppi[2] == "0"])} neg), '
          f'n_test={len(ppis_test)} ({len([ppi for ppi in ppis_test if ppi[2] == "1"])} pos/{len([ppi for ppi in ppis_test if ppi[2] == "0"])} neg)'
          )
    ppis_train = balance_ppis_list(ppis_train, organism)
    ppis_test = balance_ppis_list(ppis_test, organism)
    print(f'New status: n={len(ppis_train+ppis_test)},'
          f'n_train={len(ppis_train)} ({len([ppi for ppi in ppis_train if ppi[2] == "1"])} pos/{len([ppi for ppi in ppis_train if ppi[2] == "0"])} neg), '
          f'n_test={len(ppis_test)} ({len([ppi for ppi in ppis_test if ppi[2] == "1"])} pos/{len([ppi for ppi in ppis_test if ppi[2] == "0"])} neg)'
          )
    return ppis_train, ppis_test


def balance_ppis_list(ppis, organism):
    from algorithms.Custom.load_datasets import node2vec_embeddings_to_dict
    pos_ppis = [x for x in ppis if x[2] == '1']
    pos_len = len(pos_ppis)
    neg_ppis = [x for x in ppis if x[2] == '0']
    neg_len = len(neg_ppis)
    if neg_len > pos_len:
        print(f'randomly dropping negatives ({pos_len} positives, {neg_len} negatives)...')
        to_delete = set(random.sample(range(len(neg_ppis)), len(neg_ppis) - len(pos_ppis)))
        neg_ppis = [x for i, x in enumerate(neg_ppis) if not i in to_delete]
        ppis = pos_ppis
        ppis.extend(neg_ppis)
    elif len(pos_ppis) > len(neg_ppis):
        print(f'sampling more negatives ({pos_len} positives, {neg_len} negatives)...')
        id_dict_PCA_MDS = load_id_dict(organism, n2v=False)
        id_dict_n2v = load_id_dict(organism, n2v=True)
        if organism == 'yeast':
            n2v_emb = node2vec_embeddings_to_dict('../Custom/data/yeast.emb')
        else:
            n2v_emb = node2vec_embeddings_to_dict('../Custom/data/human.emb')
        candidates_n2v = {key for key, value in id_dict_n2v.items() if value in n2v_emb.keys()}
        candidates_PCA_MDS = {key for key, value in id_dict_PCA_MDS.items()}
        candidates = candidates_n2v.intersection(candidates_PCA_MDS)

        while pos_len > neg_len:
            prot1 = random.choice(tuple(candidates))
            prot1_list = [pair[0] for pair in ppis if pair[1] == prot1] + [pair[1] for pair in ppis if
                                                                           pair[0] == prot1]
            # protein should occur in the dataset
            while len(prot1_list) == 0:
                prot1 = random.choice(tuple(candidates))
                prot1_list = [pair[0] for pair in ppis if pair[1] == prot1] + [pair[1] for pair in ppis if
                                                                               pair[0] == prot1]
            prot2 = random.choice(tuple(candidates))
            prot2_list = [pair[0] for pair in ppis if pair[1] == prot2] + [pair[1] for pair in ppis if
                                                                           pair[0] == prot2]

            while prot1 == prot2 or prot2 in prot1_list or len(prot2_list) == 0:
                prot2 = random.choice(tuple(candidates))
                prot2_list = [pair[0] for pair in ppis if pair[1] == prot2] + [pair[1] for pair in ppis if
                                                                               pair[0] == prot2]
            ppis.append([prot1, prot2, '0'])
            neg_len += 1
    return ppis


def load_id_dict(organism, n2v=True):
    from algorithms.Custom.load_datasets import read_matrix_colnames, read_nodelist
    if organism == 'yeast':
        if n2v:
            id_dict = read_nodelist('../Custom/data/yeast.nodelist')
        else:
            id_dict = read_matrix_colnames('../../network_data/SIMAP2/matrices/sim_matrix_yeast_colnames.txt')
    else:
        if n2v:
            id_dict = read_nodelist('../Custom/data/human.nodelist')
        else:
            id_dict = read_matrix_colnames('../../network_data/SIMAP2/matrices/sim_matrix_human_colnames.txt')
    return id_dict


def generate_RDPN(ppis, expected=True, add_mirrors=False):
    import networkx as nx
    print('Rewiring ...')
    pos_g = nx.Graph()
    neg_g = nx.Graph()
    pos_edges = []
    neg_edges = []
    id = 0
    uid_to_newid = dict()
    for ppi in ppis:
        if ppi[0] not in uid_to_newid:
            pos_g.add_node(id, uid=ppi[0])
            neg_g.add_node(id, uid=ppi[0])
            uid_to_newid[ppi[0]] = id
            id += 1
        if ppi[1] not in uid_to_newid:
            pos_g.add_node(id, uid=ppi[1])
            neg_g.add_node(id, uid=ppi[1])
            uid_to_newid[ppi[1]] = id
            id += 1

        if ppi[2] == '1':
            pos_edges.append((uid_to_newid.get(ppi[0]),
                              uid_to_newid.get(ppi[1])))
        else:
            neg_edges.append((uid_to_newid.get(ppi[0]),
                              uid_to_newid.get(ppi[1])))
    pos_g.add_edges_from(pos_edges)
    neg_g.add_edges_from(neg_edges)
    if expected:
        degree_view = pos_g.degree()
        degree_sequence = [degree_view[node] for node in pos_g.nodes()]
        rewired_network = nx.expected_degree_graph(degree_sequence, seed=1234, selfloops=False)
    else:
        import graph_tool.all as gt
        d = nx.to_dict_of_lists(pos_g)
        edges = [(i, j) for i in d for j in d[i]]
        GT = gt.Graph(directed=False)
        GT.add_vertex(sorted(pos_g.nodes())[-1])
        GT.add_edge_list(edges)

        gt.random_rewire(GT, model="constrained-configuration", n_iter=100, edge_sweep=True)

        edges_new = list(GT.get_edges())
        edges_new = [tuple(x) for x in edges_new]
        rewired_network = nx.Graph()
        rewired_network.add_nodes_from(pos_g.nodes())
        rewired_network.add_edges_from(edges_new)
    edge_list = nx.generate_edgelist(rewired_network)
    ppis_rewired = []
    idx = 0
    for edge in edge_list:
        uid0 = pos_g.nodes[int(edge.split()[0])]['uid']
        uid1 = pos_g.nodes[int(edge.split()[1])]['uid']
        ppis_rewired.append([uid0, uid1, '1'])
        idx += 1

    edge_list1 = nx.generate_edgelist(neg_g)
    for edge in edge_list1:
        uid0 = neg_g.nodes[int(edge.split()[0])]['uid']
        uid1 = neg_g.nodes[int(edge.split()[1])]['uid']
        ppis_rewired.append([uid0, uid1, '0'])
        idx += 1
    if add_mirrors:
        mirror_edgelist = [[ppi[1], ppi[0], ppi[2]] for ppi in ppis_rewired]
        ppis_rewired.extend(mirror_edgelist)

    return ppis_rewired


def write_sprint(data, prefix, rewired):
    if rewired:
        pos_file = open(f'data/rewired/{prefix}_pos.txt', 'w')
        neg_file = open(f'data/rewired/{prefix}_neg.txt', 'w')
    else:
        pos_file = open(f'data/original/{prefix}_pos.txt', 'w')
        neg_file = open(f'data/original/{prefix}_neg.txt', 'w')
    for ppi in data:
        if ppi[2] == '0':
            neg_file.write(f'{ppi[0]} {ppi[1]}\n')
        else:
            pos_file.write(f'{ppi[0]} {ppi[1]}\n')
    pos_file.close()
    neg_file.close()


def rewrite_guo(rewired=False):
    print('############################ GUO DATASET ############################')
    ppis = []
    with open('../../algorithms/seq_ppi/yeast/preprocessed/protein.actions.tsv') as f:
        for line in f:
            line_split = line.strip().split('\t')
            ppis.append(line_split)
    ppis = clean_dataset(ppis)
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    ppis_train, ppis_test = balance_ppis(ppis_train, ppis_test, organism='yeast')
    write_sprint(ppis_train, 'guo_train', rewired)
    write_sprint(ppis_test, 'guo_test', rewired)


def read_huang_file(path):
    prot_list = []
    with open(path, 'r') as f:
        i = 0
        for line in f:
            if i % 2 == 0:
                id = line.split('|')[1].strip().split(':')[1]
                prot_list.append(id)
            i = i + 1
    return prot_list


def rewrite_huang(rewired=False):
    print('############################ HUANG DATASET ############################')
    ppis = []
    prots_pos_A = read_huang_file('../DeepFE-PPI/dataset/human/positive/Protein_A.txt')
    prots_pos_B = read_huang_file('../DeepFE-PPI/dataset/human/positive/Protein_B.txt')
    for i in range(len(prots_pos_A)):
        if prots_pos_A[i] != '' and prots_pos_B[i] != '':
            ppis.append([prots_pos_A[i], prots_pos_B[i], '1'])
    prots_neg_A = read_huang_file('../DeepFE-PPI/dataset/human/negative/Protein_A.txt')
    prots_neg_B = read_huang_file('../DeepFE-PPI/dataset/human/negative/Protein_B.txt')
    for i in range(len(prots_neg_A)):
        if prots_neg_A[i] != '' and prots_neg_B[i] != '':
            ppis.append([prots_neg_A[i], prots_neg_B[i], '0'])
    ppis = clean_dataset(ppis)
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    ppis_train, ppis_test = balance_ppis(ppis_train, ppis_test, organism='human')
    write_sprint(ppis_train, 'huang_train', rewired)
    write_sprint(ppis_test, 'huang_test', rewired)


def rewrite_du(rewired=False):
    print('############################ DU DATASET ############################')
    ppis = []
    with open('../../Datasets_PPIs/Du_yeast_DIP/SupplementaryS1.csv') as f:
        for line in f:
            line_split = line.strip().split(',')
            ppis.append(line_split)
    ppis = clean_dataset(ppis)
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    ppis_train, ppis_test = balance_ppis(ppis_train, ppis_test, organism='yeast')
    write_sprint(ppis_train, 'du_train', rewired)
    write_sprint(ppis_test, 'du_test', rewired)


def rewrite_pan(rewired=False):
    print('############################ PAN DATASET ############################')
    from algorithms.Custom.load_datasets import make_swissprot_to_dict
    prefix_dict, seq_dict = make_swissprot_to_dict('../../Datasets_PPIs/SwissProt/human_swissprot.fasta')
    print('Mapping Protein IDs ...')
    mapping_dict = iterate_pan(prefix_dict, seq_dict, '../seq_ppi/sun/preprocessed/SEQ-Supp-ABCD.tsv')
    ppis = []
    with open('../seq_ppi/sun/preprocessed/Supp-AB.tsv', 'r') as f:
        for line in f:
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
                        ppis.append([uniprot_id0, uniprot_id1, label])
    ppis = clean_dataset(ppis)
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    ppis_train, ppis_test = balance_ppis(ppis_train, ppis_test, organism='human')
    write_sprint(ppis_train, 'pan_train', rewired)
    write_sprint(ppis_test, 'pan_test', rewired)


def iterate_pan(prefix_dict, seq_dict, path_to_pan):
    lines = open(path_to_pan, 'r').readlines()
    encountered_ids = []
    n = 30
    unmapped = 0
    mapping_dict = dict()
    for line in lines:
        old_id = line.strip().split('\t')[0]
        if old_id not in encountered_ids:
            encountered_ids.append(old_id)
            seq = line.strip().split('\t')[1]
            first_n = seq[0:n]
            if first_n not in prefix_dict.keys():
                uniprot_id = ''
            elif isinstance(prefix_dict[first_n], list):
                uniprot_ids = prefix_dict[first_n]
                uniprot_id = ''
                for id in uniprot_ids:
                    # just take the first mapped ID
                    if seq_dict[id] == seq:
                        uniprot_id = id
                        break
            else:
                uniprot_id = prefix_dict[first_n]
            if uniprot_id == '':
                unmapped += 1
            mapping_dict[old_id] = uniprot_id
    print(f'#unmapped IDs: {unmapped}')
    return mapping_dict



def read_richoux_file(path):
    ppis = []
    with open(path, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                continue
            else:
                id1 = line_split[0]
                id2 = line_split[1]
                label = line_split[4]
                ppis.append([id1, id2, label])
    return ppis


def rewrite_richoux(dataset='regular', rewired=False):
    if dataset == 'regular':
        print('############################ RICHOUX REGULAR DATASET ############################')
        # 85,104
        path_to_train = '../DeepPPI/data/mirror/medium_1166_train_mirror.txt'
        # 12,822
        path_to_val = '../DeepPPI/data/mirror/medium_1166_val_mirror.txt'
        # 12,806
        path_to_test = '../DeepPPI/data/mirror/medium_1166_test_mirror.txt'
    else:
        print('############################ RICHOUX STRICT DATASET ############################')
        # 91,036
        path_to_train = '../DeepPPI/data/mirror/double/double-medium_1166_train_mirror.txt'
        # 12,506
        path_to_val = '../DeepPPI/data/mirror/double/double-medium_1166_val_mirror.txt'
        # 720
        path_to_test = '../DeepPPI/data/mirror/double/test_double_mirror.txt'

    ppis_train = read_richoux_file(path_to_train)
    ppis_val = read_richoux_file(path_to_val)
    ppis_train.extend(ppis_val)
    print('Cleaning train+val dataset')
    ppis_train = clean_dataset(ppis_train)
    ppis_test = read_richoux_file(path_to_test)
    print('Cleaning test dataset')
    ppis_test = clean_dataset(ppis_test)
    if rewired:
        ppis_train = generate_RDPN(ppis_train, add_mirrors=True)
    ppis_train, ppis_test = balance_ppis(ppis_train, ppis_test, organism='human')
    write_sprint(ppis_train, f'richoux_{dataset}_train', rewired)
    write_sprint(ppis_test, f'richoux_{dataset}_test', rewired)


if __name__ == '__main__':
   sys.path.append('../../')
   rewrite_guo(rewired=False)
   rewrite_huang(rewired=False)
   rewrite_du(rewired=False)
   rewrite_pan(rewired=False)
   rewrite_richoux(dataset='regular', rewired=False)
   rewrite_richoux(dataset='strict', rewired=False)
