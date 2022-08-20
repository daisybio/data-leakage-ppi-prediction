from sklearn.model_selection import train_test_split


def generate_RDPN(ppis, expected=True):
    import numpy as np
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

    return ppis_rewired


def write_sprint(data, prefix):
    pos_file = open(f'data/rewired/{prefix}_pos.txt', 'w')
    neg_file = open(f'data/rewired/{prefix}_neg.txt', 'w')
    for ppi in data:
        if ppi[2] == '0':
            neg_file.write(f'{ppi[0]} {ppi[1]}\n')
        else:
            pos_file.write(f'{ppi[0]} {ppi[1]}\n')
    pos_file.close()
    neg_file.close()


def rewrite_guo(rewired=False):
    ppis = []
    with open('../../Datasets_PPIs/Guo_yeast_DIP/protein.actions.tsv') as f:
        for line in f:
            line_split = line.strip().split('\t')
            ppis.append(line_split)
    print(f'Guo: n={len(ppis)}')
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    print(f'n_train={len(ppis_train)}, n_test={len(ppis_test)}')
    write_sprint(ppis_train, 'guo_train')
    write_sprint(ppis_test, 'guo_test')


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
    print(f'Huang: n={len(ppis)}')
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    print(f'n_train={len(ppis_train)}, n_test={len(ppis_test)}')
    write_sprint(ppis_train, 'huang_train')
    write_sprint(ppis_test, 'huang_test')


def rewrite_du(rewired=False):
    ppis = []
    with open('../../Datasets_PPIs/Du_yeast_DIP/SupplementaryS1.csv') as f:
        for line in f:
            line_split = line.strip().split(',')
            ppis.append(line_split)
    print(f'Du: n={len(ppis)}')
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    print(f'n_train={len(ppis_train)}, n_test={len(ppis_test)}')
    write_sprint(ppis_train, 'du_train')
    write_sprint(ppis_test, 'du_test')


def rewrite_pan(rewired=False):
    from algorithms.Custom.load_datasets import make_swissprot_to_dict, iterate_pan
    prefix_dict, seq_dict = make_swissprot_to_dict('../../Datasets_PPIs/SwissProt/human_swissprot.fasta')
    print('Mapping Protein IDs ...')
    mapping_dict = iterate_pan(prefix_dict, seq_dict, '../../Datasets_PPIs/Pan_human_HPRD/SEQ-Supp-ABCD.tsv')
    ppis = []
    with open('../../Datasets_PPIs/Pan_human_HPRD/Supp-AB.tsv', 'r') as f:
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
    print(f'Pan: n={len(ppis)}')
    ppis_train, ppis_test = train_test_split(ppis, test_size=0.2, random_state=42)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    print(f'n_train={len(ppis_train)}, n_test={len(ppis_test)}')
    write_sprint(ppis_train, 'pan_train')
    write_sprint(ppis_test, 'pan_test')


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
        # 85,104
        path_to_train = '../DeepPPI/data/mirror/medium_1166_train_mirror.txt'
        # 12,822
        path_to_val = '../DeepPPI/data/mirror/medium_1166_val_mirror.txt'
        # 12,806
        path_to_test = '../DeepPPI/data/mirror/medium_1166_test_mirror.txt'
    else:
        # 91,036
        path_to_train = '../DeepPPI/data/mirror/double/double-medium_1166_train_mirror.txt'
        # 12,506
        path_to_val = '../DeepPPI/data/mirror/double/double-medium_1166_val_mirror.txt'
        # 720
        path_to_test = '../DeepPPI/data/mirror/double/test_double_mirror.txt'

    ppis_train = read_richoux_file(path_to_train)
    ppis_val = read_richoux_file(path_to_val)
    ppis_train.extend(ppis_val)
    ppis_test = read_richoux_file(path_to_test)
    if rewired:
        ppis_train = generate_RDPN(ppis_train)
    print(f'Richoux {dataset}: n={len(ppis_train) + len(ppis_test)}')
    print(f'n_train={len(ppis_train)}, n_test={len(ppis_test)}')
    write_sprint(ppis_train, f'richoux_{dataset}_train')
    write_sprint(ppis_test, f'richoux_{dataset}_test')


if __name__ == '__main__':
   rewrite_guo(rewired=True)
   rewrite_huang(rewired=True)
   rewrite_du(rewired=True)
   rewrite_pan(rewired=True)
   rewrite_richoux(dataset='regular', rewired=True)
   rewrite_richoux(dataset='strict', rewired=True)
