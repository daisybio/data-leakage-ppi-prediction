import numpy as np
import networkx as nx
from tqdm import tqdm


def read_protnames(path):
    prot_list = list()
    f = open(path, 'r').readlines()
    for line in tqdm(f):
        prot_list.append(line.strip())
    return prot_list


def compute_PCA(matrix):
    '''
    Computes a PCA on the standardized protein-protein similarity matrix.
    :param matrix: protein-protein similarity matrix
    :return:
    '''
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    print('Computing PCA ...')
    standardizedData = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=128)
    pca.fit(standardizedData)
    print(pca.explained_variance_ratio_)

    plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1, len(pca.explained_variance_) + 1),
             np.cumsum(pca.explained_variance_),
             c='red',
             label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    return pca.fit_transform(standardizedData)


def compute_MDS(matrix):
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import MDS
    print('Standardizing Data ...')
    standardizedData = StandardScaler().fit_transform(matrix)
    embedding = MDS(n_components=128)
    print('Computing MDS ...')
    mds = embedding.fit_transform(standardizedData)
    return mds


def compute_node2vec(network, organism):
    path_to_edgelist = f'data/{organism}.edgelist'
    path_to_nodelist = f'data/{organism}.nodelist'
    nx.write_edgelist(network, path_to_edgelist, data=['bitscore'])
    nodes = list(network.nodes.data('uniprot_id'))
    with open(path_to_nodelist, 'w') as f:
        for node in nodes:
            f.write(f'{node[0]}\t{node[1]}\n')


if __name__ == "__main__":
    sim_matrix_yeast = np.load(
        '../../network_data/SIMAP2/matrices/sim_matrix_yeast.npy')
    yeast_pca = compute_PCA(sim_matrix_yeast)
    np.save('data/yeast_pca.npy', yeast_pca)
    yeast_mds = compute_MDS(sim_matrix_yeast)
    np.save('data/yeast_mds.npy', yeast_mds)

    sim_matrix_human = np.load(
        '../../network_data/SIMAP2/matrices/sim_matrix_human.npy')
    human_pca = compute_PCA(sim_matrix_human)
    np.save('data/human_pca.npy', human_pca)
    human_mds = compute_MDS(sim_matrix_human)
    np.save('data/human_mds.npy', human_mds)

    yeast_network = nx.read_graphml(
        '../../network_data/SIMAP2/yeast_networks/only_yeast.graphml')
    compute_node2vec(yeast_network, 'yeast')
    human_network = nx.read_graphml(
        '../../network_data/SIMAP2/human_networks/only_human.graphml')
    compute_node2vec(human_network, 'human')

    ### run
    # cd snap/examples/node2vec
    # make all
    # ./node2vec -i:../../../algorithms/Custom/data/yeast.edgelist -o:../../../algorithms/Custom/data/yeast.emb
    # ./node2vec -i:../../../algorithms/Custom/data/human.edgelist -o:../../../algorithms/Custom/data/human.emb
