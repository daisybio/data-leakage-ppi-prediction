# with KaHIP we want to minimize the total cut --> sum of edge weights of the cut
# --> higher pairwise similarity --> higher edge weight; e.g. bit score, %identity
import networkx as nx


def parse_SIMAP2(path_to_file):
    import pandas as pd
    col_names = ['Query',
                 'Match',
                 'BLOSUM50_score',
                 'bitscore',
                 'perc_id',
                 'perc_pos',
                 'overlap',
                 'align_begin_query',
                 'align_end_query',
                 'align_begin_match',
                 'align_end_match'
                 ]
    table = pd.read_csv(path_to_file,
                        sep='\t',
                        names=col_names)
    table[['query_prefix', 'query_id', 'query_name']] = table.Query.str.split("|", expand=True)
    table[['match_prefix', 'match_id', 'match_name']] = table.Match.str.split("|", expand=True)
    table = table[['query_id',
                   'query_name',
                   'match_id',
                   'match_name',
                   'BLOSUM50_score',
                   'bitscore',
                   'perc_id',
                   'perc_pos']]
    return table


def untangle_sim_list(sims):
    human_mask_query = sims['query_name'].str.contains('HUMAN', case=True, na=False)
    human_mask_match = sims['match_name'].str.contains('HUMAN', case=True, na=False)
    h_list = sims[human_mask_query][human_mask_match]

    yeast_mask_query = sims['query_name'].str.contains('YEAST', case=True, na=False)
    yeast_mask_match = sims['match_name'].str.contains('YEAST', case=True, na=False)
    y_list = sims[yeast_mask_query][yeast_mask_match]

    return h_list, y_list


def list_to_network(sims):
    import networkx as nx
    G = nx.from_pandas_edgelist(sims, 'query_id', 'match_id',
                                ['BLOSUM50_score',
                                 'bitscore',
                                 'perc_id',
                                 'perc_pos'])
    # remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def write_graphml_files(path_simap, output_yeast, output_human):
    print('parsing SIMAP2 ...')
    sim_list = parse_SIMAP2(path_simap)
    print('separate human and yeast ...')
    human_list, yeast_list = untangle_sim_list(sim_list)
    print('Make yeast similarity network ...')
    G_yeast = list_to_network(yeast_list)
    print('Make human similarity network ...')
    G_human = list_to_network(human_list)
    print('Converting labels to integers: yeast ...')
    G_yeast = nx.convert_node_labels_to_integers(G_yeast, label_attribute='uniprot_id', first_label=1)
    print('Converting labels to integers: human ...')
    G_human = nx.convert_node_labels_to_integers(G_human, label_attribute='uniprot_id', first_label=1)
    print('Export yeast ...')
    nx.write_graphml(G_yeast, output_yeast)
    print('Export human ...')
    nx.write_graphml(G_human, output_human)


def write_metis(G, path, attribute):
    """Convert a graph to the numbered adjacency list structure expected by
    METIS.
    """
    with open(path, 'w') as f:
        f.write('% Yeast network by SIMAP2\n')
        f.write(f'{G.number_of_nodes()}\t{G.number_of_edges()}\t1\n')
        edge_counter = 0
        for node in G.nodes:
            line = []
            for edge in G.edges(node, data=True):
                edge_counter += 1
                weight = round(edge[2][attribute])
                line.append(f'{edge[1]} {weight}')
            line = '  '.join(line)
            f.write(line)
            f.write('\n')
    print(f'{edge_counter} edges')


if __name__ == "__main__":
    path_to_similarities = 'network_data/SIMAP2/submatrix.tsv'
    graphml_yeast = 'network_data/SIMAP2/yeast_networks/only_yeast.graphml'
    graphml_human = 'network_data/SIMAP2/human_networks/only_human.graphml'
    write_graphml_files(path_to_similarities, graphml_yeast, graphml_human)
    print('Reading yeast ...')
    G_yeast = nx.read_graphml(graphml_yeast)
    # write METIS
    attribute = 'bitscore'
    print(f'Writing yeast METIS file with {attribute} weights ...')
    write_metis(G_yeast, f'network_data/SIMAP2/yeast_networks/only_yeast_{attribute}.graph', attribute)

    print('Reading human ...')
    G_human = nx.read_graphml(graphml_human)
    print(f'Writing human METIS file with {attribute} weights ...')
    write_metis(G_human, f'network_data/SIMAP2/human_networks/only_human_{attribute}.graph', attribute)
