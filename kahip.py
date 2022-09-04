import os
import subprocess

import networkx as nx


def run_kahip():
    process = subprocess.Popen(['./KaHIP/deploy/kaffpa',
                                './network_data/SIMAP2/only_yeast_bitscore.graph',
                                '--seed=1234',
                                '--output_filename="./network_data/SIMAP2/only_yeast_partition_bitscore.txt"',
                                '--k=2',
                                '--preconfiguration=strong'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)

    process = subprocess.Popen(['./KaHIP/deploy/kaffpa',
                                './network_data/SIMAP2/only_human_bitscore.graph',
                                '--seed=1234',
                                '--output_filename="./network_data/SIMAP2/only_human_partition_bitscore.txt"',
                                '--k=2',
                                '--preconfiguration=strong'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)


def write_partition(yeast=True):
    if yeast:
        path_to_network = 'network_data/SIMAP2/yeast_networks/only_yeast.graphml'
        path_to_partition = './network_data/SIMAP2/yeast_networks/only_yeast_partition_bitscore.txt'
        path_to_output = './network_data/SIMAP2/yeast_networks/only_yeast_partition.graphml'
        path_to_list_output = './network_data/SIMAP2/yeast_networks/only_yeast_partition_nodelist.txt'
    else:
        path_to_network = 'network_data/SIMAP2/human_networks/only_human.graphml'
        path_to_partition = './network_data/SIMAP2/human_networks/only_human_partition_bitscore.txt'
        path_to_output = './network_data/SIMAP2/human_networks/only_human_partition.graphml'
        path_to_list_output = './network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt'
    G = nx.read_graphml(path_to_network)
    node = 1
    attrs = dict()
    print('Parsing KaHIP output ...')
    with open(path_to_partition, 'r') as f:
        for line in f:
            if line == '0\n':
                attrs[str(node)] = {'partition': 0}
            else:
                attrs[str(node)] = {'partition': 1}
            node += 1
    nx.set_node_attributes(G, attrs)
    print('Exporting ...')
    with open(path_to_list_output, 'w') as f:
        f.write('Node\tPartition\n')
        for key, value in attrs.items():
            f.write(f'{G.nodes[key]["uniprot_id"]}\t{value["partition"]}\n')
    nx.write_graphml(G, path_to_output)
    return G


if __name__ == "__main__":
    #run_kahip()
    print("Yeast ...")
    G_yeast = write_partition(yeast=True)
    print("Human ...")
    G_human = write_partition(yeast=False)