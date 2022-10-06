import networkx as nx

file = open('node_degrees.csv', 'w')
file.write('Node,Degree,Test,Dataset,Split,Network\n')
for test in ['original', 'rewired']:
    for dataset in ['du', 'guo', 'huang', 'pan', 'richoux_regular', 'richoux_strict']:
        for split in ['train', 'test']:
            for network in ['pos', 'neg']:
                G = nx.read_edgelist(f'{test}/{dataset}_{split}_{network}.txt')
                degree_sequence = {node: G.degree()[node] for node in G.nodes()}
                for key, val in degree_sequence.items():
                    file.write(f'{key},{val},{test},{dataset},{split},{network}\n')

for dataset in ['du', 'guo', 'huang', 'pan', 'richoux']:
    for partition in ['0', '1', 'both']:
        for network in ['pos', 'neg']:
            G = nx.read_edgelist(f'partitions/{dataset}_partition_{partition}_{network}.txt')
            degree_sequence = {node: G.degree()[node] for node in G.nodes()}
            for key, val in degree_sequence.items():
                file.write(f'{key},{val},partition,{dataset},{partition},{network}\n')
file.close()