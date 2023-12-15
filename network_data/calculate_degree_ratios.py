import networkx as nx
import pandas as pd


def read_as_graph(path):
    edge_list = []
    with open(path, 'r') as file:
        for line in file:
            id0, id1 = line.strip().split(' ')
            edge_list.append((id0, id1))
    g = nx.from_edgelist(edge_list)
    return g


def create_degree_df(train_file_pos, train_file_neg, dataset):
    pos_g = read_as_graph(train_file_pos)
    neg_g = read_as_graph(train_file_neg)
    pos_degrees = pos_g.degree
    pos_degrees = {entry[0]: entry[1] for entry in pos_degrees}
    neg_degrees = neg_g.degree
    neg_degrees = {entry[0]: entry[1] for entry in neg_degrees}
    df_pos = pd.DataFrame.from_dict(pos_degrees, orient='index')
    df_neg = pd.DataFrame.from_dict(neg_degrees, orient='index')
    df = df_pos.join(df_neg, how='outer', lsuffix='pos', rsuffix='neg')
    df = df.rename(columns={'0pos': 'pos_degree', '0neg': 'neg_degree'})
    df = df.fillna(0)
    df['degree_ratio'] = df['pos_degree'] / (df['pos_degree'] + df['neg_degree'])
    df['dataset'] = dataset
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'protein'}, inplace=True)
    return df


for test in ['original', 'rewired']:
    df_all = pd.DataFrame(columns=['protein', 'pos_degree', 'neg_degree', 'degree_ratio', 'dataset'])
    for ds in ['du', 'guo', 'huang', 'pan', 'richoux_regular', 'richoux_strict', 'dscript', 'gold-standard']:
        if ds != 'gold-standard':
            tr_file_pos = f'../algorithms/SPRINT/data/{test}/{ds}_train_pos.txt'
            tr_file_neg = f'../algorithms/SPRINT/data/{test}/{ds}_train_neg.txt'
        elif ds == 'gold-standard' and test == 'original':
            tr_file_pos = '../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt'
            tr_file_neg = '../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt'
        else:
            break
        df_ds = create_degree_df(tr_file_pos, tr_file_neg, ds)
        df_all = pd.concat([df_all, df_ds])
    df_all.to_csv(f'{test}_degree_ratios.csv', index=False)


df_all = pd.DataFrame(columns=['protein', 'pos_degree', 'neg_degree', 'degree_ratio', 'dataset'])
for ds in ['du', 'guo', 'huang', 'pan', 'richoux', 'dscript']:
    for partition in ['both', '0', '1']:
        tr_file_pos = f'../algorithms/SPRINT/data/partitions/{ds}_partition_{partition}_pos.txt'
        tr_file_neg = f'../algorithms/SPRINT/data/partitions/{ds}_partition_{partition}_neg.txt'
        df_ds = create_degree_df(tr_file_pos, tr_file_neg, f'{ds}_{partition}')
        df_all = pd.concat([df_all, df_ds])
df_all.to_csv(f'partition_degree_ratios.csv', index=False)
