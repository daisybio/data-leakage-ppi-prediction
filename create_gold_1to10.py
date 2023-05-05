import os

def sample_block(block):
    print(f'######## {block} ###########')
    from create_gold_standard import sample_negatives, export_block
    file_pos = f'Datasets_PPIs/Hippiev2.3/{block}_pos_rr.txt'
    ppis_pos = read_block(file_pos)
    file_neg = f'Datasets_PPIs/Hippiev2.3/{block}_neg_rr.txt'
    ppis_neg = read_block(file_neg)
    all_ppis = ppis_pos.union(ppis_neg)
    ppis_neg = sample_negatives(ppis_pos, ppis_neg, all_ppis, factor=10)
    export_block(ppis_pos, f'Datasets_PPIs/unbalanced_gold/{block}_pos.txt')
    export_block(ppis_neg, f'Datasets_PPIs/unbalanced_gold/{block}_neg.txt')


def read_block(file):
    ppis = set()
    with open(file, 'r') as f:
        for line in f:
            prot_1, prot_2 = line.strip().split(' ')
            ppis.add((prot_1, prot_2))
    return ppis


if __name__ == '__main__':
    os.mkdir('Datasets_PPIs/unbalanced_gold')
    sample_block('Intra0')
    sample_block('Intra1')
    sample_block('Intra2')