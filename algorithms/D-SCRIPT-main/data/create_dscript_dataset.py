import random

def balance_ppis_list(ppis, universe):
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
        candidates = set( ppi for entry in ppis for ppi in entry[:2] )

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
        print(f"pos: {len([x for x in ppis if x[2] == '1'])}, neg: {len([x for x in ppis if x[2] == '0'])}")
    return ppis


def create_dataset(dataset, folder, fold, organism):
    universe = set()
    with open(f"../../../Datasets_PPIs/SwissProt/{organism}_proteins.txt") as universe_file:
        for line in universe_file:
            universe.add(line.strip())

    ppis = []
    with open(f"../../SPRINT/data/{folder}/{dataset}_{fold}_pos.txt", "r") as f_in_pos, \
            open(f"../../SPRINT/data/{folder}/{dataset}_{fold}_neg.txt", "r") as f_in_neg:

        # Process positive examples
        for line in f_in_pos:
            columns = line.strip().split(" ")
            if columns[0] in universe and columns[1] in universe:
                ppis.append([columns[0], columns[1], '1'])

        # Process negative examples
        for line in f_in_neg:
            columns = line.strip().split(" ")
            if columns[0] in universe and columns[1] in universe:
                ppis.append([columns[0], columns[1], '0'])

    ppis = balance_ppis_list(ppis, universe)

    with open(f"{folder}/{dataset}_{fold}.txt", "w") as f_out:
        for ppi in ppis:
            f_out.write(f"{ppi[0]}\t{ppi[1]}\t{ppi[2]}\n")


if __name__ == '__main__':
    for dataset in ['guo', 'du']:
        print(dataset)
        create_dataset(dataset, 'original', 'train', 'yeast')
        create_dataset(dataset, 'original', 'test', 'yeast')
        create_dataset(dataset, 'rewired', 'train', 'yeast')
        create_dataset(dataset, 'rewired', 'test', 'yeast')
        create_dataset(dataset, 'partitions', 'partition_0', 'yeast')
        create_dataset(dataset, 'partitions', 'partition_1', 'yeast')
        create_dataset(dataset, 'partitions', 'partition_both', 'yeast')
    for dataset in ['huang', 'pan', 'richoux']:
        print(dataset)
        if dataset == 'richoux':
            create_dataset('richoux_regular', 'original', 'train', 'human')
            create_dataset('richoux_regular', 'original', 'test', 'human')
            create_dataset('richoux_regular', 'rewired', 'train', 'human')
            create_dataset('richoux_regular', 'rewired', 'test', 'human')
            create_dataset('richoux_strict', 'original', 'train', 'human')
            create_dataset('richoux_strict', 'original', 'test', 'human')
            create_dataset('richoux_strict', 'rewired', 'train', 'human')
            create_dataset('richoux_strict', 'rewired', 'test', 'human')
            create_dataset(dataset, 'partitions', 'partition_0', 'human')
            create_dataset(dataset, 'partitions', 'partition_1', 'human')
            create_dataset(dataset, 'partitions', 'partition_both', 'human')
        else:
            create_dataset(dataset, 'original', 'train', 'human')
            create_dataset(dataset, 'original', 'test', 'human')
            create_dataset(dataset, 'rewired', 'train', 'human')
            create_dataset(dataset, 'rewired', 'test', 'human')
            create_dataset(dataset, 'partitions', 'partition_0', 'human')
            create_dataset(dataset, 'partitions', 'partition_1', 'human')
            create_dataset(dataset, 'partitions', 'partition_both', 'human')