import random


def balance_ppis_list(ppis, factor=1):
    pos_ppis = set(x for x in ppis if x[2] == '1')
    pos_len = len(pos_ppis)
    neg_ppis = set(x for x in ppis if x[2] == '0')
    neg_len = len(neg_ppis)
    if neg_len > (factor * pos_len):
        print(f'randomly dropping negatives ({pos_len} positives, {neg_len} negatives)...')
        to_delete = set(random.sample(range(len(neg_ppis)), len(neg_ppis) - (factor * len(pos_ppis))))
        neg_ppis = set(x for i, x in enumerate(neg_ppis) if not i in to_delete)
    elif (factor * pos_len) > neg_len:
        print(f'sampling more negatives ({pos_len} positives, {neg_len} negatives)...')
        candidates = set(ppi for entry in ppis for ppi in entry[:2])
        to_generate = (factor * pos_len) - neg_len
        while (factor * pos_len) > neg_len:
            if to_generate % 100 == 0:
                print(f'Still {to_generate} proteins left to generate!')
            prot1 = random.choice(tuple(candidates))
            prot2 = random.choice(tuple(candidates))
            while prot1 == prot2 or (prot1, prot2) in ppis or (prot2, prot1) in ppis or (prot2, prot1) in neg_ppis:
                prot2 = random.choice(tuple(candidates))
            neg_ppis.add((prot1, prot2, '0'))
            neg_len = len(neg_ppis)
            to_generate = (factor * pos_len) - neg_len
    ppis = pos_ppis
    ppis.update(neg_ppis)
    print(f"pos: {len([x for x in ppis if x[2] == '1'])}, neg: {len([x for x in ppis if x[2] == '0'])}")
    return ppis


def process_files_from_sprint(ppis, pos_file, neg_file, universe, folder, fold, factor=1, seed=None):
    with open(pos_file, "r") as f_in_pos, \
            open(neg_file, "r") as f_in_neg:

        # Process positive examples
        for line in f_in_pos:
            columns = line.strip().split(" ")
            if columns[0] in universe and columns[1] in universe:
                ppis.add((columns[0], columns[1], '1'))

        # Process negative examples
        for line in f_in_neg:
            columns = line.strip().split(" ")
            if columns[0] in universe and columns[1] in universe:
                ppis.add((columns[0], columns[1], '0'))

    ppis = balance_ppis_list(ppis, factor=factor)

    if seed is not None:
        with open(f"multiple_runs/{folder}_{dataset}_{fold}_{seed}.txt", "w") as f_out:
            for ppi in ppis:
                f_out.write(f"{ppi[0]}\t{ppi[1]}\t{ppi[2]}\n")
    else:
        with open(f"{folder}/{dataset}_{fold}.txt", "w") as f_out:
            for ppi in ppis:
                f_out.write(f"{ppi[0]}\t{ppi[1]}\t{ppi[2]}\n")


def create_dataset(dataset, folder, fold, organism, multiple_random_splits=False):
    universe = set()
    with open(f"../../../Datasets_PPIs/SwissProt/{organism}_proteins_lengths.txt") as universe_file:
        for line in universe_file:
            protein, length = line.strip().split('\t')
            if 50 < float(length) < 1000:
                universe.add(protein)

    ppis = set()
    if dataset == 'dscript':
        factor = 10
    else:
        factor = 1
    if multiple_random_splits:
        for seed in ["7413", "17612", "29715", "30940", "31191", "42446", "50495", "60688", "75212", "81645"]:
            pos_file = f"../../SPRINT/data/{folder}/multiple_random_splits/{dataset}_{fold}_pos_{seed}.txt"
            neg_file = f"../../SPRINT/data/{folder}/multiple_random_splits/{dataset}_{fold}_neg_{seed}.txt"
            process_files_from_sprint(ppis, pos_file, neg_file, universe, folder, fold, factor, seed)
    else:
        process_files_from_sprint(ppis, f"../../SPRINT/data/{folder}/{dataset}_{fold}_pos.txt",
                                  f"../../SPRINT/data/{folder}/{dataset}_{fold}_neg.txt",
                                  universe, folder, fold, factor)


def create_gold_standard(name, unbalanced=False):
    universe = set()
    with open(f"../../../Datasets_PPIs/SwissProt/human_proteins_lengths.txt") as universe_file:
        for line in universe_file:
            protein, length = line.strip().split('\t')
            if float(length) > 50 and float(length) < 1000:
                universe.add(protein)

    ppis = set()
    if unbalanced:
        file_pos = f"../../../Datasets_PPIs/unbalanced_gold/{name}_pos.txt"
        file_neg = f"../../../Datasets_PPIs/unbalanced_gold/{name}_neg.txt"
        factor = 10
    else:
        file_pos = f"../../../Datasets_PPIs/Hippiev2.3/{name}_pos_rr.txt"
        file_neg = f"../../../Datasets_PPIs/Hippiev2.3/{name}_neg_rr.txt"
        factor = 1
    with open(file_pos, "r") as f_in_pos, \
            open(file_neg, "r") as f_in_neg:

        # Process positive examples
        ppi_counter = 0
        for line in f_in_pos:
            ppi_counter += 1
            columns = line.strip().split(" ")
            if columns[0] in universe and columns[1] in universe:
                ppis.add((columns[0], columns[1], '1'))
        print(f'{len(ppis)} positives in {name} of {ppi_counter}')

        # Process negative examples
        for line in f_in_neg:
            ppi_counter += 1
            columns = line.strip().split(" ")
            if columns[0] in universe and columns[1] in universe:
                ppis.add((columns[0], columns[1], '0'))
        print(f'{len(ppis)} overall in {name} of {ppi_counter}')

    ppis = balance_ppis_list(ppis, factor=factor)
    print(f'{len(ppis)} overall in {name} after balancing')
    if unbalanced:
        outfile = f"gold/{name}_unbalanced.txt"
    else:
        outfile = f"gold/{name}.txt"
    with open(outfile, "w") as f_out:
        for ppi in ppis:
            f_out.write(f"{ppi[0]}\t{ppi[1]}\t{ppi[2]}\n")


if __name__ == '__main__':
    # execute in the Datasets_PPIs/SwissProt directory:
    # awk '/^>/ {printf("%s\t",substr($0,2)); next;} {print length}' yeast_swissprot_oneliner.fasta > yeast_proteins_lengths.txt
    # awk '/^>/ {printf("%s\t",substr($0,2)); next;} {print length}' human_swissprot_oneliner.fasta > human_proteins_lengths.txt
    #create_gold_standard('Intra0')
    #create_gold_standard('Intra1')
    #create_gold_standard('Intra2')

    for dataset in ['guo', 'du']:
        print(dataset)
        '''
        create_dataset(dataset, 'original', 'train', 'yeast')
        create_dataset(dataset, 'original', 'test', 'yeast')
        create_dataset(dataset, 'rewired', 'train', 'yeast')
        create_dataset(dataset, 'rewired', 'test', 'yeast')
        create_dataset(dataset, 'partitions', 'partition_0', 'yeast')
        create_dataset(dataset, 'partitions', 'partition_1', 'yeast')
        create_dataset(dataset, 'partitions', 'partition_both', 'yeast')
        '''
        create_dataset(dataset, 'original', 'train', 'yeast', multiple_random_splits=True)
        create_dataset(dataset, 'original', 'test', 'yeast', multiple_random_splits=True)
        create_dataset(dataset, 'rewired', 'train', 'yeast', multiple_random_splits=True)
        create_dataset(dataset, 'rewired', 'test', 'yeast', multiple_random_splits=True)
    for dataset in ['huang', 'pan', 'richoux', 'dscript']:
        print(dataset)
        if dataset == 'richoux':
            '''
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
            '''
            create_dataset('richoux_regular', 'original', 'train', 'human', multiple_random_splits=True)
            create_dataset('richoux_regular', 'original', 'test', 'human', multiple_random_splits=True)
            create_dataset('richoux_regular', 'rewired', 'train', 'human', multiple_random_splits=True)
            create_dataset('richoux_regular', 'rewired', 'test', 'human', multiple_random_splits=True)
            create_dataset('richoux_strict', 'original', 'train', 'human', multiple_random_splits=True)
            create_dataset('richoux_strict', 'original', 'test', 'human', multiple_random_splits=True)
            create_dataset('richoux_strict', 'rewired', 'train', 'human', multiple_random_splits=True)
            create_dataset('richoux_strict', 'rewired', 'test', 'human', multiple_random_splits=True)
        else:
            '''
            create_dataset(dataset, 'original', 'train', 'human')
            create_dataset(dataset, 'original', 'test', 'human')
            create_dataset(dataset, 'rewired', 'train', 'human')
            create_dataset(dataset, 'rewired', 'test', 'human')
            create_dataset(dataset, 'partitions', 'partition_0', 'human')
            create_dataset(dataset, 'partitions', 'partition_1', 'human')
            create_dataset(dataset, 'partitions', 'partition_both', 'human')
            '''
            create_dataset(dataset, 'original', 'train', 'human', multiple_random_splits=True)
            create_dataset(dataset, 'original', 'test', 'human', multiple_random_splits=True)
            create_dataset(dataset, 'rewired', 'train', 'human', multiple_random_splits=True)
            create_dataset(dataset, 'rewired', 'test', 'human', multiple_random_splits=True)
