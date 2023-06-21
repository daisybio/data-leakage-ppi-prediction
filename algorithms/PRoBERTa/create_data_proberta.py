import sys
import pandas as pd


def make_swissprot_to_dict(path_to_swissprot):
    prefix_dict = {}
    seq_dict = {}
    header_line = False
    last_id = ''
    last_seq = ''
    n = 30
    f = open(path_to_swissprot, 'r')
    for line in f:
        if line.startswith('>'):
            if last_id != '':
                seq_dict[last_id] = last_seq
                last_seq = ''
            header_line = True
            uniprot_id = line.split('|')[1]
            last_id = uniprot_id
        elif header_line is True:
            last_seq += line.strip()
            first_n = line[0:n]
            if first_n in prefix_dict.keys():
                if isinstance(prefix_dict[first_n], list):
                    prefix_dict[first_n].append(last_id)
                else:
                    prefix_dict[first_n] = [prefix_dict[first_n], last_id]
            else:
                prefix_dict[first_n] = last_id
            header_line = False
        else:
            last_seq += line.strip()
    f.close()
    return prefix_dict, seq_dict


def transform_SPRINT_to_JSON(folder, dataset, fold, seq_dict):
    ppis = set()
    with open(f"../SPRINT/data/{folder}/{dataset}_{fold}_pos.txt", "r") as f_in_pos, \
            open(f"../SPRINT/data/{folder}/{dataset}_{fold}_neg.txt", "r") as f_in_neg:

        # Process positive examples
        for line in f_in_pos:
            columns = line.strip().split(" ")
            if seq_dict.get(columns[0]) != None and seq_dict.get(columns[1]) != None:
                seq1 = seq_dict.get(columns[0])
                seq2 = seq_dict.get(columns[1])
                ppis.add((seq1, seq2, 1))

        # Process negative examples
        for line in f_in_neg:
            columns = line.strip().split(" ")
            if seq_dict.get(columns[0]) != None and seq_dict.get(columns[1]) != None:
                seq1 = seq_dict.get(columns[0])
                seq2 = seq_dict.get(columns[1])
                ppis.add((seq1, seq2, 0))
    df = pd.DataFrame(ppis, columns=['from', 'to', 'link'])
    df.to_json(f'data/{folder}/{dataset}_{fold}.json')

if __name__ == "__main__":
    prefix_dict, seq_dict = make_swissprot_to_dict('../../Datasets_PPIs/SwissProt/yeast_swissprot.fasta')
    transform_SPRINT_to_JSON('original', 'guo', 'train', seq_dict)
    transform_SPRINT_to_JSON('original', 'guo', 'test', seq_dict)