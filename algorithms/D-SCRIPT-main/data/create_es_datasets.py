import random
import os


def split_dataset(file):
    # reads in the file of protein pairs of format
    # protein1\t protein2\t label
    # and splits it 0.9/0.1 into train and validation sets

    ppis = []
    # read in the file
    with open(file, 'r') as f:
        for line in f:
            protein1, protein2, label = line.strip().split('\t')
            ppis.append((protein1, protein2, label))

    # shuffle the data
    random.shuffle(ppis)

    # split into train and validation sets
    train = ppis[:int(len(ppis)*0.9)]
    val = ppis[int(len(ppis)*0.9):]

    # write out the train and validation sets
    if file.endswith('_0.txt') or file.endswith('_both.txt'):
        new_train_filename = file.replace('.txt', '_train_es.txt')
        new_val_filename = file.replace('.txt', '_val_es.txt')
    else:
        new_train_filename = file.replace('.txt', '_es.txt')
        new_val_filename = file.replace('_train.txt', '_val_es.txt')

    with open(new_train_filename, 'w') as f:
        for ppi in train:
            f.write('\t'.join(ppi) + '\n')

    with open(new_val_filename, 'w') as f:
        for ppi in val:
            f.write('\t'.join(ppi) + '\n')


if __name__ == '__main__':
    # loop over all files in data folder and its subfolders and call split_dataset
    for root, dirs, files in os.walk('../data'):
        for file in files:
            if file.endswith('train.txt') or file.endswith('_0.txt') or file.endswith('_both.txt'):
                print(os.path.join(root, file))
                split_dataset(os.path.join(root, file))
