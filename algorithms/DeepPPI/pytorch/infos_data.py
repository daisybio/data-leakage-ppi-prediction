from __future__ import print_function
import data
import pandas as pd
import argparse
import numpy as np
import csv

def max_seq(filename):
    return data.max_sequence_size(filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Protein Sequence stats')
    parser.add_argument('-f', type=str, help='datafile to use')
    parser.add_argument('-s', type=str, help='file to save')
    parser.add_argument('-l', type=int, help='max sequence size to keep')
    parser.add_argument('-stats', default=False, help='True to display stats(default:False)')
    args = parser.parse_args()
    
    data_file = args.f
    save_file = args.s
    sequence_size = args.l
    stats = args.stats
    if stats == False:
        
        d = csv.writer(open(save_file, "w"))
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                lines = line[0].split(" ")
                if len(lines[2]) < sequence_size and len(lines[3]) < sequence_size:
                    d.writerow(line)
    else:
        data_cols =['Prot1', 'Prot2', 'Seq1', 'Seq2', 'Interaction']
        data_set = pd.read_csv(data_file, sep=' ', usecols=[0,1,2,3,4], header=None, names=data_cols)
        print("Le fichier comprend ces quantites de donnees avec interaction(1) et sans (0)\n")
        
        zeroes = data_set[data_set['Interaction'] == 0]
        ones = data_set[data_set['Interaction'] == 1]
        count_zero = len(zeroes.count(1))
        count_ones = len(ones.count(1))
        total = len(data_set)
        print(count_zero)
        print(count_ones)
        print(total)
        if total == count_ones + count_zero:
            print("ok")
        else:
            print("error")

        proportion_zero = count_zero / total * 100
        print("Proportion de zero sur le total : {}".format(proportion_zero))
        proportion_ones = count_ones / total * 100
        print("Proportion de uns sur le total: {}".format(proportion_ones))
        dcopy = data_set.copy()
        dcopy = data_set.sort_values(by='Interaction')

        train_size = total / 100 * 80
        test_size = total - train_size
        
        
