import torch.utils.data as data

import data as d
import collections
import sys


class ClassPresenceInteractionDataSet(data.Dataset):
    '''
    A dataset to process the data to implement ideas presented in 
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1700-2
    '''

    def __init__(self, datafile, max_size):
        try:
            d.has_proper_structure(datafile)
        except d.DataFileFormatException:
            print("Format problem")
            sys.exit(0)
            
        self.dataframe = pnd.read_csv(datafile, sep=' ', names=['Protein1', 'Protein2', 'Sequence1', 'Sequence 2', 'HasInteraction'])
        self.max_size = max_size
    
    def __getitem__(self, index):
        '''
        Gets one sample from the elements of the dataset
        and formats the sequences to be useable by the NN 
        in the form of a cuda.tensor
        '''
        data = self.dataframe.iloc[index]
        seq1 = cluster_sequence(data[2])
        seq2 = cluster_sequence(data[3])
        seq1 = calc_frequency(seq1)
        seq2 = calc_frequency(seq2)
        tensor = d.tensorize(seq1, seq2)
        sample = {'name1': data[0],
                  'name2': data[1],
                  'tensor': tensor,
                  'interaction': data[4]}
        return sample
    
    def __len__(self):
        '''Returns the length of the dataset'''
        return len(self.dataframe)
                    

def cluster_sequence(sequence):
    '''
    following the conjoint triad method of paper to replace sequence
    amino acids by their cluster group starting from 0 to 6
    '''
    ct_sequence =[]
    for item in sequence:
        if item is not None:
            if item is 'A' or item is 'G' or item is 'V':
                ct_sequence.append('0')
            elif item is 'I' or item is 'L' or item is 'F' or item is 'P':
                ct_sequence.append('1')
            elif item is 'Y' or item is 'M' or item is 'T' or item is 'S':
                ct_sequence.append('2')
            elif item is 'H' or item is 'N' or item is 'Q' or item is 'W':
                ct_sequence.append('3')
            elif item is 'R' or item is 'K':
                ct_sequence.append('4')
            elif item is 'D' or item is 'E':
                ct_sequence.append('5')
            elif item is 'C':
                ct_sequence.append('6')
    return ct_sequence


def calc_frequency(sequence):
    '''
    calculates the frequency of apparition of triad of amino acids
    over a sequence and returns an array of these triads 
    '''
    array = np.zeros(343)
    association = {}
    modulo = len(sequence) % 3
    triad_amount = 0
    #exceeding 1, or 2 amino acids are 'lost' information
    if modulo != 0:
        sequence = sequence[:len(sequence) - modulo]
    for i in range(0, len(sequence), 3):
        triad_amount += 1 
        triad = sequence[i] + sequence[i+1] + sequence[i+2]
        if triad in association:
            cur = association[triad]
            association[triad] = cur + 1
        else:
            association[triad] = 1
    association = collections.OrderedDict(sorted(association.items()))
    for key, value in association.items():
        array[int(key, 7)] = value / triad_amount 
    return array
