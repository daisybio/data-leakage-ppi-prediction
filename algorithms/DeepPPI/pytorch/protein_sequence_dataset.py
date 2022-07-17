import torch.utils.data as data
import data as d

import pandas as pnd
import numpy as np
import sys

import sys

class ProteinSequenceInteractionDataset(data.Dataset):
    '''
    A dataset to process the data and make it usable by the
    model  of neural network (Conv)
    datafile : path to the csv with the data
    '''
    
    def __init__(self, datafile, max_size):
        '''
        Initialises the dataset with
        datafile : file path of the data
        max_size : max size of the sequences
        '''
        try:
            d.has_proper_structure(datafile)
        except d.DataFileFormatException:
            print('Problem with the data structure')
            #TODO: do better than just catch it!
            sys.exit(0)
        if ',' in datafile:
            sep = ','
        else:
            sep = ' '
        self.dataframe = pnd.read_csv(datafile, sep, names=['Protein1', 'Protein2', 'Sequence1', 'Sequence2', 'HasInteraction'])
        self.maximum = max_size

    def __getitem__(self, index):
        '''
        Gets one sample from the elements of the dataset 
        and formats the sequences to be useable by the NN
        in the form of a cuda.tensor
        '''
        data = self.dataframe.iloc[index]
        seq1 = d.sequence_to_vector(data[2])
        seq2 = d.sequence_to_vector(data[3])
        d.padd_sequence(seq1, self.maximum)
        d.padd_sequence(seq2, self.maximum)
        tensor = d.tensorize(seq1, seq2)
        sample = {'name1': data[0],
                  'name2': data[1],
                  'tensor': tensor,
                  'interaction': data[4]}
        return sample
    
    def __len__(self):
        '''Returns the length of the dataset'''
        return len(self.dataframe)

