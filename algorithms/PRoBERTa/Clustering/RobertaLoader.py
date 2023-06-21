import torch
from torch.utils.data import Dataset

import numpy
import pandas as pd
import swifter

import fairseq
from fairseq.models.roberta import RobertaModel

class RobertaDataset(Dataset):
    ### Dataset that generates RoBERTa feature vectors on the fly from tokenized sequences.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.df):
            return None
        with torch.no_grad():
            return self.sumLayers(self.extract(
                self.df.loc[index, "Tokenized Sequence"]), self.layers)

    def encode(self, sequence):
        if self.use_cuda:
            torch.cuda.empty_cache()
            return self.roberta.task.dictionary.encode_line(sequence).type('torch.cuda.LongTensor')
        return self.roberta.task.dictionary.encode_line(sequence)

    def pad(self, sequence, length):
        ### Pads the encoded sequence to a length
        ### Truncates the encoded sequence if its length is greater than specified length
        if self.use_cuda:
            torch.cuda.empty_cache()
        if len(sequence) >= length:
            return sequence[:length]
        return torch.cat((sequence, torch.zeros(length - len(sequence), dtype = torch.long, device = self.device)))[:length]

    def extract(self, sequence):
        if self.use_cuda:
            torch.cuda.empty_cache()
        sequence = self.df[self.df['Tokenized Sequence']==sequence]['Sequence Embeddings'].to_numpy()[0].long()
        return self.roberta.extract_features(sequence, return_all_hiddens = True)

    def getLayer(self, feature, n):
        return feature[n]

    def getEmbedding(self, sequence):
        return self.sumLayers(self.extract(sequence),self.layers)

    def sumLayers(self, feature, layers):
        ### Returns a tensor with all requested layers of features summed element-wise
        combined = torch.zeros_like(feature[0], dtype = torch.float)
        for n in layers:
            if 0 <= n and n < len(feature):
                combined = torch.add(combined, feature[n])
            else:
                raise Exception("Error to access layer: " + str(n)
                    + ".Feature only contains " + str(len(feature)) + " layers.")
        return combined
    
    def __init__(self, model, weights, df, layers = [8, 9, 10, 11], length = 512):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
       
        self.roberta = RobertaModel.from_pretrained(model, checkpoint_file = weights)
        if self.use_cuda:
            self.roberta = self.roberta.cuda()
        self.roberta.eval()
        
        self.df = df.copy()
        self.layers = layers
        self.df["Sequence Embeddings"] = self.df["Tokenized Sequence"] \
                    .swifter.apply(self.encode) \
                    .swifter.apply(self.pad, length = length)
