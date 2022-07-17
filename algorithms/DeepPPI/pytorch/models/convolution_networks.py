import torch
import torch.nn as nn
import torch.nn.functional as F

class IMConv(nn.Module):
    '''A class to define a custom neural network to classify if there is
    interaction between two proteins with their sequences
    '''

    def __init__(self):
        '''Initializes the Neural Network
        the value of input for the second convolution
        depends on the size of the input 
        '''
        super(IMConv, self).__init__()
        self.conv1 = nn.Conv1d(2, 20, 80, stride=20)
        self.conv1d = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(20, 10, 80, stride=20)
        self.conv2d = nn.Dropout()
        
        # (MAX_SIZE - kernel size +1) / 2
        self.fc0 = nn.Linear(130, 60)
        self.fc0d = nn.Dropout()
        self.fc1 = nn.Linear(60, 2)
        self.fc1d = nn.Dropout(p=0.2)

        self.batch_norm1 = nn.BatchNorm1d(20)
        self.batch_norm2 = nn.BatchNorm1d(10)
        self.batch_norm3 = nn.BatchNorm1d(60)
    
    def forward(self, x):
        '''Activations'''
        #print(x.size())
        x = self.conv1d(x)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = self.batch_norm1(x)
        x = self.conv2d(x)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.batch_norm2(x)
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.fc0d(x)
        x = F.relu(self.fc0(x))
        x = self.batch_norm3(x)
        x = self.fc1d(x)
        x = self.fc1(x)
        x = F.softmax(x, 1)
        return x

    
class IMConvPlus(nn.Module):
    '''A class defining a custom neural network to classify protein
    protein interaction from two sequences using convolutions
    of a kernel with 20 amino acids
    '''

    def __init__(self):
        super(IMConvPlus, self).__init__()
        self.conv1 = nn.Conv1d(2, 10, 400, stride=20)
        self.conv1d = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(10, 10, 400, stride=20)
        self.conv2d = nn.Dropout()
        # (MAX_SIZE - kernel size +1) / 2
        self.fc0 = nn.Linear(40, 20)
        self.fc0d = nn.Dropout()
        self.fc1 = nn.Linear(20, 2)
        self.fc1d = nn.Dropout(p=0.2)
        #batch norm layers (20 * sorties ?)
        self.batch_norm1 = nn.BatchNorm1d(10)
        self.batch_norm2 = nn.BatchNorm1d(10)
        self.batch_norm3 = nn.BatchNorm1d(20)

    def forward(self, x):
        '''Activations'''
        x = self.conv1d(x)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = self.batch_norm1(x)
        x = self.conv2d(x)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.batch_norm2(x)
        x = x.view(x.size(0),-1)
        x = self.fc0d(x)
        x = F.relu(self.fc0(x))
        x = self.batch_norm3(x)
        x = self.fc1d(x)
        x = self.fc1(x)
        x = F.softmax(x, 1)
        return x
