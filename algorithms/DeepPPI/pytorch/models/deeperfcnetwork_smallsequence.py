import torch
import torch.nn as nn
import torch.nn.functional as F



class SmallSequencesInteractionPredicterLongerNN(nn.Module):
    ''' 
    A neural network to do prediction of protein protein interaction
    on samples with sequences of 20 AA
    And a deeper network with more layers
    '''
    def __init__(self):
        super(SmallSequencesInteractionPredicter, self).__init__()
        self.fc1 = nn.Linear(400, 300)
        self.fc1b = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc2b = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc3b = nn.Linear(200, 100)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 2)

        self.batch1 = nn.BatchNorm1d(300)
        self.batch1b = nn.BatchNorm1d(300)
        self.batch2 = nn.BatchNorm1d(200)
        self.batch2b = nn.BatchNorm1d(200)
        self.batch3 = nn.BatchNorm1d(100)
        self.batch3b = nn.BatchNorm1d(100)
        self.batch4 = nn.BatchNorm1d(100)
        self.batch5 = nn.BatchNorm1d(50)
        

    def forward(self, x):
        test = x.split(1, dim=1)
        x1 = test[0]
        x1 = x1.contiguous()
        x1 = x1.view(x.size(0),-1)
        x2 = test[1]
        x2 = x2.contiguous()
        x2 = x2.view(x.size(0),-1)

        x1 = F.relu(self.fc1(x1))
        x1 = self.batch1(x1)
        x2 = F.relu(self.fc1b(x2))
        x2 = self.batch1b(x2)

        x1 = F.relu(self.fc2(x1))
        x1 = self.batch2(x1)
        x2 = F.relu(self.fc2b(x2))
        x2 = self.batch2b(x2)

        x1 = F.relu(self.fc3(x1))
        x1 = self.batch3(x1)
        x2 = F.relu(self.fc3b(x2))
        x2 = self.batch3b(x2)

        x = torch.cat((x1,x2), 1)
        x = F.relu(self.fc4(x))
        x = self.batch4(x)
        x = F.relu(self.fc5(x))
        x = self.batch5(x)
        x = F.softmax(self.fc6(x), 1)
        return x
