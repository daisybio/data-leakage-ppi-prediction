import torch
import torch.nn as nn
import torch.nn.functional as F


class IMFC2(nn.Module):
    '''A class defining a custom neural network to classify protein
    protein interaction with 2 sequences. Full connected
    '''
    
    def __init__(self):
        super(IMFC2, self).__init__()

        self.fc1 = nn.Linear(23320, 4000)
        self.fc1d = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc2d = nn.Dropout()
        self.fc3 = nn.Linear(1000, 200)
        self.fc3d = nn.Dropout()
        
        self.fc4 = nn.Linear(400, 200)
        self.fc4d = nn.Dropout()
        self.fc5 = nn.Linear(200, 100)
        self.fc5d = nn.Dropout()
        self.fc6 = nn.Linear(100, 2)
        self.fc6d = nn.Dropout(p=0.2)

        self.norm1a = nn.BatchNorm1d(4000)
        self.norm1b = nn.BatchNorm1d(1000)
        self.norm1c = nn.BatchNorm1d(200)

        self.norm3 = nn.BatchNorm1d(200)
        self.norm4 = nn.BatchNorm1d(100)


    def forward(self, x):
        #split input in 2
        test = x.split(1, dim=1)
        x1 = test[0]
        x1 = x1.contiguous()
        x1 = x1.view(x.size(0),-1)
        x2 = test[1]
        x2 = x2.contiguous()
        x2 = x2.view(x.size(0),-1)

        #go through both sides with same modules, so that weights are the same

        x1 = self.fc1d(x1)
        x1 = F.relu(self.fc1(x1))
        x1 = self.norm1a(x1)
        x1 = self.fc2d(x1)
        x1 = F.relu(self.fc2(x1))
        x1 = self.norm1b(x1)
        x1 = self.fc3d(x1)
        x1 = F.relu(self.fc3(x1))
        x1 = self.norm1c(x1)
        x2 = self.fc1d(x2)
        x2 = F.relu(self.fc1(x2))
        x2 = self.norm1a(x2)
        x2 = self.fc2d(x2)
        x2 = F.relu(self.fc2(x2))
        x2 = self.norm1b(x2)
        x2 = self.fc3d(x2)
        x2 = F.relu(self.fc3(x2))
        x2 = self.norm1c(x2)
        
        
        #rejoin them as one
        x = torch.cat((x1,x2), 1)
        x = self.fc4d(x)
        x = F.relu(self.fc4(x))
        x = self.norm3(x)
        x = self.fc5d(x)
        x = F.relu(self.fc5(x))
        x = self.norm4(x)
        x = self.fc6d(x)
        x = F.softmax(self.fc6(x), 1)
        return x
