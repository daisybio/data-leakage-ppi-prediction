from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)

    
def train(net, dataloader, batch_size, max_size, device, learning_rate, momentum, optimizer_to_use):
    '''Train the network
    Parameters :
    net : the neural network
    dataloader : the DataLoader for the data
    batch_size : size of batches (divisor of #samples)
    max_size : size of biggest sample
    device : device to use (cuda:x or cpu)
    learning_rate : learning rate of the model
    momentum : momentum for sgd
    optimizer_to_use : pick the optimizer, 0 for sgd, 1 for adam
    '''
    net.train()
    
    #define the optimizer to use
    if optimizer_to_use == 0:
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum)
    elif optimizer_to_use == 1:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer_to_use == 2:
        optimizer = torch.optim.Adadelta(net.parameters())
                
    #loss function   
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    total_loss = 0.0
    i = 0

    for i_batch, sample_batched in enumerate(dataloader):
        # in the next line, anything else than -1 will make the network crash unless batch_size is 1
        out = net(sample_batched['tensor'].to(device).view(batch_size, -1, max_size * 20))
        target = sample_batched['interaction'].to(device)
        
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().item()
        total_loss += loss.cpu().item()
        i = i +1 
        if i_batch % 500 == 499:    # print every 500 mini-batches
            print('[%5d] loss: %.3f' %
                  ( i_batch + 1, running_loss / 500))
            running_loss = 0.0

    loss_avg = total_loss / i
    return loss_avg


def validate(net, dataloader, batch_size, max_size, device):
    '''Validation of the network against the validation dataset'''
    #pass the network in eval mode
    net.eval()

    criterion_loss = 0.0
    correct = 0.0
    preds = []
    targets = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            data = sample_batched['tensor'].view(batch_size, -1, max_size * 20).to(device)
            target = sample_batched['interaction'].to(device)
            output = net(data)
            debug, pred = torch.max(output, dim=1)
            #print("{} debug, {} pred \n".format(debug, pred))
            criterion_loss += criterion(output, target).item()
            preds.append(pred.data)
            targets.append(target.data)
                
    validate_dat_len = len(dataloader.dataset)
    criterion_loss /= validate_dat_len

    return (criterion_loss, preds, targets)


def train_rnn(net, dataloader, batch_size, device, learning_rate, hidden):
    '''Broken for now'''
    net.train()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    running_loss = 0.0
    i = 0
    for i_batch, sample_batched in enumerate(dataloader):
        hidden.detach()
        hidden = Variable(hidden.data, requires_grad=True)
        #400 = input size, 2 = 2 sequences
        data = sample_batched['tensor'].to(device).view(batch_size, -1, 800)
        temp_tar = sample_batched['interaction'].to(device)
        targets = []
        
        for i in range(batch_size):
            tar = torch.full((1, 40), temp_tar.data[i], dtype=torch.long).to(device)
            targets.append(tar)
        target = torch.cat((targets[0], targets[1]), 0).to(device)
        for i in range(2, len(targets)):
            target = torch.cat((target, targets[i]), 0).to(device)
            
        output, hidden = net(data, hidden)
        
        loss = criterion(output, target)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in net.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        running_loss += loss.cpu().item()
        total_loss += loss.cpu().item()
        i = i +1 
        if i_batch % 500 == 499:    # print every 500 mini-batches
            print('[%5d] loss: %.3f' %
                  ( i_batch + 1, running_loss / 500))
            running_loss = 0.0

    loss_avg = total_loss / i
            
            
    return loss_avg, hidden
