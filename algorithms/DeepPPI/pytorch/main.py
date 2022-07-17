import deep as de
import utils as ut

import data as da
#import class_presence_dataset as class
import protein_sequence_dataset as prot

import models.convolution_networks as conv
import models.fcnetwork_smallsequence as small
import models.full_connected1 as fc1
import models.full_connected2 as fc2
import models.full_connected3 as fc3

#import models.rnn as rnn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method

import subprocess
import time
import argparse
import sys

#disable cudnn to see if it is the reason for the RAM memory leak
#torch.backends.cudnn.enabled = False


DEBUG = False

if __name__ == '__main__':
    
    parser = ut.make_parser()        
    args = parser.parse_args()
    save_file = args.f
    if save_file is not None:
        save_file = "results/" + save_file
    save_model = args.save
    if save_model is not None:
        save_model = "model_weights/" + save_model
    load_model = args.load
    epochs = args.epochs
    train_file = args.data
    gpu_to_use = int(args.gpu)
    learning_rate = args.lr
    momentum = args.m
    model_type = args.model
    optimizer = args.o
    BATCH_SIZE = args.b
    validation_file = args.val

    if model_type != 4 and model_type != 5:
        MAX_SIZE = 1166
    else:
        MAX_SIZE = 20
        
    start = time.time()
    
    #create the dataset :
    print("Creating the dataset.")
    dataset = prot.ProteinSequenceInteractionDataset(train_file, MAX_SIZE)
    print("Dataset created.")

    if DEBUG is True:
        #compute stats about dataset, positive = interaction, negative = no interaction:
        positive_information = 0
        negative_information = 0
        for data in dataset:
            if data['interaction'] == 1:
                positive_information += 1
            else:
                negative_information += 1
        percent_neg = negative_information / float(positive_information + negative_information)
        percent_pos = positive_information / float(positive_information + negative_information)
        print("Positive {}".format(positive_information))
        print("Negative {}".format(negative_information))

    print("Creating the dataloader.")
    dataloader = DataLoader(dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = True,
                            num_workers = 1, drop_last=True)
    print("Dataloader created.")
    
    if args.val:        
        print("Validation will begin after each training epoch.")
        validation_batch = 1
        print("Creating validation dataset")
        validation_dataset = prot.ProteinSequenceInteractionDataset(validation_file, MAX_SIZE)
        print("Test dataset created.\nCreating test dataloader.")
        validation_dataloader = DataLoader(validation_dataset,
                                 batch_size = validation_batch,
                                 shuffle = False,
                                 num_workers = 1, drop_last=True)
        print("Test dataloader created.")
    
    print("Creating the neural network model.")
    if model_type == 0:
        model = conv.IMConv()
    elif model_type == 1:
        model = fc1.IMFC()
    elif model_type == 2:
        model = fc2.IMFC2()
    #avoid 3 currently has memory issues 
    elif model_type == 3:
        model = conv.IMConvPlus()
    #this model doesn't use the same dataset
    elif model_type == 4:
        model = small.SmallSequencesInteractionPredicter()
    #rnn on 20AA sized sequence
    elif model_type == 5:
        model = nn.RNN(800, 40, 1, batch_first=True)
    elif model_type == 6:
        model = nn.RNN(46640, 2332, 1, batch_first=True)
    elif model_type == 7:
        model = fc3.IMFC3()
        print("Model created.")

    #load the model if the file has been provided
    if load_model is not None:
        print("Loading saved model.")
        model.load_state_dict(torch.load(load_model))
        print("Model loaded.")
    else:
        #Otherzise apply weight init
        if model_type == 5 or model_type == 6:
             print("Initializing hidden layers")
             hidden = torch.zeros(1, BATCH_SIZE, model.hidden_size).cuda(gpu_to_use)
             print("Hidden layers initialized")
        else:
            print("Initializing model weights.")
            model.apply(de.weights_init)
            print("Model weights initialized.")

    #check gpu availability, can force cpu use with -1 as gpu
    cuda = torch.cuda.is_available()
    if cuda and gpu_to_use > -1:
        use_cuda = True
        torch.cuda.device(gpu_to_use)
        torch.cuda.seed()
        device = torch.device("cuda", gpu_to_use)
        print("Using CUDA.")
    else:
        use_cuda = False
        device = torch.device("cpu")
        print("Using CPU.")

    net = model.to(device)
    
    try:
        save_file_loss = save_file + "_loss"
        #training the model going through the data several times
        for i in range(epochs):
            #get the loss on the epoch
            print ("Current Epoch:{}".format(i))
            if model_type != 5 and model_type != 6:
                training_res = de.train(net, dataloader, BATCH_SIZE, MAX_SIZE, device, learning_rate, momentum, optimizer)
                
                if args.val:
                    validation_res = de.validate(net, validation_dataloader, validation_batch, MAX_SIZE, device)
                else:
                    validation_res = []
                ut.save_data(save_file_loss, training_res, validation_res, i, DEBUG)
            else:
                training_res = de.train_rnn(net, dataloader, BATCH_SIZE, device, learning_rate, hidden)
                #TODO: validation for rnn
                ut.save_data(save_file_loss, training_res, [], i, DEBUG)
                    
        end = time.time()
        elapsed = end - start
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        print("Epochs: {}\nBatches:{}".format(epochs, BATCH_SIZE))
        
        #sauvegarde les résultats
        ut.save_parameters(save_file,
                           epochs,
                           elapsed,
                           BATCH_SIZE,
                           model.__repr__(),
                           learning_rate,
                           momentum,
                           optimizer)
        #sauvegarde le modèle
        if save_model is not None:
            print("SAVING")
            torch.save(model.state_dict(), save_model)
        
    except KeyboardInterrupt as e :
        save_interrupted = save_file + "Unfinished"
        cut_end = time.time()
        elapsed = cut_end - start
        ut.save_parameters(save_interrupted,
                     epochs,
                     elapsed,
                     BATCH_SIZE,
                     model.__repr__(),
                     learning_rate,
                     momentum,
                     optimizer)
        message = e
        ut.error_output(message)
        print("Interupted by keyboard")
        
    except RuntimeError as e:
        message = e
        ut.error_output(message)
        print("The network encountered an error. Check logs")
