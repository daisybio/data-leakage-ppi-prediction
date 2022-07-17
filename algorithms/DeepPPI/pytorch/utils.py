import time
import argparse


def make_parser():
    '''
    Parsing function for the training and validation of networks
    '''
    parser = argparse.ArgumentParser(description='Protein-Protein interaction predicter')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('-val', type=str, help='if given, file containing validate samples')
    parser.add_argument('-f', type=str, default='temp.txt', help='save file name for results')
    parser.add_argument('-data', type=str, help='file name of dataset')
    parser.add_argument('-gpu', type=int, default=0, help='gpu device to use[default: 0], -1 to force cpu usage')
    parser.add_argument('-b', type=int, help='batch size, bigger batch = bigger memory needed')
    parser.add_argument('-m', type=float, default=0.0, help='momentum for the optimizer')
    parser.add_argument('-model', type=int, default=0, help='Choose the model you want to use. 0 = conv, 1 = FC, 2 = FC2, 3 = Conv(memory problems), 4 = Working on 20 AA sequencesm fc, 5 = RNN attempt')
    parser.add_argument('-o', type=int, default=0, help='Choose the optimizer to use, 0 for SGD, 1 for Adam, 2 for Adadelta. [default 0]')
    parser.add_argument('-save', type=str, help='name of file to save model weights into')
    parser.add_argument('-load', type=str, help='name of file to load model weights from, select the right model for it')
    return parser


def save_parameters(save_file, epoch, elapsed, batch_size, model, lr, momentum, optimizer):
    ''' Saves the parameters of the network in a file name 
    save_file : name of the file to save the data in
    epochs : number of epochs
    elapsed : time elapsed
    batch_size : size of batches
    model : structure of the neural network
    lr : learning rate of the network
    momentum : momentum for the learning
    optimizer : an int representing the optimizer 0 = SGD, 1 = Adam
    '''
    result = open(save_file, 'w')
    result.write("Model:\n{}".format(model))
    if optimizer == 0:
        opt = "SGD"
    elif optimizer == 1:
        opt = "Adam"
    elif optimizer == 2:
        opt = "Adadelta"
    result.write("\nOptimizer: {}".format(opt))
    result.write("\nEpochs: {}\n".format(epoch))
    result.write(time.strftime("%d:%H:%M:%S", time.gmtime(elapsed)))
    result.write("\nSize of batches: {} ".format(batch_size))
    result.write("\nLearning rate: {}".format(lr))
    result.write("\nMomentum: {} ".format(momentum))
    result.close()
    
    
def save_data(filename, train_values, validate_values, epoch, DEBUG):
    result = open(filename, 'a')
    result.write("Epoch: {} Loss: {}\n".format(epoch, train_values))
    if DEBUG is True:
        debug = open("debug", 'a')
        for i in range(len(validate_values)):
            predictions = validate_values[1]
            targets = validate_values[2]
            debug.write("epoch / predictions / cible\n")
            for i in range(len(predictions)):
                debug.write("{} / {} / {}\n".format(i, predictions[i], targets[i]))
            debug.close()
    if len(validate_values) > 0:
        result.write("Epoch: {} ValidationLoss: {}\n".format(epoch, validate_values[0]))
        for i in range(len(validate_values)):
            predictions = validate_values[1]
            targets = validate_values[2]
            true_pos = 0
            false_pos = 0
            false_neg = 0
            true_neg = 0
                
            for j in range(len(predictions)):
                pred = predictions[j].item()

                #print("Pred {}".format(pred))
                target = targets[j].item()

                #print("Target {}".format(target))
                if pred >= 0.5 and target >= 0.5:
                    true_pos += 1
                elif pred < 0.5 and target < 0.5:
                    true_neg += 1
                elif pred >= 0.5 and target < 0.5:
                    false_pos += 1
                elif pred < 0.5 and target >= 0.5:
                    false_neg += 1 
                    
        result.write("True Positives: {}\n".format(true_pos))
        result.write("False Positives: {}\n".format(false_pos))
        result.write("True Negatives: {}\n".format(true_neg))
        result.write("False Negatives: {}\n".format(false_neg))

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        
        result.write("Accuracy {} Precision {} Recall {}\n".format(accuracy, precision, recall))
    
    
    result.close()
    
def error_output(message):
    result = open("errorLog", 'w')
    result.write(time.strftime("%H:%M:%S", time.gmtime(time.time())))
    result.write("\nError broke the computing : {}\n".format(message))
    
        
