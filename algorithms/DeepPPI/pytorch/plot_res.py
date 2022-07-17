import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_model_info(model_file_name):
    model_info = open(model_file_name, 'r')
    line_c = 0
    model_name = ''
    optimiser = ''
    epochs = 0
    batch_size = 0
    learning_rate = 0.0
    momentum = 0.0
    for line in model_info.readlines():
        line_c += 1
        one_line = line.rstrip('\n')
        if line_c == 2:
            model_name = one_line.rstrip('(')
        elif one_line.startswith("Optimizer"):
            elements = one_line.split(' ')
            optimiser = elements[1]
        elif one_line.startswith("Epochs"):
            elements = one_line.split(' ')
            epochs = int(elements[1])
        elif one_line.startswith("Size of batches:"):
            elements = one_line.split(' ')
            batch_size = int(elements[3])
        elif one_line.startswith("Learning rate"):
            elements = one_line.split(' ')
            learning_rate = float(elements[2])
        elif one_line.startswith("Momentum"):
            elements = one_line.split(' ')
            momentum = float(elements[1])
    return model_name, optimiser, epochs, batch_size, learning_rate, momentum


def get_losses(save_file):

    results = open(save_file, 'r')
    training = []
    testing = []
    recall = 0.0
    precision = 0.0
    cpt = 0
    for line in results.readlines():
        line_elements = line.split(' ')
        print(line_elements)
        if cpt == 0:
            training.append(float(line_elements[3]))
            cpt += 1
        elif cpt == 1:
            testing.append(float(line_elements[3]))
            cpt += 1
        elif cpt == 2:
            cpt += 1
        elif cpt == 3:
            cpt += 1
        elif cpt == 4:
            cpt += 1
        elif cpt == 5:
            cpt += 1
        elif cpt == 6:
            recall = float(line_elements[1])
            precision = float(line_elements[3])
            cpt = 0
    return training, testing
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plots graphs from the PPI results.')
    parser.add_argument('-f', type=str, help='save file name containing the results')
    parser.add_argument('-m', type=str, help='file name of the model')
    parser.add_argument('-s', type=str, help='name for save of figure')

    args = parser.parse_args()
    data = args.f
    model = args.m
    save = args.s
    
    #recover infos from model
    model_name, optimiser, epochs, batch_size, learning_rate, momentum = get_model_info(model)

    graph_title = "{} with {} optimiser, {} learning rate and {} batch size".format(model_name, optimiser, learning_rate, batch_size)

    training, testing = get_losses(data)

    
    t = np.linspace(0, epochs, num=epochs)

    fig, ax = plt.subplots()
    #ax.plot(t, training, testing)
    line, = ax.plot(t, training, label ="Training loss")
    line2, = ax.plot(t, testing, label ="Testing loss")
    ax.legend()
    
    ax.set(xlabel='epochs', ylabel='loss',
           title=graph_title)
    #if no file name has been defined, use model name
    if save is None:
        save = "{}.png".format(model_name)
    fig.savefig(save)
        
    plt.show()
