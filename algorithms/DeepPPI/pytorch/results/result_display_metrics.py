import sys
from matplotlib import pyplot as plt
import numpy as np

def usage():
    print("Usage: {} DATAFILE [savefile]".format(sys.argv[0]))
    sys.exit(1)

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


def get_metrics(data_file):

    results = open(data_file, 'r')
    accuracy = []
    precision = []
    recall = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for line in results.readlines():
        if line.startswith("True Positives: "):
            tp = int(line.split(" ")[2])
        if line.startswith("False Positives: "):
            fp = int(line.split(" ")[2])
        if line.startswith("True Negatives: "):
            tn = int(line.split(" ")[2])
        if line.startswith("False Negatives: "):
            fn = int(line.split(" ")[2])
            accuracy.append(float(tp+tn)/(tp+tn+fp+fn))
            precision.append(float(tp)/(tp+fp))
            recall.append(float(tp)/(tp+fn))

    return accuracy, precision, recall
    

if __name__ == '__main__':

    if len(sys.argv) != 2 and len(sys.argv) != 3 :
        usage()

    data_model = sys.argv[1]
    data = data_model + "_loss"

    if len(sys.argv) == 2:
        save = data_model + "_metrics.png"
    else:
        save = sys.argv[2]

    #recover infos from model
    model_name, optimiser, epochs, batch_size, learning_rate, momentum = get_model_info(data_model)

    graph_title = "{} with {} optimiser, {} learning rate and {} batch size".format(model_name, optimiser, learning_rate, batch_size)

    accuracy, precision, recall = get_metrics(data)

    t = np.linspace(0, epochs, num=epochs)

    fig, ax = plt.subplots()
    #ax.plot(t, training, testing)
    line, = ax.plot(t, accuracy, label ="Accuracy")
    line2, = ax.plot(t, precision, label ="Precision")
    line3, = ax.plot(t, recall, label ="Recall")
    ax.legend()
    
    ax.set(xlabel='epochs', ylabel='metrics',
           title=graph_title)
    #if no file name has been defined, use model name

    fig.savefig(save)
        
    plt.show()
