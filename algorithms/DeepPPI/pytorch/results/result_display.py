import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_data(filename):
    data_file = open(filename, 'r')
    data_content = data_file.read()
    lines = data_content.splitlines()
    epochs = 0
    batch = 0
    learning = 0.0
    momentum = 0.0
    train_data = []
    test_data = []
    is_train_data = False
    is_test_data = False
    for idl, line in enumerate(lines):
        if is_train_data:
            if line.startswith("Testing Loss"):
                is_train_data = False
                is_test_data = True
            elif line.startswith(" "):
                train_data.append(float(line.split(" ")[3][:7]))
            else:
                #train_data.append(float(line.split(" ")[1][:7]))
                train_data.append(float(line.split(" ")[2][:7]))
                #train_data.append(float(line))
        elif line.startswith("Epochs: "):
            epochs = line.split(" ")[1]
        elif line.startswith("Size of batches:"):
            batch = line.split(" ")[1]
        elif line.startswith("Learning rate:"):
            learning = line.split(" ")[1]
        elif line.startswith("Momentum:"):
            momentum = line.split(" ")[1]
        elif line.startswith("Training") or line.startswith("Loss"):
            is_train_data = True
        elif is_test_data:
            if line.startswith(" "):
                test_data.append(float(line.split(" ")[3][:6]))
            else:
                test_data.append(float(line.split(" ")[2][:6]))
    return (train_data, test_data)


def make_figure(filename, title):
    data = get_data(filename)
    data1 = data[0]
    data2 = data[1]
    #fun
    plt.xkcd()
    if len(data[1]) == 0:
        fig, ax1 = plt.subplots()
    else:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax2.spines['right'].set_color('none')
        ax2.spines['top'].set_color('none')
        ax2.plot(data2)
        ax2.set_title("Testing")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
    
    fig.suptitle(title)
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.plot(data1)
    ax1.set_title("Training")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")


    savefile = filename + 'fig'
    plt.savefig(savefile)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein-Protein interaction predicter result Display')
    parser.add_argument('-f', type=str, help='file to plot')
    parser.add_argument('-t', type=str, default='Model', help='optional name for the title of display')
    args = parser.parse_args()
    filename = args.f
    title = args.t
    
    make_figure(filename, title)
