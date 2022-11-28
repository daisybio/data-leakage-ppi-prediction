import os
import sys
import datetime
import argparse
from contextlib import redirect_stdout
from time import time

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import data_loader as dl
# import mail_callback
from models.lstm32_3conv3_2dense import LSTM32_3Conv3_2Dense
from models.lstm32_2conv3_2dense_shared import LSTM32_2Conv3_2Dense_S
from models.lstm32_2conv3_4dense_shared import LSTM32_2Conv3_4Dense_S
from models.lstm32_3conv3_2dense_shared import LSTM32_3Conv3_2Dense_S
from models.lstm32_3conv4_2dense_shared import LSTM32_3Conv4_2Dense_S
from models.lstm32_3conv3_3dense_shared import LSTM32_3Conv3_3Dense_S
from models.lstm64_3conv3_2dense_shared import LSTM64_3Conv3_2Dense_S
from models.lstm64drop_3conv3_3dense_shared import LSTM64Drop_3Conv3_3Dense_S
from models.lstm64x2_3conv3_10dense_shared import LSTM64x2_3Conv3_10Dense_S
from models.lstm64x2_embed2_10dense_shared import LSTM64x2_Embed2_10Dense_S
from models.lstm64x2_embed4_10dense_shared import LSTM64x2_Embed4_10Dense_S
from models.fc6_embed3_2dense import FC6_Embed3_2Dense
from models.fc2_2dense import FC2_2Dense
from models.fc2_100_2dense import FC2_100_2Dense
from models.fc2_20_2dense import FC2_20_2Dense
from models.fc2_2_2dense import FC2_2_2Dense
from models.conv3_3_2dense_shared import Conv3_3_2Dense_S

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers as opti
from tensorflow.keras.utils import plot_model


# from tensorflow.keras.utils import multi_gpu_model
# import tensorflow.keras.backend.tensorflow_backend as KTF

def usage():
    print(
        "Usage: {} [train_set OR load_weights + test_set] <OPTIONS>\nEnter {} -h to have the list of optional arguments".format(
            sys.argv[0], sys.argv[0]))
    sys.exit(1)


## From https://stackoverflow.com/a/43357954/2007142
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def metrics(y_true, y_pred):
    # Count positive samples.
    diff = y_true + y_pred - 1
    true_positive = sum(diff == 1)
    pred_positive = sum(y_pred == 1)
    real_positive = sum(y_true == 1)

    # print('TP={}, pred pos={}, real pos={}'.format(true_positive, pred_positive, real_positive))

    # If there are no true samples, fix the F1 score at 0.
    if real_positive == 0:
        return 0

    # How many selected items are relevant?
    precision = true_positive / pred_positive

    # How many relevant items are selected?
    recall = true_positive / real_positive

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


# This factory also returns a string with the model name mainly to avoid
# the situation where we miswrite a model name and we are applying the
# default fc_flat without being aware of it
def factory_model(model_name):
    if model_name == 'lstm32_3conv3_2dense':
        return LSTM32_3Conv3_2Dense(), 'lstm32_3conv3_2dense'
    elif model_name == 'lstm32_2conv3_2dense_shared':
        return LSTM32_2Conv3_2Dense_S(), 'lstm32_2conv3_2dense_shared'
    elif model_name == 'lstm32_2conv3_4dense_shared':
        return LSTM32_2Conv3_4Dense_S(), 'lstm32_2conv3_4dense_shared'
    elif model_name == 'lstm32_3conv3_2dense_shared':
        return LSTM32_3Conv3_2Dense_S(), 'lstm32_3conv3_2dense_shared'
    elif model_name == 'lstm32_3conv4_2dense_shared':
        return LSTM32_3Conv4_2Dense_S(), 'lstm32_3conv4_2dense_shared'
    elif model_name == 'lstm32_3conv3_3dense_shared':
        return LSTM32_3Conv3_3Dense_S(), 'lstm32_3conv3_3dense_shared'
    elif model_name == 'lstm64_3conv3_2dense_shared':
        return LSTM64_3Conv3_2Dense_S(), 'lstm64_3conv3_2dense_shared'
    elif model_name == 'lstm64drop_3conv3_3dense_shared':
        return LSTM64Drop_3Conv3_3Dense_S(), 'lstm64drop_3conv3_3dense_shared'
    elif model_name == 'lstm64x2_3conv3_10dense_shared':
        return LSTM64x2_3Conv3_10Dense_S(), 'lstm64x2_3conv3_10dense_shared'
    elif model_name == 'lstm64x2_embed2_10dense_shared':
        return LSTM64x2_Embed2_10Dense_S(), 'lstm64x2_embed2_10dense_shared'
    elif model_name == 'lstm64x2_embed4_10dense_shared':
        return LSTM64x2_Embed4_10Dense_S(), 'lstm64x2_embed4_10dense_shared'
    elif model_name == 'fc6_embed3_2dense':
        return FC6_Embed3_2Dense(), 'fc6_embed3_2dense'
    elif model_name == 'fc2_2dense':
        return FC2_2Dense(), 'fc2_2dense'
    elif model_name == 'fc2_100_2dense':
        return FC2_100_2Dense(), 'fc2_100_2dense'
    elif model_name == 'fc2_20_2dense':
        return FC2_20_2Dense(), 'fc2_20_2dense'
    elif model_name == 'fc2_2_2dense':
        return FC2_2_2Dense(), 'fc2_2_2dense'
    elif model_name == 'conv3_3_2dense_shared':
        return Conv3_3_2Dense_S(), 'conv3_3_2dense_shared'
    else:
        print("Model unknown. Terminating.")
        sys.exit(1)


# This factory also returns a string with the optimizer name mainly to avoid
# the situation where we miswrite a optimizer name and we are applying the
# default adam without being aware of it
def factory_optimizer(optimizer_name, lr=0.001):
    if optimizer_name == 'sgd':
        return opti.SGD(learning_rate=lr), 'sgd'
    elif optimizer_name == 'rmsprop':
        return opti.RMSprop(learning_rate=lr), 'rmsprop'
    elif optimizer_name == 'adagrad':
        return opti.Adagrad(learning_rate=lr), 'adagrad'
    elif optimizer_name == 'adadelta':
        return opti.Adadelta(learning_rate=lr), 'adadelta'
    elif optimizer_name == 'adamax':
        return opti.Adamax(learning_rate=lr), 'adamax'
    elif optimizer_name == 'nadam':
        return opti.Nadam(learning_rate=lr), 'nadam'
    else:
        return opti.Adam(learning_rate=lr), 'adam'


def make_parser():
    '''
    Parsing function for the training and validation of networks
    '''
    parser = argparse.ArgumentParser(description='Protein-Protein interaction predicter')
    parser.add_argument('-train_pos', type=str, help='File containing the positive training set')
    parser.add_argument('-train_neg', type=str, help='File containing the negative training set')
    parser.add_argument('-val_pos', type=str, help='File containing the positive validation set')
    parser.add_argument('-val_neg', type=str, help='File containing the negative validation set')
    parser.add_argument('-test_pos', type=str, help='File containing the positive test set')
    parser.add_argument('-test_neg', type=str, help='File containing the negative test set')
    parser.add_argument('-model', type=str,
                        help='choose among: lstm32_3conv3_2dense, lstm32_2conv3_2dense_shared, lstm32_3conv3_2dense_shared, lstm32_2conv3_4dense_shared, lstm32_3conv4_2dense_shared, lstm64_3conv3_2dense_shared, lstm64drop_3conv3_3dense_shared, lstm64x2_3conv3_10dense_shared, lstm64x2_embed2_10dense_shared, lstm64x2_embed4_10dense_shared, fc6_embed3_2dense, fc2_2dense, fc2_100_2dense, fc2_20_2dense, fc2_2_2dense, conv3_3_2dense_shared')
    parser.add_argument('-epochs', type=int, default=50, help='Number of epochs [default: 50]')
    parser.add_argument('-batch', type=int, default=64, help='Batch size [default: 64]')
    parser.add_argument('-patience', type=int, default=0,
                        help='Number of epochs before triggering the early stopping criterion [default: infinite patience]')
    parser.add_argument('-optimizer', type=str, default='adam',
                        help='Choose among: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam [default: adam]')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate [default: 0.001]')
    parser.add_argument('-gpu', type=int, default=0, help='If you have several GPUs, which one to use [default: 0]')
    parser.add_argument('-nb_gpu', type=int, default=1,
                        help='Number of GPU devices to use. Incompatible with the -gpu option [default: 1]')
    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                        help='To save weights of your model')
    parser.add_argument('-tensorboard', type=str2bool, nargs='?', const=True, default=False,
                        help='Save logs for TensorBoard')
    # parser.add_argument('-mail', type=str2bool, nargs='?', const=True, default=False, help='To automatically send an e-mail once training is over (private_mail_data.txt must be set properly)')
    parser.add_argument('-load', type=str,
                        help='File containing weights to load. You must also give a test set with this option.')
    parser.add_argument('-name', type=str,
                        help='Name complement to produced files, written at the end of the name file.')
    return parser


def parse_ppis(pos_file, neg_file, seq_dict, max_len=1166):
    ppis = []
    with open(pos_file, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            id0 = line_split[0]
            id1 = line_split[1]
            label = '1'
            if seq_dict.get(id0) is None or seq_dict.get(id1) is None:
                continue
            if len(seq_dict.get(id0)) > max_len or len(seq_dict.get(id1)) > max_len:
                continue
            ppis.append([id0, id1, label])
    with open(neg_file, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            id0 = line_split[0]
            id1 = line_split[1]
            label = '0'
            if seq_dict.get(id0) is None or seq_dict.get(id1) is None:
                continue
            if len(seq_dict.get(id0)) > max_len or len(seq_dict.get(id1)) > max_len:
                continue
            ppis.append([id0, id1, label])
    return ppis


def custom_load_data(ppis, seq_dict, max_size=1166):
    nr_ppis = len(ppis)
    X = [np.zeros(shape=(nr_ppis, max_size, 24), dtype=np.int8), np.zeros(shape=(nr_ppis, max_size, 24), dtype=np.int8)]
    y = np.zeros(shape=(nr_ppis, ), dtype=np.int8)
    idx = 0
    for ppi in ppis:
        X[0][idx] = dl.one_hot(dl.sequence2array(seq_dict[ppi[0]]), max_size, num_classes=24)
        X[1][idx] = dl.one_hot(dl.sequence2array(seq_dict[ppi[1]]), max_size, num_classes=24)
        y[idx] = ppi[2]
        idx += 1
    return X, y


def read_in_seqdict(organism):
    if organism == 'yeast':
        path = '../../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta'
    else:
        path = '../../../Datasets_PPIs/SwissProt/human_swissprot_oneliner.fasta'
    seq_dict = {}
    line_count = 0
    last_id = ''
    for line in open(path, 'r'):
        if line_count % 2 == 0:
            #id line
            last_id = line.strip().split('>')[1]
        else:
            seq = line.strip()
            seq_dict[last_id] = seq
        line_count += 1
    return seq_dict


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    accuracy = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    recall = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    f1_score = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score


def write_results(path, y_true, y_pred):
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(' ===========  test ===========')
    y_pred = np.round(y_pred).astype(np.int8)
    auc_test = roc_auc_score(y_true, y_pred)
    pr_test = average_precision_score(y_true, y_pred)
    tp_test, fp_test, tn_test, fn_test, accuracy_test, precision_test, sensitivity_test, recall_test, specificity_test, MCC_test, f1_score_test = calculate_performace(
        len(y_pred), y_pred, y_true)

    scores = {'Accuracy': [round(accuracy_test, 4)],
              'Precision': [round(precision_test, 4)],
              'Recall': [round(recall_test, 4)],
              'Specificity': [round(specificity_test, 4)],
              'MCC': [round(MCC_test, 4)],
              'F1': [round(f1_score_test, 4)],
              'AUC': [round(auc_test, 4)],
              'AUPR': [round(pr_test, 4)]}

    sc = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 4,
                           ):
        print(sc)
    sc.to_csv(path, mode='a', header=False)


if __name__ == '__main__':
    t_start = time()
    # To make sure TS is only booking the right amount of GPU memory, instead of all memory available
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    parser = make_parser()
    args = parser.parse_args()
    model_name = args.model
    epochs = int(args.epochs)
    number_gpu = int(args.nb_gpu)
    which_gpu = '/gpu:' + str(args.gpu)
    batch_size = int(args.batch) * number_gpu
    patience = args.patience
    optimizer_name = args.optimizer
    lr = args.lr

    file_name = args.name

    train_set_pos = args.train_pos
    train_set_neg = args.train_neg
    val_set_pos = args.val_pos
    val_set_neg = args.val_neg
    test_set_pos = args.test_pos
    test_set_neg = args.test_neg

    if int(patience) == 0:
        patience = args.epochs


    # Result files will be saved using a name starting with file_name
    now = datetime.datetime.now()
    if 'du' in file_name or 'guo' in file_name:
        organism='yeast'
    else:
        organism='human'
    seq_dict = read_in_seqdict(organism=organism)
    ppis_train = parse_ppis(train_set_pos, train_set_neg, seq_dict)
    ppis_test = parse_ppis(test_set_pos, test_set_neg, seq_dict)
    print("Loading training data")
    train_data, labels = custom_load_data(ppis_train, seq_dict)
    print(f'{len(labels)} protein pairs in training ({sum(labels)}/{len(labels)-sum(labels)})!')
    if val_set_pos:
        print("Loading validation data")
        ppis_val = parse_ppis(val_set_pos, val_set_neg, seq_dict)
        val_data, val_labels = custom_load_data(ppis_val, seq_dict)
        val_data = (val_data, val_labels)
        callbacks_list = [
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0008, cooldown=1,
                                        verbose=1)]
        callbacks_list.append(callbacks.EarlyStopping(monitor='val_acc', patience=patience, verbose=1))
    else:
        callbacks_list = [callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, min_lr=0.0008, cooldown=1,
                                                      verbose=1),
                          callbacks.EarlyStopping(monitor='acc', patience=patience, verbose=1)]

    with tf.device(which_gpu):
        # Build one model among available ones
        abstract_model, model_name = factory_model(model_name)
        model = abstract_model.get_model()

    optimizer, optimizer_name = factory_optimizer(optimizer_name, lr)
    print(f'Model: {model_name}')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    print("Training model")
    if val_set_pos:
        history = model.fit(train_data,
                            labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks_list,
                            validation_data=val_data)
    else:
        history = model.fit(train_data,
                            labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks_list)

    print("Loading test data")
    test_data, test_labels = custom_load_data(ppis_test, seq_dict)
    print(f'{len(test_labels)} protein pairs in test ({sum(test_labels)}/{len(test_labels)-sum(test_labels)})!')
    with open(f'results_custom/{file_name}.csv', 'w') as f:
        f.write(f'variable,value\n')
        f.write(f'n,{len(labels)+len(test_labels)}\n')
        f.write(f'n_pos,{sum(labels)+sum(test_labels)}\n')
        f.write(f'n_neg,{len(labels)+len(test_labels)-sum(labels)-sum(test_labels)}\n')
        f.write(f'n_train,{len(labels)}\n')
        f.write(f'n_train_pos,{sum(labels)}\n')
        f.write(f'n_train_neg,{len(labels) - sum(labels)}\n')
        f.write(f'n_test,{len(test_labels)}\n')
        f.write(f'n_test_pos,{sum(test_labels)}\n')
        f.write(f'n_test_neg,{len(test_labels) - sum(test_labels)}\n')
    score, acc = model.evaluate(test_data, test_labels)
    predict = model.predict(test_data, batch_size=batch_size, verbose=1)
    predict = np.reshape(predict, -1)
    print('Exporting results ...')
    write_results(path=f'results_custom/{file_name}.csv', y_true=test_labels, y_pred=predict)
    with open(f'results_custom/all_times.txt', 'a+') as f:
        f.write(f'{file_name}\t{time() - t_start}\n')
