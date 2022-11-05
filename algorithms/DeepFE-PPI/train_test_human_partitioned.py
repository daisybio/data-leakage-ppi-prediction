# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:07:38 2018

@author: yaoyu
"""
import keras
from keras.layers import BatchNormalization, Dense, Dropout, concatenate
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import utils.tools as utils
from keras.regularizers import l2
import pandas as pd
from gensim.models.word2vec import Word2Vec
import copy
import h5py
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
import psutil
import os
import tensorflow as tf
import sklearn


#### PLOT FUNCTIONS

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def plot(length):
    reversed_length = sorted(length, reverse=True)
    x = np.linspace(0, len(length), len(length))
    plt.plot(x, reversed_length)

    plt.title('line chart')
    plt.xlabel('x')
    plt.ylabel('reversed_length')

    plt.show()


def max_min_avg_length(seq):
    length = []
    for string in seq:
        length.append(len(string))
    plot(length)
    maxNum = max(length)  # maxNum = 5
    minNum = min(length)  # minNum = 1

    avg = averagenum(length)

    print('The longest length of protein is: ' + str(maxNum))
    print('The shortest length of protein is: ' + str(minNum))
    print('The avgest length of protein is: ' + str(avg))


#### LOAD DATA FUNCTIONS

def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])
        token_dataset.append(seq)

    return token_dataset


def padding_J(protein, maxlen):
    padded_protein = copy.deepcopy(protein)
    for i in range(len(padded_protein)):
        if len(padded_protein[i]) < maxlen:
            for j in range(len(padded_protein[i]), maxlen):
                padded_protein[i].append('J')
    return padded_protein


def read_traingingData(file_name):
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i % 2 == 1:
                seq.append(line.split('\n')[0])
            i = i + 1
    return seq


def protein_representation(wv, tokened_seq_protein, maxlen, size):
    represented_protein = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j] == 'J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))
    return represented_protein


def get_training_dataset(wv, maxlen, size):
    file_1 = 'dataset/human/positive/Protein_A_0.txt'
    file_2 = 'dataset/human/positive/Protein_B_0.txt'
    file_3 = 'dataset/human/negative/Protein_A_0.txt'
    file_4 = 'dataset/human/negative/Protein_B_0.txt'
    # positive seq protein A
    pos_seq_protein_A = read_traingingData(file_1)
    pos_seq_protein_B = read_traingingData(file_2)
    neg_seq_protein_A = read_traingingData(file_3)
    neg_seq_protein_B = read_traingingData(file_4)
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)
    pos_neg_seq_protein_B = copy.deepcopy(pos_seq_protein_B)
    pos_neg_seq_protein_B.extend(neg_seq_protein_B)
    seq = []
    seq.extend(pos_neg_seq_protein_A)
    seq.extend(pos_neg_seq_protein_B)
    max_min_avg_length(seq)

    # token
    token_pos_neg_seq_protein_A = token(pos_neg_seq_protein_A)
    token_pos_neg_seq_protein_B = token(pos_neg_seq_protein_B)
    # padding
    tokened_token_pos_neg_seq_protein_A = padding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = padding_J(token_pos_neg_seq_protein_B, maxlen)
    # protein reprsentation
    feature_protein_A = protein_representation(wv, tokened_token_pos_neg_seq_protein_A, maxlen, size)
    feature_protein_B = protein_representation(wv, tokened_token_pos_neg_seq_protein_B, maxlen, size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A), np.array(feature_protein_B)))
    #  creat label
    label = np.ones(len(feature_protein_A))
    label[len(pos_seq_protein_A):] = 0

    return feature_protein_AB, label


def get_test_dataset(wv, maxlen, size, partition):
    if partition == 'partition_1':
        file_1 = 'dataset/human/positive/Protein_A_1.txt'
        file_2 = 'dataset/human/positive/Protein_B_1.txt'
        file_3 = 'dataset/human/negative/Protein_A_1.txt'
        file_4 = 'dataset/human/negative/Protein_B_1.txt'
    else:
        file_1 = 'dataset/human/positive/Protein_A_both.txt'
        file_2 = 'dataset/human/positive/Protein_B_both.txt'
        file_3 = 'dataset/human/negative/Protein_A_both.txt'
        file_4 = 'dataset/human/negative/Protein_B_both.txt'
    # positive seq protein A
    pos_seq_protein_A = read_traingingData(file_1)
    pos_seq_protein_B = read_traingingData(file_2)
    neg_seq_protein_A = read_traingingData(file_3)
    neg_seq_protein_B = read_traingingData(file_4)
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)
    pos_neg_seq_protein_B = copy.deepcopy(pos_seq_protein_B)
    pos_neg_seq_protein_B.extend(neg_seq_protein_B)
    seq = []
    seq.extend(pos_neg_seq_protein_A)
    seq.extend(pos_neg_seq_protein_B)
    max_min_avg_length(seq)

    # token
    token_pos_neg_seq_protein_A = token(pos_neg_seq_protein_A)
    token_pos_neg_seq_protein_B = token(pos_neg_seq_protein_B)
    # padding
    tokened_token_pos_neg_seq_protein_A = padding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = padding_J(token_pos_neg_seq_protein_B, maxlen)
    # protein reprsentation
    feature_protein_A = protein_representation(wv, tokened_token_pos_neg_seq_protein_A, maxlen, size)
    feature_protein_B = protein_representation(wv, tokened_token_pos_neg_seq_protein_B, maxlen, size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A), np.array(feature_protein_B)))
    #  creat label
    label = np.ones(len(feature_protein_A))
    label[len(pos_seq_protein_A):] = 0

    return feature_protein_AB, label


#### REWRITE DATASET FUNCTIONS

def export_all_seqs(pos_protein_A, pos_protein_B, neg_protein_A, neg_protein_B):
    prefix = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/human/'
    seq_dict = {}
    for index, row in pos_protein_A.iterrows():
        seq_dict[row['id']] = row['seq']
    for index, row in pos_protein_B.iterrows():
        seq_dict[row['id']] = row['seq']
    for index, row in neg_protein_A.iterrows():
        seq_dict[row['id']] = row['seq']
    for index, row in neg_protein_B.iterrows():
        seq_dict[row['id']] = row['seq']
    with open(prefix + 'all_seqs.fasta', 'w') as f:
        for key, value in seq_dict.items():
            f.write(f'>{key}\n{value}\n')


def make_swissprot_to_dict():
    prefix_dict = {}
    seq_dict = {}
    header_line = False
    last_id = ''
    last_seq = ''
    n = 30
    f = open('../../Datasets_PPIs/SwissProt/human_swissprot.fasta', 'r')
    for line in f:
        if line.startswith('>'):
            if last_id != '':
                seq_dict[last_id] = last_seq
                last_seq = ''
            header_line = True
            uniprot_id = line.split('|')[1]
            last_id = uniprot_id
        elif header_line is True:
            last_seq += line.strip()
            first_n = line[0:n]
            if first_n in prefix_dict.keys():
                if isinstance(prefix_dict[first_n], list):
                    prefix_dict[first_n].append(last_id)
                else:
                    prefix_dict[first_n] = [prefix_dict[first_n], last_id]
            else:
                prefix_dict[first_n] = last_id
            header_line = False
        else:
            last_seq += line.strip()
    f.close()
    return prefix_dict, seq_dict


def write_files(path_to_file, df, prefix_dict, seq_dict):
    prefix = '/Users/judithbernett/PycharmProjects/PPIs/DeepFE-PPI/dataset/human/'
    n = 30
    with open(prefix + path_to_file, 'w') as f:
        for index, row in df.iterrows():
            first_n = row['seq'][0:n]
            if first_n not in prefix_dict.keys():
                uniprot_id = ''
            elif isinstance(prefix_dict[first_n], list):
                uniprot_ids = prefix_dict[first_n]
                uniprot_id = ''
                for id in uniprot_ids:
                    if seq_dict[id] == row['seq']:
                        uniprot_id = id
                        break
            else:
                uniprot_id = prefix_dict[first_n]
            f.write(f'>old_id:{row["id"]}|sw:{uniprot_id}\n{row["seq"]}\n')


def export_human_proteins(pos_protein_A, pos_protein_B, neg_protein_A, neg_protein_B):
    # export_all_seqs(pos_protein_A, pos_protein_B, neg_protein_A, neg_protein_B)
    # map to uniprot
    print('Making swissprot dict ...')
    prefix_dict, seq_dict = make_swissprot_to_dict()
    print('Writing positive/Protein_A.txt....')
    write_files('positive/Protein_A.txt', pos_protein_A, prefix_dict, seq_dict)
    print('Writing positive/Protein_B.txt....')
    write_files('positive/Protein_B.txt', pos_protein_B, prefix_dict, seq_dict)
    print('Writing negative/Protein_A.txt....')
    write_files('negative/Protein_A.txt', neg_protein_A, prefix_dict, seq_dict)
    print('Writing negative/Protein_B.txt....')
    write_files('negative/Protein_B.txt', neg_protein_B, prefix_dict, seq_dict)


#### MODEL FUNCTIONS

def merged_DBN_functional(sequence_len):
    # left model
    model_left_input = keras.Input(shape=(sequence_len,), name='left')
    model_left = Dense(2048, activation='relu', kernel_regularizer=l2(0.01), name='left_dense1')(model_left_input)
    model_left = BatchNormalization(name='left_BN1')(model_left)
    model_left = Dropout(0.5, name='left_Dropout1')(model_left)
    model_left = Dense(1024, activation='relu', kernel_regularizer=l2(0.01), name='left_dense2')(model_left)
    model_left = BatchNormalization(name='left_BN2')(model_left)
    model_left = Dropout(0.5, name='left_Dropout2')(model_left)
    model_left = Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='left_dense3')(model_left)
    model_left = BatchNormalization(name='left_BN3')(model_left)
    model_left = Dropout(0.5, name='left_Dropout3')(model_left)
    model_left = Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='left_dense4')(model_left)
    model_left = BatchNormalization(name='left_BN4')(model_left)

    # right model
    model_right_input = keras.Input(shape=(sequence_len,), name='right')
    model_right = Dense(2048, activation='relu', kernel_regularizer=l2(0.01), name='right_dense1')(model_right_input)
    model_right = BatchNormalization(name='right_BN1')(model_right)
    model_right = Dropout(0.5, name='right_Dropout1')(model_right)
    model_right = Dense(1024, activation='relu', kernel_regularizer=l2(0.01), name='right_dense2')(model_right)
    model_right = BatchNormalization(name='right_BN2')(model_right)
    model_right = Dropout(0.5, name='right_Dropout2')(model_right)
    model_right = Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='right_dense3')(model_right)
    model_right = BatchNormalization(name='right_BN3')(model_right)
    model_right = Dropout(0.5, name='right_Dropout3')(model_right)
    model_right = Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='right_dense4')(model_right)
    model_right = BatchNormalization(name='right_BN4')(model_right)
    # together
    merged = concatenate([model_left, model_right])
    merged = Dense(8, activation='relu', kernel_regularizer=l2(0.01), name='merged_dense')(merged)
    merged = BatchNormalization(name='merged_BN')(merged)
    merged = Dropout(0.5, name='merged_Dropout')(merged)
    ppi_pred = Dense(2, activation='softmax', name='ppi_pred')(merged)

    model = keras.Model(inputs=[model_left_input, model_right_input],
                        outputs=[ppi_pred])
    model.summary()

    return model


# define the function
def training_vis(hist, i, plot_dir, swm, be):
    loss = hist.history['loss']
    # val_loss = hist.history['val_loss']
    precision = hist.history['precision']
    # val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    # ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Traingng Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(precision, label='train_precision')
    # ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision on Traingng Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + swm + be + '/round_' + str(i) + '.png')


#### UTILS

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def getMemorystate():
    phymem = psutil.virtual_memory()
    line = "Memory: %5s%% %6s/%s" % (phymem.percent,
                                     str(int(phymem.used / 1024 / 1024)) + "M",
                                     str(int(phymem.total / 1024 / 1024)) + "M")
    return line


def test_data(X_test_left, X_test_right, y_test):
    predictions_test = model.predict([X_test_left, X_test_right])

    auc_test = roc_auc_score(y_test[:, 1], predictions_test[:, 1])
    pr_test = average_precision_score(y_test[:, 1], predictions_test[:, 1])

    label_predict_test = utils.categorical_probas_to_classes(predictions_test)
    tp_test, fp_test, tn_test, fn_test, accuracy_test, precision_test, sensitivity_test, recall_test, specificity_test, MCC_test, f1_score_test, _, _, _ = utils.calculate_performace(
        len(label_predict_test), label_predict_test, y_test[:, 1])
    print(' ===========  test:')
    print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f' % (tp_test, fp_test, tn_test, fn_test))
    print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
          % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test,
             f1_score_test))
    print('\tauc=%0.4f,pr=%0.4f' % (auc_test, pr_test))


# %%
if __name__ == "__main__":
    # load dictionary
    model_wv = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')
    #    runInfo_dir= 'runInfo/human/'
    #    mkdir(runInfo_dir)
    plot_dir = 'plot/human/'

    sizes = [20]
    windows = [4]
    maxlens = [850]  # 550,650,750,850
    batch_sizes = [256]  # 32,64,128,256
    nb_epoches = [45]
for size in sizes:
    for window in windows:
        for maxlen in maxlens:
            for batch_size in batch_sizes:
                for nb_epoch in nb_epoches:
                    sequence_len = size * maxlen
                    # get training data
                    train_fea_protein_AB, train_label = get_training_dataset(model_wv.wv, maxlen, size)
                    # split in 9/10 1/10
                    train_fea_protein_AB, test_fea_protein_AB_split, train_label, test_label_split = sklearn.model_selection.train_test_split(
                        train_fea_protein_AB, train_label, test_size=0.1, random_state=42)
                    test_fea_protein_AB_p1, test_label_p1 = get_test_dataset(model_wv.wv, maxlen, size, 'partition_1')
                    test_fea_protein_AB_both, test_label_both = get_test_dataset(model_wv.wv, maxlen, size,
                                                                                 'both')

                    print('dataset is loaded')
                    swm = 'swissProt_size_' + str(size) + '_window_' + str(window) + '_maxlen_' + str(maxlen)

                    #                        #scaler
                    scaler = StandardScaler().fit(train_fea_protein_AB)
                    train_fea_protein_AB = scaler.transform(train_fea_protein_AB)
                    scaler = StandardScaler().fit(test_fea_protein_AB_split)
                    test_fea_protein_AB_split = scaler.transform(test_fea_protein_AB_split)
                    scaler = StandardScaler().fit(test_fea_protein_AB_p1)
                    test_fea_protein_AB_p1 = scaler.transform(test_fea_protein_AB_p1)
                    scaler = StandardScaler().fit(test_fea_protein_AB_both)
                    test_fea_protein_AB_both = scaler.transform(test_fea_protein_AB_both)
                    #
                    y_train = utils.to_categorical(train_label)
                    y_test_split = utils.to_categorical(test_label_split)
                    y_test_p1 = utils.to_categorical(test_label_p1)
                    y_test_both = utils.to_categorical(test_label_both)

                    # train
                    print("================")
                    X_train_left = train_fea_protein_AB[:, 0:sequence_len]
                    X_train_right = train_fea_protein_AB[:, sequence_len:sequence_len * 2]

                    X_test_left_split = np.array(test_fea_protein_AB_split[:, 0:sequence_len])
                    X_test_right_split = np.array(test_fea_protein_AB_split[:, sequence_len:sequence_len * 2])
                    X_test_left_p1 = np.array(test_fea_protein_AB_p1[:, 0:sequence_len])
                    X_test_right_p1 = np.array(test_fea_protein_AB_p1[:, sequence_len:sequence_len * 2])
                    X_test_left_both = np.array(test_fea_protein_AB_both[:, 0:sequence_len])
                    X_test_right_both = np.array(test_fea_protein_AB_both[:, sequence_len:sequence_len * 2])

                    # turn to np.array
                    X_train_left = np.array(X_train_left)
                    X_train_right = np.array(X_train_right)

                    model = merged_DBN_functional(sequence_len)
                    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)

                    model.compile(loss='categorical_crossentropy',
                                  optimizer=sgd,
                                  metrics=[tf.keras.metrics.Precision()])

                    hist = model.fit(
                        {'left': X_train_left, 'right': X_train_right},
                        {'ppi_pred': y_train},
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        verbose=1
                    )
                    print('******   model created!  ******')

                    print('****** Predictions 1/10 split partition 0 ******')
                    test_data(X_test_left_split, X_test_right_split, y_test_split)

                    print('****** Predictions partition 1 ******')
                    test_data(X_test_left_p1, X_test_right_p1, y_test_p1)

                    print('****** Predictions both ******')

                    test_data(X_test_left_both, X_test_right_both, y_test_both)
