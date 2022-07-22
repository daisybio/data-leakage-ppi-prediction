from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../../../embeddings' not in sys.path:
    sys.path.append('../../../embeddings')

from seq2tensor import s2t
from keras import Model
from keras.layers import Input, GRU, Bidirectional, MaxPool1D, GlobalAveragePooling1D, Dense, LeakyReLU, Conv1D, concatenate, multiply
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,  RMSprop
import numpy as np


def process_ppis(ppis, id2index, seqs, seq_size, dim, seq2t):
    n_ppi = len(ppis)
    X = [np.zeros(shape=(n_ppi, seq_size, dim)), np.zeros(shape=(n_ppi, seq_size, dim))]
    class_labels = np.zeros(shape=(n_ppi, 2))
    nr_ppi = 0
    for ppi in ppis:
        idx0 = id2index.get(ppi[0])
        idx1 = id2index.get(ppi[1])
        embedded_0 = seq2t.embed_normalized(seqs[idx0], seq_size)
        embedded_1 = seq2t.embed_normalized(seqs[idx1], seq_size)
        X[0][nr_ppi] = embedded_0
        X[1][nr_ppi] = embedded_1
        if ppi[2] == '0':
            class_labels[nr_ppi][1] = 1
        else:
            class_labels[nr_ppi][0] = 1
        nr_ppi += 1
    return X, class_labels


def read_in_dataset(dataset, test, partition, seq_size):
    if partition:
        datasets = ['guo_both_0', 'guo_both_1', 'guo_0_1',
                    'huang_both_0', 'huang_both_1', 'huang_0_1',
                    'du_both_0', 'du_both_1', 'du_0_1',
                    'pan_both_0', 'pan_both_1', 'pan_0_1',
                    'richoux_both_0', 'richoux_both_1', 'richoux_0_1']
    else:
        datasets = ['du', 'guo', 'huang', 'pan', 'richoux_regular', 'richoux_strict']
    if dataset not in datasets:
        raise ValueError(f'Dataset must be in {datasets}!')
    seq2t = s2t('../../../embeddings/vec5_CTC.txt')
    dim = seq2t.dim
    if partition:
        ds_split = dataset.split('_')
        name = ds_split[0]
        if name in ['du_both_0', 'du_both_1', 'du_0_1', 'guo_both_0', 'guo_both_1', 'guo_0_1']:
            organism='yeast'
        else:
            organism='human'
        id2index, seqs = read_in_seqdict(organism)
        if not test:
            partition = ds_split[1]
        else:
            partition = ds_split[2]
        file_pos = f'../../../../SPRINT/data/partitions/{name}_partition_{partition}_pos.txt'
        file_neg = f'../../../../SPRINT/data/partitions/{name}_partition_{partition}_neg.txt'
        ppis = read_ppis_from_sprint(file_pos, file_neg, id2index)
    else:
        if dataset in ['guo', 'du']:
            organism='yeast'
        else:
            organism='human'
        id2index, seqs = read_in_seqdict(organism)
        if not test:
            prefix='train'
        else:
            prefix='test'
        train_file_pos = f'../../../../SPRINT/data/original/{dataset}_{prefix}_pos.txt'
        train_file_neg = f'../../../../SPRINT/data/original/{dataset}_{prefix}_neg.txt'
        ppis = read_ppis_from_sprint(train_file_pos, train_file_neg, id2index)
    print('Embedding ...')
    X, class_labels = process_ppis(ppis, id2index, seqs, seq_size, dim, seq2t)
    return dim, X, class_labels


def read_ppis_from_sprint(pos_file, neg_file, id2index):
    ppis = []
    pos_count = 0
    with open(pos_file, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if id2index.get(line_split[0]) is None or id2index.get(line_split[1]) is None:
                continue
            ppis.append([line_split[0], line_split[1], '1'])
            pos_count += 1
    neg_count = 0
    with open(neg_file, 'r') as f:
        for line in f:
            if neg_count == pos_count:
                break
            line_split = line.strip().split(' ')
            if id2index.get(line_split[0]) is None or id2index.get(line_split[1]) is None:
                continue
            ppis.append([line_split[0], line_split[1], '0'])
            neg_count += 1
    return ppis


def build_model(seq_size, dim):
    hidden_dim = 50
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(GRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(GRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(GRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(GRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(GRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPool1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPool1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPool1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPool1D(3)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPool1D(3)(l5(s1))
    s1=concatenate([r5(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPool1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPool1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPool1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPool1D(3)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPool1D(3)(l5(s2))
    s2=concatenate([r5(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model


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
    auc_test = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    pr_test = average_precision_score(y_true[:, 1], y_pred[:, 1])
    tp_test, fp_test, tn_test, fn_test, accuracy_test, precision_test, sensitivity_test, recall_test, specificity_test, MCC_test, f1_score_test = calculate_performace(
        len(y_pred), y_pred[:, 1], y_true[:, 1])

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
    sc.to_csv(path)


def read_in_seqdict(organism):
    if organism == 'yeast':
        path = '../../../../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta'
    else:
        path = '../../../../../Datasets_PPIs/SwissProt/human_swissprot_oneliner.fasta'
    id2index = {}
    seqs = []
    index = 0
    line_count = 0
    last_id = ''
    for line in open(path, 'r'):
        if line_count % 2 == 0:
            #id line
            last_id = line.strip().split('>')[1]
        else:
            seq = line.strip()
            id2index[last_id] = index
            seqs.append(seq)
            index += 1
        line_count += 1
    return id2index, seqs


def training_vis(hist, path):
    import matplotlib.pyplot as plt
    #from deepfe-ppi
    loss = hist.history['loss']
    acc = hist.history['accuracy']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy on Training Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    partition=False
    seq_size = 2000
    n_epochs = 100
    batch_size = 256
    for dataset in ['guo', 'huang', 'du', 'pan', 'richoux_regular', 'richoux_strict']:
        print(f'####################### {dataset} Dataset #######################')
        print('Reading training data ...')
        dim, X_train, y_train = read_in_dataset(dataset=dataset, test=False, partition=partition, seq_size=seq_size)
        print('Reading test data ...')
        dim_test, X_test, y_test = read_in_dataset(dataset=dataset, test=True, partition=partition, seq_size=seq_size)
        print('###########################')
        print(
            f'The {dataset} dataset contains {int(len(y_train[:, 0]) + len(y_test[:, 0]))} samples ({int(sum(y_train[:, 0]) + sum(y_test[:, 0]))} positives, {int(len(y_train[:, 0]) + len(y_test[:, 0]) - sum(y_train[:, 0]) - sum(y_test[:, 0]))} negatives).\n'
            f'training/test split results in train: {int(len(y_train[:, 0]))} ({int(sum(y_train[:, 0]))}/{int(len(y_train[:, 0])) - int(sum(y_train[:, 0]))}),'
            f' test: {int(len(y_test[:, 0]))} ({int(sum(y_test[:, 0]))}/{int(len(y_test[:, 0])) - int(sum(y_test[:, 0]))})')
        print('###########################')
        print('Building model ...')
        merge_model = build_model(seq_size, dim)
        adam = Adam(learning_rate=0.001, amsgrad=True, epsilon=1e-6)
        merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        hist = merge_model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs)
        training_vis(hist, f'results/training_vis_{dataset}')
        print('Predicting ...')
        y_pred = merge_model.predict(X_test)
        print('Exporting results ...')
        write_results(path=f'results/{dataset}.csv', y_true=y_test, y_pred=y_pred)

