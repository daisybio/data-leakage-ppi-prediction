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
from time import time

#tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.config.threading.set_inter_op_parallelism_threads(32)


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


def read_in_dataset(dataset, test, partition, seq_size, rewired=False):
    if dataset.startswith('gold_standard'):
        datasets = ['gold_standard_train', 'gold_standard_val', 'gold_standard_test']
    elif partition:
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
        if dataset in ['du_both_0', 'du_both_1', 'du_0_1', 'guo_both_0', 'guo_both_1', 'guo_0_1']:
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
    elif dataset.startswith('gold_standard'):
        id2index, seqs = read_in_seqdict('human')
        if dataset == 'gold_standard_train':
            train_file_pos = f'../../../../../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt'
            train_file_neg = f'../../../../../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt'
        elif dataset == 'gold_standard_val':
            train_file_pos = f'../../../../../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt'
            train_file_neg = f'../../../../../Datasets_PPIs/Hippiev2.3/Intra1_neg_rr.txt'
        else:
            train_file_pos = f'../../../../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt'
            train_file_neg = f'../../../../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt'
        ppis = read_ppis_from_sprint(train_file_pos, train_file_neg, id2index)
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
        if rewired:
            folder = 'rewired'
        else:
            folder = 'original'
        print(f'Getting {folder} ...')
        train_file_pos = f'../../../../SPRINT/data/{folder}/{dataset}_{prefix}_pos.txt'
        train_file_neg = f'../../../../SPRINT/data/{folder}/{dataset}_{prefix}_neg.txt'
        ppis = read_ppis_from_sprint(train_file_pos, train_file_neg, id2index)
    print('Embedding ...')
    X, class_labels = process_ppis(ppis, id2index, seqs, seq_size, dim, seq2t)
    return dim, X, class_labels


def read_ppis_from_sprint(pos_file, neg_file, id2index):
    ppis = []
    with open(pos_file, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if id2index.get(line_split[0]) is None or id2index.get(line_split[1]) is None:
                continue
            ppis.append([line_split[0], line_split[1], '1'])
    with open(neg_file, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if id2index.get(line_split[0]) is None or id2index.get(line_split[1]) is None:
                continue
            ppis.append([line_split[0], line_split[1], '0'])
    return ppis


def build_model(seq_size, dim):
    hidden_dim = 25
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


def calculate_performace(test_num, pred_y, true_y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if true_y[index][0] > 0.:
            if pred_y[index][0] > pred_y[index][1]:
                tp += + 1
            else:
                fn += 1
        else:
            if pred_y[index][0] > pred_y[index][1]:
                fp = fp + 1
            else:
                tn = tn + 1
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


if __name__ == '__main__':
    args = sys.argv[1:]
    print(f'########################### {args[0]} ###########################')
    if args[0] == 'original':
        partition = False
        rewired = False
        prefix = 'original_'
    elif args[0] == 'rewired':
        partition = False
        rewired = True
        prefix = 'rewired_'
    elif args[0] == 'partition':
        partition = True
        rewired = False
        prefix = 'partition_'
    else:
        partition = False
        rewired = False
        prefix = 'gold_standard_'
    seq_size = 2000
    n_epochs = 50
    batch_size = 256
    if prefix == 'gold_standard_':
        datasets = ['gold_standard']
    elif partition:
        datasets = ['guo_both_0', 'guo_both_1', 'guo_0_1',
                    'huang_both_0', 'huang_both_1', 'huang_0_1',
                    'du_both_0', 'du_both_1', 'du_0_1',
                    'pan_both_0', 'pan_both_1', 'pan_0_1',
                    'richoux_both_0', 'richoux_both_1', 'richoux_0_1']
    else:
        datasets = ['huang', 'guo', 'du', 'pan', 'richoux_regular', 'richoux_strict']
    for dataset in datasets:
        t_start = time()
        print(f'####################### {dataset} Dataset #######################')
        if dataset == 'gold_standard':
            print('Reading training data ...')
            dim, X_train, y_train = read_in_dataset(dataset='gold_standard_train', test=False, partition=partition, seq_size=seq_size,
                                                    rewired=rewired)
            print('Reading validation data ...')
            dim_val, X_val, y_val = read_in_dataset(dataset='gold_standard_val', test=False, partition=partition,
                                                    seq_size=seq_size,
                                                    rewired=rewired)
            print('Reading test data ...')
            dim_test, X_test, y_test = read_in_dataset(dataset='gold_standard_test', test=True, partition=partition,
                                                       seq_size=seq_size, rewired=rewired)
            print('###########################')
            print(f'Train: {int(len(y_train[:, 0]))} ({int(sum(y_train[:, 0]))}/{int(len(y_train[:, 0])) - int(sum(y_train[:, 0]))}),'
                  f'Validation: {int(len(y_val[:, 0]))} ({int(sum(y_val[:, 0]))}/{int(len(y_val[:, 0])) - int(sum(y_val[:, 0]))}),'
                  f'Test: {int(len(y_test[:, 0]))} ({int(sum(y_test[:, 0]))}/{int(len(y_test[:, 0])) - int(sum(y_test[:, 0]))}),')
        else:
            print('Reading training data ...')
            dim, X_train, y_train = read_in_dataset(dataset=dataset, test=False, partition=partition, seq_size=seq_size, rewired=rewired)
            print('Reading test data ...')
            dim_test, X_test, y_test = read_in_dataset(dataset=dataset, test=True, partition=partition, seq_size=seq_size, rewired=rewired)
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
        print(f'Dim train: {X_train[0].shape}')
        if dataset == 'gold_standard':
            print(f'Dim val: {X_val[0].shape}')
            hist = merge_model.fit(X_train, y_train,
                                   validation_data=(X_val, y_val),
                                   batch_size=batch_size,
                                   epochs=n_epochs)
        else:
            hist = merge_model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs)
        print('Predicting ...')
        y_pred = merge_model.predict(X_test)
        print('Exporting results ...')
        write_results(path=f'results/{prefix}{dataset}.csv', y_true=y_test, y_pred=y_pred)
        with open(f'results/all_times.txt', 'a+') as f:
            f.write(f'{prefix}{dataset}\t{time() - t_start}')

