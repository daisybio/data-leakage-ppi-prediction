import sys, os
from sklearn.metrics import roc_auc_score, average_precision_score
import keras
from time import time
from keras.layers import BatchNormalization, Dense, Dropout, concatenate
import numpy as np
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras import backend as K
from tensorflow.keras import callbacks
import random


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def max_min_avg_length(seq):
    length = []
    for string in seq:
        length.append(len(string))
    maxNum = max(length)  # maxNum = 5
    minNum = min(length)  # minNum = 1

    avg = averagenum(length)

    print('Longest protein length: ' + str(maxNum))
    print('Shortest protein length: ' + str(minNum))
    print('Average protein length: ' + str(avg))


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
def training_vis(hist, plot_dir, filename):
    loss = hist.history['loss']
    acc = hist.history['precision']

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
    ax2.plot(acc, label='train_precision')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision on Training Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + filename)
    plt.show()


def read_file(file_name):
    pro_swissProt = []
    with open(file_name, 'r') as fp:
        protein = ''
        for line in fp:
            if line.startswith('>sp|'):
                pro_swissProt.append(protein)
                protein = ''
            elif line.startswith('>sp|') == False:
                protein = protein + line.strip()

    return pro_swissProt[1:]


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


def protein_representation(wv, pos_seq_protein_A, pos_seq_protein_B, maxlen, size, feature_protein_AB):
    for i in range(len(pos_seq_protein_A)):
        if i % 5000 == 0:
            print(f'Processing PPI {i}/{len(pos_seq_protein_A)}')
        for j in range(maxlen):
            if pos_seq_protein_A[i][j] == 'J':
                feature_protein_AB[i][j*size:j*size+size] = np.zeros(size)
            else:
                feature_protein_AB[i][j*size:j*size+size] = wv[pos_seq_protein_A[i][j]]
            if pos_seq_protein_B[i][j] == 'J':
                feature_protein_AB[i][j*size+maxlen*size:j*size+size+maxlen*size] = np.zeros(size)
            else:
                feature_protein_AB[i][j*size+maxlen*size:j*size+size+maxlen*size] = wv[pos_seq_protein_B[i][j]]
    return feature_protein_AB


def read_trainingData(file_name):
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i % 2 == 1:
                seq.append(line.split('\n')[0])
            i = i + 1
    return seq


def process_sequence_pairs(wv, maxlen, size, pos_seq_protein_A, neg_seq_protein_A, pos_seq_protein_B,
                           neg_seq_protein_B):
    # remember lengths
    len_pos = len(pos_seq_protein_A)
    n_ppis = len(pos_seq_protein_A) + len(neg_seq_protein_A)
    # edit: trying to save space, save pos and neg in pos_seq
    pos_seq_protein_A.extend(neg_seq_protein_A)
    pos_seq_protein_B.extend(neg_seq_protein_B)
    print(
        f'Read in dataset! {n_ppis} PPIs, {len_pos} positives, {n_ppis-len_pos} negatives')

    # token
    print('Making token ...')
    pos_seq_protein_A = token(pos_seq_protein_A)
    pos_seq_protein_B = token(pos_seq_protein_B)
    # padding
    print('Padding ...')
    pos_seq_protein_A = padding_J(pos_seq_protein_A, maxlen)
    pos_seq_protein_B = padding_J(pos_seq_protein_B, maxlen)
    print('Representing proteins ...')
    feature_protein_AB = np.zeros(shape=(n_ppis, 2 * maxlen * size))
    feature_protein_AB = protein_representation(wv, pos_seq_protein_A, pos_seq_protein_B, maxlen, size, feature_protein_AB)
    #  create label
    print("creating labels ...")
    label = np.ones(n_ppis)
    label[len_pos:] = 0

    return feature_protein_AB, label


def get_training_dataset(wv, maxlen, size, dataset, partition, rewired, seed=None):
    if partition:
        datasets=['dscript_both_0','dscript_both_1','dscript_0_1',
                  'guo_both_0','guo_both_1','guo_0_1',
                'huang_both_0', 'huang_both_1', 'huang_0_1',
                'du_both_0', 'du_both_1', 'du_0_1',
                'pan_both_0', 'pan_both_1', 'pan_0_1',
                'richoux_both_0', 'richoux_both_1', 'richoux_0_1']
    elif dataset == 'gold_standard':
        datasets = ['gold_standard']
    elif dataset == 'gold_standard_unbalanced':
        datasets = ['gold_standard_unbalanced']
    else:
        datasets = ['dscript', 'du', 'guo', 'huang', 'pan', 'richoux_regular', 'richoux_strict']
    if dataset not in datasets:
        raise ValueError(f'Dataset must be in {datasets}!')
    if partition:
        ds_split = dataset.split('_')
        name = ds_split[0]
        if name in ['du', 'guo']:
            organism='yeast'
        else:
            organism='human'
        train_partition = ds_split[1]
        train_file_pos = f'../SPRINT/data/partitions/{name}_partition_{train_partition}_pos.txt'
        train_file_neg = f'../SPRINT/data/partitions/{name}_partition_{train_partition}_neg.txt'
    elif dataset == 'gold_standard':
        print('Getting gold standard dataset ...')
        train_file_pos = '../../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt'
        train_file_neg = '../../Datasets_PPIs/Hippiev2.3/Intra1_neg_rr.txt'
        organism = 'human'
    elif dataset == 'gold_standard_unbalanced':
        print('Getting gold standard unbalanced dataset ...')
        train_file_pos = '../../Datasets_PPIs/unbalanced_gold/Intra1_pos.txt'
        train_file_neg = '../../Datasets_PPIs/unbalanced_gold/Intra1_neg.txt'
        organism = 'human'
    else:
        if rewired:
            folder = 'rewired'
        else:
            folder = 'original'
        if dataset in ['du', 'guo']:
            organism='yeast'
        else:
            organism='human'
        print(f'Getting {folder} dataset ...')
        if seed is None:
            train_file_pos = f'../SPRINT/data/{folder}/{dataset}_train_pos.txt'
            train_file_neg = f'../SPRINT/data/{folder}/{dataset}_train_neg.txt'
        else:
            train_file_pos = f'../SPRINT/data/{folder}/multiple_random_splits/{dataset}_train_pos_{seed}.txt'
            train_file_neg = f'../SPRINT/data/{folder}/multiple_random_splits/{dataset}_train_neg_{seed}.txt'
    pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = read_sprint_files(train_file_pos,
                                                                                                   train_file_neg,
                                                                                                   organism)

    feature_protein_AB, label = process_sequence_pairs(wv, maxlen, size, pos_seq_protein_A, neg_seq_protein_A,
                                                       pos_seq_protein_B, neg_seq_protein_B)
    return feature_protein_AB, label


def get_test_partition(wv, maxlen, size, dataset):
    ds_split = dataset.split('_')
    name = ds_split[0]
    if name in ['du', 'guo']:
        organism = 'yeast'
    else:
        organism = 'human'
    test_partition = ds_split[2]
    test_file_pos = f'../SPRINT/data/partitions/{name}_partition_{test_partition}_pos.txt'
    test_file_neg = f'../SPRINT/data/partitions/{name}_partition_{test_partition}_neg.txt'
    pos_A_test, pos_B_test, neg_A_test, neg_B_test = read_sprint_files(test_file_pos,
                                                                                                      test_file_neg,
                                                                                                      organism)
    feature_protein_AB, label = process_sequence_pairs(wv, maxlen, size, pos_A_test, neg_A_test,
                                                       pos_B_test, neg_B_test)
    return feature_protein_AB, label


def get_test_set(wv, maxlen, size, dataset, rewired, seed=None):
    if dataset.startswith('gold_standard'):
        organism = 'human'
        if dataset == 'gold_standard_val':
            test_file_pos = '../../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt'
            test_file_neg = '../../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt'
        elif dataset == 'gold_standard_unbalanced_val':
            test_file_pos = '../../Datasets_PPIs/unbalanced_gold/Intra0_pos.txt'
            test_file_neg = '../../Datasets_PPIs/unbalanced_gold/Intra0_neg.txt'
        elif dataset == 'gold_standard_unbalanced_test':
            test_file_pos = '../../Datasets_PPIs/unbalanced_gold/Intra2_pos.txt'
            test_file_neg = '../../Datasets_PPIs/unbalanced_gold/Intra2_neg.txt'
        else:
            test_file_pos = '../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt'
            test_file_neg = '../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt'
    else:
        if rewired:
            folder = 'rewired'
        else:
            folder = 'original'
        if dataset in ['du', 'guo']:
            organism = 'yeast'
        else:
            organism = 'human'
        if seed is None:
            test_file_pos = f'../SPRINT/data/{folder}/{dataset}_test_pos.txt'
            test_file_neg = f'../SPRINT/data/{folder}/{dataset}_test_neg.txt'
        else:
            test_file_pos = f'../SPRINT/data/{folder}/multiple_random_splits/{dataset}_test_pos_{seed}.txt'
            test_file_neg = f'../SPRINT/data/{folder}/multiple_random_splits/{dataset}_test_neg_{seed}.txt'
    pos_A_test, pos_B_test, neg_A_test, neg_B_test = read_sprint_files(test_file_pos,
                                                                       test_file_neg,
                                                                       organism)
    feature_protein_AB, label = process_sequence_pairs(wv, maxlen, size, pos_A_test, neg_A_test,
                                                       pos_B_test, neg_B_test)
    return feature_protein_AB, label


def make_swissprot_to_dict(organism='yeast'):
    prefix_dict = {}
    seq_dict = {}
    header_line = False
    last_id = ''
    last_seq = ''
    n = 30
    if organism == 'yeast':
        f = open('../../Datasets_PPIs/SwissProt/yeast_swissprot.fasta', 'r')
    else:
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


def convert_edgelist_to_seqlist(file, seq_dict):
    seq_A = []
    seq_B = []
    with open(file, 'r') as f:
        for line in f:
            ppi = line.strip().split(' ')
            if ppi[0] in seq_dict.keys() and ppi[1] in seq_dict.keys():
                seq_A.append(seq_dict[ppi[0]])
                seq_B.append(seq_dict[ppi[1]])
    return seq_A, seq_B


def read_sprint_files(pos_file, neg_file, organism):
    prefix_dict, seq_dict = make_swissprot_to_dict(organism=organism)
    pos_seq_protein_A, pos_seq_protein_B = convert_edgelist_to_seqlist(pos_file, seq_dict)
    neg_seq_protein_A, neg_seq_protein_B = convert_edgelist_to_seqlist(neg_file, seq_dict)
    return pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B


# %%
if __name__ == "__main__":
    args = sys.argv[1:]
    print(f'########################### {args[0]} ###########################')
    if args[0] == 'original':
        partition = False
        rewired = False
        original = True
        prefix = 'original_'
    elif args[0] == 'rewired':
        partition = False
        rewired = True
        original = False
        prefix = 'rewired_'
    elif args[0] == 'partition':
        partition = True
        rewired = False
        original = False
        prefix = 'partition_'
    elif args[0] == 'gold':
        partition = False
        rewired = False
        original = False
        prefix = 'gold_standard_'
    else:
        partition = False
        rewired = False
        original = False
        prefix = 'gold_standard_unbalanced_'
    if len(args) > 1 and args[1] == 'split_train':
        split_train = True
        datasets = None
        seed = None
    elif len(args) > 1:
        split_train = False
        datasets = [arg for arg in args[1].split(',')]
        print(f'Using dataset list {datasets}')
        if len(args) > 2:
            seed = int(args[2])
            print(f'Using seed {seed}')
        else:
            seed = None
    else:
        split_train = False
        seed = None
        datasets = None

    if (rewired or original) and datasets is None:
        datasets = ['huang', 'guo', 'du', 'pan', 'richoux_strict', 'richoux_regular', 'dscript']
    elif partition:
        datasets = ['huang_both_0', 'huang_both_1', 'huang_0_1',
                    'guo_both_0','guo_both_1','guo_0_1',
                    'du_both_0', 'du_both_1', 'du_0_1',
                    'pan_both_0', 'pan_both_1', 'pan_0_1',
                    'richoux_both_0', 'richoux_both_1', 'richoux_0_1',
                    'dscript_both_0', 'dscript_both_1', 'dscript_0_1']
    elif args[0] == 'gold':
        datasets = ['gold_standard']
    else:
        datasets = datasets

    for dataset in datasets:
        print(f'Dataset: {dataset}')
        # load dictionary
        model_wv = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')

        size = 20
        window = 4
        maxlen = 850
        batch_size = 256
        nb_epoch = 45
        sequence_len = size * maxlen

        # get training data
        t_start = time()
        X_train, y_train = get_training_dataset(model_wv.wv, maxlen, size, dataset=dataset, partition=partition, rewired=rewired, seed=seed)
        y_train = utils.to_categorical(y_train)
        print('dataset is loaded')
        print(f'Train: {int(len(y_train[:, 1]))} ({int(sum(y_train[:, 1]))}/{int(len(y_train[:, 1])) - int(sum(y_train[:, 1]))})')
        # scaler
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        if dataset == 'gold_standard' or dataset == 'gold_standard_unbalanced':
            if dataset == 'gold_standard':
                X_val, y_val = get_test_set(model_wv.wv, maxlen, size, 'gold_standard_val', rewired)
                dataset = 'gold_standard_test'
            else:
                X_val, y_val = get_test_set(model_wv.wv, maxlen, size, 'gold_standard_unbalanced_val', rewired)
                dataset = 'gold_standard_unbalanced_test'
            y_val = utils.to_categorical(y_val)
            X_val = scaler.transform(X_val)
            print(f'Val: {int(len(y_val[:, 1]))} ({int(sum(y_val[:, 1]))}/{int(len(y_val[:, 1])) - int(sum(y_val[:, 1]))})')

        print('###########################')
        if partition:
            result_dir = f'result/custom/{dataset.split("_")[0]}/'
            mkdir(result_dir)
            plot_dir = f'plot/custom/{dataset.split("_")[0]}/'
            mkdir(plot_dir)
        elif seed is not None:
            result_dir = f'result/multiple_runs/'
            mkdir(result_dir)
            plot_dir = f'plot/multiple_runs/'
            mkdir(plot_dir)
        else:
            result_dir = f'result/custom/{dataset}/'
            mkdir(result_dir)
            plot_dir = f'plot/custom/{dataset}/'
            mkdir(plot_dir)


        model = merged_DBN_functional(sequence_len)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=[tf.keras.metrics.Precision()])

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_precision', patience=5, verbose=1),
            callbacks.ModelCheckpoint(filepath=f'best_models/{prefix}{dataset}_es.h5',
                                      monitor='val_precision', save_best_only=True, verbose=1)
        ]

        if dataset.startswith('gold_standard'):
            # feed data into model
            if split_train:
                hist = model.fit(
                    {'left': np.array(X_train[:, 0:sequence_len]),
                     'right': np.array(X_train[:, sequence_len:sequence_len * 2])},
                    {'ppi_pred': y_train},
                    validation_data=[{'left': np.array(X_val[:, 0:sequence_len]),
                                      'right': np.array(X_val[:, sequence_len:sequence_len * 2])},
                                     {'ppi_pred': y_val}],
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=callbacks_list
                )
            else:
                hist = model.fit(
                    {'left': np.array(X_train[:, 0:sequence_len]),
                     'right': np.array(X_train[:, sequence_len:sequence_len * 2])},
                    {'ppi_pred': y_train},
                    validation_data=[{'left': np.array(X_val[:, 0:sequence_len]),
                     'right': np.array(X_val[:, sequence_len:sequence_len * 2])},
                    {'ppi_pred': y_val}],
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    verbose=1
                )
        elif split_train:
            hist = model.fit(
                {'left': np.array(X_train[:, 0:sequence_len]),
                 'right': np.array(X_train[:, sequence_len:sequence_len * 2])},
                {'ppi_pred': y_train},
                validation_split=0.1,
                epochs=nb_epoch,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks_list
            )
        else:
            # feed data into model
            hist = model.fit(
                {'left': np.array(X_train[:, 0:sequence_len]),
                 'right': np.array(X_train[:, sequence_len:sequence_len * 2])},
                {'ppi_pred': y_train},
                epochs=nb_epoch,
                batch_size=batch_size,
                verbose=1
            )

        print('******   model created!  ******')
        if seed is None:
            training_vis(hist, plot_dir, f'training_vis_{dataset}')
        else:
            training_vis(hist, plot_dir, f'training_vis_{dataset}_{seed}')

        if not partition:
            X_test, y_test = get_test_set(model_wv.wv, maxlen, size, dataset, rewired, seed)
            y_test = utils.to_categorical(y_test)
            X_test = scaler.transform(X_test)
        else:
            X_test, y_test = get_test_partition(model_wv.wv, maxlen, size, dataset)
            y_test = utils.to_categorical(y_test)
            X_test = scaler.transform(X_test)
        print(f'Test: {int(len(y_test[:, 1]))} ({int(sum(y_test[:, 1]))}/{int(len(y_test[:, 1])) - int(sum(y_test[:, 1]))})')

        if split_train:
            print('Evaluating on the best model')
            model = tf.keras.models.load_model(f'best_models/{prefix}{dataset}_es.h5')
            dataset = f'{dataset}_es'
        predictions_test = model.predict([np.array(X_test[:, 0:sequence_len]),
                                          np.array(X_test[:, sequence_len:sequence_len * 2])])

        auc_test = roc_auc_score(y_test[:, 1], predictions_test[:, 1])
        pr_test = average_precision_score(y_test[:, 1], predictions_test[:, 1])

        label_predict_test = utils.categorical_probas_to_classes(predictions_test)
        tp_test, fp_test, tn_test, fn_test, accuracy_test, precision_test, sensitivity_test, recall_test, specificity_test, MCC_test, f1_score_test, _, _, _ = utils.calculate_performace(
            len(label_predict_test), label_predict_test, y_test[:, 1])
        print(' ===========  test ===========')

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
        if seed is None:
            sc.to_csv(result_dir + f'{prefix}scores_{dataset}.csv')
        else:
            sc.to_csv(result_dir + f'{prefix}scores_{dataset}_{seed}.csv')
        time_elapsed = time() - t_start
        print(f'time elapsed: {time_elapsed}')
        if seed is None:
            with open(result_dir + f'time_{prefix}{dataset}.txt', 'w') as f:
                f.write(str(time_elapsed))
        else:
            with open(result_dir + f'time_{prefix}{dataset}_{seed}.txt', 'w') as f:
                f.write(str(time_elapsed))
        K.clear_session()
