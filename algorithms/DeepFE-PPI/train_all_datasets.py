import os
import keras
from time import time
from keras.layers import BatchNormalization, Dense, Dropout, concatenate
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


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


def protein_representation(wv, tokened_seq_protein, maxlen, size):
    represented_protein = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        if i % 1000 == 0:
            print(f'Processing PPI {i}/{len(tokened_seq_protein)}')
        for j in range(maxlen):
            if tokened_seq_protein[i][j] == 'J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))
    return represented_protein


def read_deepFE_files(file1, file2, file3, file4):
    pos_seq_protein_A = read_trainingData(file1)
    pos_seq_protein_B = read_trainingData(file2)
    neg_seq_protein_A = read_trainingData(file3)
    neg_seq_protein_B = read_trainingData(file4)
    return pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B


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
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)
    pos_neg_seq_protein_B = copy.deepcopy(pos_seq_protein_B)
    pos_neg_seq_protein_B.extend(neg_seq_protein_B)
    seq = []
    seq.extend(pos_neg_seq_protein_A)
    seq.extend(pos_neg_seq_protein_B)
    print(
        f'Read in dataset! {len(pos_neg_seq_protein_A)} PPIs, {len(pos_seq_protein_A)} positives, {len(neg_seq_protein_A)} negatives')
    max_min_avg_length(seq)

    # token
    print('Making token ...')
    token_pos_neg_seq_protein_A = token(pos_neg_seq_protein_A)
    token_pos_neg_seq_protein_B = token(pos_neg_seq_protein_B)
    # padding
    print('Padding ...')
    tokened_token_pos_neg_seq_protein_A = padding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = padding_J(token_pos_neg_seq_protein_B, maxlen)
    # protein reprsentation
    print('Representing proteins ...')
    feature_protein_A = protein_representation(wv, tokened_token_pos_neg_seq_protein_A, maxlen, size)
    feature_protein_B = protein_representation(wv, tokened_token_pos_neg_seq_protein_B, maxlen, size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A), np.array(feature_protein_B)))
    #  create label
    label = np.ones(len(feature_protein_A))
    label[len(pos_seq_protein_A):] = 0

    return feature_protein_AB, label


def get_training_dataset(wv, maxlen, size, dataset):
    datasets = ['du', 'guo', 'huang', 'pan', 'richoux_regular', 'richoux_strict']
    if dataset not in datasets:
        raise ValueError(f'Dataset must be in {datasets}!')
    if dataset == 'guo':
        file_1 = 'dataset/training_and_test_dataset/positive/Protein_A.txt'
        file_2 = 'dataset/training_and_test_dataset/positive/Protein_B.txt'
        file_3 = 'dataset/training_and_test_dataset/negative/Protein_A.txt'
        file_4 = 'dataset/training_and_test_dataset/negative/Protein_B.txt'
        pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = read_deepFE_files(file_1, file_2,
                                                                                                       file_3, file_4)
    elif dataset == 'huang':
        file_1 = 'dataset/human/positive/Protein_A.txt'
        file_2 = 'dataset/human/positive/Protein_B.txt'
        file_3 = 'dataset/human/negative/Protein_A.txt'
        file_4 = 'dataset/human/negative/Protein_B.txt'
        pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = read_deepFE_files(file_1, file_2,
                                                                                                       file_3, file_4)
    elif dataset == 'du':
        pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = convert_du_to_deepFE()
    elif dataset == 'pan':
        pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = convert_pan_to_deepFE()
    elif dataset == 'richoux_regular':
        pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = convert_richoux_training_to_deepFE(
            regular=True)
    else:
        pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B = convert_richoux_training_to_deepFE(
            regular=False)

    feature_protein_AB, label = process_sequence_pairs(wv, maxlen, size, pos_seq_protein_A, neg_seq_protein_A,
                                                       pos_seq_protein_B, neg_seq_protein_B)
    return feature_protein_AB, label


def get_test_richoux(wv, maxlen, size, dataset):
    if dataset == 'richoux_regular':
        # 12,806
        path_to_test = '../DeepPPI/data/mirror/medium_1166_test_mirror.txt'
    else:
        # 720
        path_to_test = '../DeepPPI/data/mirror/double/test_double_mirror.txt'
    pos_A_test, pos_B_test, neg_A_test, neg_B_test = read_richoux_file(path_to_test)
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


def convert_du_to_deepFE():
    prefix_dict, seq_dict = make_swissprot_to_dict(organism='yeast')
    pos_seq_protein_A = []
    pos_seq_protein_B = []
    neg_seq_protein_A = []
    neg_seq_protein_B = []
    with open('../../Datasets_PPIs/Du_yeast_DIP/SupplementaryS1.csv', 'r') as f:
        for line in f:
            ppi = line.strip().split(',')
            if ppi[0] in seq_dict.keys() and ppi[1] in seq_dict.keys():
                seq1 = seq_dict[ppi[0]]
                seq2 = seq_dict[ppi[1]]
                label = ppi[2]
                if label == '1':
                    pos_seq_protein_A.append(seq1)
                    pos_seq_protein_B.append(seq2)
                else:
                    neg_seq_protein_A.append(seq1)
                    neg_seq_protein_B.append(seq2)
    return pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B


def make_pan_seq_dict():
    seq_dict = {}
    with open('../../Datasets_PPIs/Pan_human_HPRD/SEQ-Supp-ABCD.tsv', 'r') as f:
        for line in f:
            line_split = line.strip().split('\t')
            seq_dict[line_split[0]] = line_split[1]
    return seq_dict


def convert_pan_to_deepFE():
    pan_seq_dict = make_pan_seq_dict()
    pos_seq_protein_A = []
    pos_seq_protein_B = []
    neg_seq_protein_A = []
    neg_seq_protein_B = []
    with open('../../Datasets_PPIs/Pan_human_HPRD/Supp-AB.tsv', 'r') as f:
        for line in f:
            if line.startswith('v1'):
                # header
                continue
            else:
                line_split_pan = line.strip().split('\t')
                id1 = line_split_pan[0]
                id2 = line_split_pan[1]
                seq1 = pan_seq_dict[id1]
                seq2 = pan_seq_dict[id2]
                label = line_split_pan[2]
                if label == '1':
                    pos_seq_protein_A.append(seq1)
                    pos_seq_protein_B.append(seq2)
                else:
                    neg_seq_protein_A.append(seq1)
                    neg_seq_protein_B.append(seq2)
    return pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B


def read_richoux_file(path):
    pos_seq_protein_A = []
    pos_seq_protein_B = []
    neg_seq_protein_A = []
    neg_seq_protein_B = []
    with open(path, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                continue
            else:
                seq1 = line_split[2]
                seq2 = line_split[3]
                label = line_split[4]
                if label == '1':
                    pos_seq_protein_A.append(seq1)
                    pos_seq_protein_B.append(seq2)
                else:
                    neg_seq_protein_A.append(seq1)
                    neg_seq_protein_B.append(seq2)
    return pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B


def convert_richoux_training_to_deepFE(regular=True):
    if regular is True:
        # 85,104
        path_to_train = '../DeepPPI/data/mirror/medium_1166_train_mirror.txt'
        # 12,822
        path_to_val = '../DeepPPI/data/mirror/medium_1166_val_mirror.txt'
    else:
        # 91,036
        path_to_train = '../DeepPPI/data/mirror/double/double-medium_1166_train_mirror.txt'
        # 12,506
        path_to_val = '../DeepPPI/data/mirror/double/double-medium_1166_val_mirror.txt'
    pos_seq_protein_A = []
    pos_seq_protein_B = []
    neg_seq_protein_A = []
    neg_seq_protein_B = []
    pos_A_train, pos_B_train, neg_A_train, neg_B_train = read_richoux_file(path_to_train)
    pos_seq_protein_A.extend(pos_A_train)
    pos_seq_protein_B.extend(pos_B_train)
    neg_seq_protein_A.extend(neg_A_train)
    neg_seq_protein_B.extend(neg_B_train)
    pos_A_val, pos_B_val, neg_A_val, neg_B_val = read_richoux_file(path_to_val)
    pos_seq_protein_A.extend(pos_A_val)
    pos_seq_protein_B.extend(pos_B_val)
    neg_seq_protein_A.extend(neg_A_val)
    neg_seq_protein_B.extend(neg_B_val)
    return pos_seq_protein_A, pos_seq_protein_B, neg_seq_protein_A, neg_seq_protein_B


# %%
if __name__ == "__main__":
    dataset = 'du'
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
    X, y = get_training_dataset(model_wv.wv, maxlen, size, dataset=dataset)
    y = utils.to_categorical(y)
    print('dataset is loaded')

    # scaler
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    if dataset not in ['richoux_regular', 'richoux_strict']:
        print('Splitting dataset in train/test')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('###########################')
        print(
            f'The {dataset} dataset contains {int(len(y[:, 0]))} samples ({int(sum(y[:, 0]))} positives, {int(len(y[:, 0])) - int(sum(y[:, 0]))} negatives).\n'
            f'80/20 training/test split results in train: {int(len(y_train[:, 0]))} ({int(sum(y_train[:, 0]))}/{int(len(y_train[:, 0])) - int(sum(y_train[:, 0]))}),'
            f' test: {int(len(y_test[:, 0]))} ({int(sum(y_test[:, 0]))}/{int(len(y_test[:, 0])) - int(sum(y_test[:, 0]))})')
        print('###########################')
    else:
        X_train = X
        y_train = y
        X_test, y_test = get_test_richoux(model_wv.wv, maxlen, size, dataset)
        print('###########################')
        print(
            f'The {dataset} dataset contains {int(len(y_train[:, 0]) + len(y_test[:, 0]))} samples ({int(sum(y_train[:, 0]) + sum(y_test[:, 0]))} positives, {int(len(y_train[:, 0]) + len(y_test[:, 0]) - sum(y_train[:, 0]) - sum(y_test[:, 0]))} negatives).\n'
            f'training/test split results in train: {int(len(y_train[:, 0]))} ({int(sum(y_train[:, 0]))}/{int(len(y_train[:, 0])) - int(sum(y_train[:, 0]))}),'
            f' test: {int(len(y_test[:, 0]))} ({int(sum(y_test[:, 0]))}/{int(len(y_test[:, 0])) - int(sum(y_test[:, 0]))})')
        print('###########################')

    result_dir = f'result/custom/{dataset}/'
    mkdir(result_dir)
    plot_dir = f'plot/custom/{dataset}/'
    mkdir(plot_dir)

    X_train_left = X_train[:, 0:sequence_len]
    X_train_right = X_train[:, sequence_len:sequence_len * 2]

    X_test_left = X_test[:, 0:sequence_len]
    X_test_right = X_test[:, sequence_len:sequence_len * 2]

    # turn to np.array
    X_train_left = np.array(X_train_left)
    X_train_right = np.array(X_train_right)

    X_test_left = np.array(X_test_left)
    X_test_right = np.array(X_test_right)

    model = merged_DBN_functional(sequence_len)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=[tf.keras.metrics.Precision()])
    # feed data into model
    hist = model.fit(
        {'left': X_train_left, 'right': X_train_right},
        {'ppi_pred': y_train},
        epochs=nb_epoch,
        batch_size=batch_size,
        verbose=1
    )

    print('******   model created!  ******')
    training_vis(hist, plot_dir, f'training_vis_{dataset}')
    predictions_test = model.predict([X_test_left, X_test_right])

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
    sc.to_csv(result_dir + f'scores_{dataset}.csv')
    print(f'time elapsed: {time() - t_start}')
