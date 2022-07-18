from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../../../embeddings' not in sys.path:
    sys.path.append('../../../embeddings')

from seq2tensor import s2t
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GRU, Bidirectional, MaxPool1D, GlobalAveragePooling1D, Dense, LeakyReLU, Conv1D, concatenate, multiply
from tensorflow.keras.optimizers import Adam,  RMSprop
import os
import numpy as np
from tqdm import tqdm


def read_ppis(path, id2index_dict, seqs_list, seq_size):
    emb_files = ['../../../embeddings/default_onehot.txt', '../../../embeddings/string_vec5.txt',
                 '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt']
    use_emb = 3
    seq_array = []
    id2_aid = {}
    sid = 0
    label_index = -1
    sid1_index = 0
    sid2_index = 1

    seq2t = s2t(emb_files[use_emb])

    max_data = -1
    limit_data = max_data > 0
    raw_data = []
    skip_head = False
    count = 0

    for line in tqdm(open(path)):
        if skip_head:
            skip_head = False
            continue
        line = line.rstrip('\n').rstrip('\r').split('\t')
        if id2index_dict.get(line[sid1_index]) is None or id2index_dict.get(line[sid2_index]) is None:
            continue
        if id2_aid.get(line[sid1_index]) is None:
            id2_aid[line[sid1_index]] = sid
            sid += 1
            seq_array.append(seqs_list[id2index_dict[line[sid1_index]]])
        line[sid1_index] = id2_aid[line[sid1_index]]
        if id2_aid.get(line[sid2_index]) is None:
            id2_aid[line[sid2_index]] = sid
            sid += 1
            seq_array.append(seqs_list[id2index_dict[line[sid2_index]]])
        line[sid2_index] = id2_aid[line[sid2_index]]
        raw_data.append(line)
        if limit_data:
            count += 1
            if count >= max_data:
                break
    print(f'len(raw data) = {len(raw_data)}')

    len_m_seq = np.array([len(line.split()) for line in seq_array])
    avg_m_seq = int(np.average(len_m_seq)) + 1
    max_m_seq = max(len_m_seq)
    print(f'Avg len seq: {avg_m_seq}, max len seq: {max_m_seq}')

    dim = seq2t.dim
    seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])

    seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
    seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])
    class_map = {'0': 1, '1': 0}
    class_labels = np.zeros((len(raw_data), 2))
    for i in range(len(raw_data)):
        class_labels[i][class_map[raw_data[i][label_index]]] = 1.
    X = [seq_tensor[seq_index1], seq_tensor[seq_index2]]
    return dim, X, class_labels


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


def write_results(path, X, y):
    # copy below
    num_hit = 0.
    num_total = 0.
    num_pos = 0.
    num_true_pos = 0.
    num_false_pos = 0.
    num_true_neg = 0.
    num_false_neg = 0.
    preds = merge_model.predict(X)
    for i in range(len(y)):
        num_total += 1
        if np.argmax(y) == np.argmax(preds[i]):
            num_hit += 1
        if y[i][0] > 0.:
            num_pos += 1.
            if preds[i][0] > preds[i][1]:
                num_true_pos += 1
            else:
                num_false_neg += 1
        else:
            if preds[i][0] > preds[i][1]:
                num_false_pos += 1
            else:
                num_true_neg += 1
    # something was wrong with the accuracy: accuracy = num_hit / num_total
    accuracy = (num_true_pos + num_true_neg) / num_total
    prec = num_true_pos / (num_true_pos + num_false_pos)
    recall = num_true_pos / num_pos
    spec = num_true_neg / (num_true_neg + num_false_neg)
    f1 = 2. * prec * recall / (prec + recall)
    mcc = (num_true_pos * num_true_neg - num_false_pos * num_false_neg) / (
                (num_true_pos + num_true_neg) * (num_true_pos + num_false_neg) * (num_false_pos + num_true_neg) * (
                    num_false_pos + num_false_neg)) ** 0.5
    print(accuracy, prec, recall, spec, f1, mcc)

    with open(path, 'w') as fp:
        fp.write('acc=' + str(accuracy) + '\tprec=' + str(prec) + '\trecall=' + str(recall) + '\tspec=' + str(
            spec) + '\tf1=' + str(f1) + '\tmcc=' + str(mcc)+'\n')
        fp.write(f'TP={num_true_pos}, FP={num_false_pos}, TN={num_true_neg}, FN={num_false_neg}')


# Note: if you use another PPI dataset, this needs to be changed to a corresponding dictionary file.
id2seq_file = '../../../yeast/preprocessed/protein.dictionary.tsv'

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1

seq_size = 2000
n_epochs=20
batch_size1 = 256
#rst_file = 'results/yeast_wvctc_rcnn_25_5.txt'

# ds_file, label_index, rst_file, use_emb, hidden_dim
ds_file = '../../../yeast/preprocessed/protein.actions.tsv'
ds_file_partition0 = '../../../yeast/preprocessed/protein.actions_partition0.tsv'
ds_file_partition1 = '../../../yeast/preprocessed/protein.actions_partition1.tsv'
ds_file_both = '../../../yeast/preprocessed/protein.actions_both_partitions.tsv'

#dim, X_train, y_train = read_ppis(path=ds_file, id2index_dict=id2index, seqs_list=seqs, seq_size=seq_size)
dim0, X_part0, y_part0 = read_ppis(path=ds_file, id2index_dict=id2index, seqs_list=seqs, seq_size=seq_size)
dim1, X_part1, y_part1 = read_ppis(path=ds_file_partition1, id2index_dict=id2index, seqs_list=seqs, seq_size=seq_size)
dimb, X_both, y_both = read_ppis(path=ds_file_both, id2index_dict=id2index, seqs_list=seqs, seq_size=seq_size)

merge_model = build_model(seq_size, dim0)
adam = Adam(learning_rate=0.001, amsgrad=True, epsilon=1e-6)
merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

merge_model.fit(X_part0, y_part0, batch_size=batch_size1, epochs=n_epochs)
#write_results(path='results/yeast_partition0.txt', X=X_part0, y=y_part0)
write_results(path='results/yeast_partition1.txt', X=X_part1, y=y_part1)
write_results(path='results/yeast_both_partitions.txt', X=X_both, y=y_both)
