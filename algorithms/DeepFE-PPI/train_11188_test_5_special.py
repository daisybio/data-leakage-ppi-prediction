# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:27:29 2018

@author: yaoyu
"""

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Concatenate, concatenate
import numpy as np
#from keras.layers.core import Dense, Dropout, Merge
import utils.tools as utils
from keras.regularizers import l2
import pandas as pd
from gensim.models.word2vec import Word2Vec
import copy
import h5py
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

     
def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])
        token_dataset.append(seq)  
        
    return  token_dataset

def connect(protein_A,protein_B):
    # contect protein A and B
    protein_AB = []
    for i in range(len(protein_A)):
        con = protein_A[i] + protein_B[i] 
        protein_AB.append(con)
        
    return np.array(protein_AB)
 
    

def pandding_J(protein,maxlen):           
    padded_protein = copy.deepcopy(protein)   
    for i in range(len(padded_protein)):
        if len(padded_protein[i])<maxlen:
            for j in range(len(padded_protein[i]),maxlen):
                padded_protein[i].append('J')
    return padded_protein  
       

def residue_representation(wv,tokened_seq_protein,maxlen,size):  
    represented_protein  = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j]=='J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))    
   
    return np.array(represented_protein)
    

    
def read_file(file_name):    
    # read sample from a file
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                seq.append(line.split('\n')[0])
            i = i+1            
    return seq 
    
    
def read_protein_pair(file_protein_A,file_protein_B):
     
    # read  sample from protein A
    seq_protein_A = read_file(file_protein_A)
    
    # read  sample from protein B
    seq_protein_B = read_file(file_protein_B)
    
    # contect protein A and B
    seq_protein_AB = connect(seq_protein_A,seq_protein_B )
          
    return seq_protein_AB
    

    
def protein_represetation(wv,seq_protein_A,seq_protein_B,maxle,size):
    # token
    token_protein_A = token(seq_protein_A)
    token_protein_B = token(seq_protein_B)
    #padding
    padded_token_protein_A = pandding_J(token_protein_A,maxlen)
    padded_token_protein_B = pandding_J(token_protein_B,maxlen)
                   
    # generate protein representation
    represented_protein_A  = residue_representation(wv,padded_token_protein_A,maxlen,size)
    represented_protein_B  = residue_representation(wv,padded_token_protein_B,maxlen,size)
      
    #put two part togeter
    represented_protein_AB = np.hstack((np.array(represented_protein_A),np.array(represented_protein_B)))

    return represented_protein_AB
    
    
def get_test_dataset(wv, file_pro_A,file_pro_B,maxlen,size): 
#    file_pro_A = file_pro_A_Celeg
#    file_pro_B = file_pro_B_Celeg
#    
    # get sequences
    seq_protein_A =  read_file(file_pro_A)
    seq_protein_B =  read_file(file_pro_B)
    
    represented_protein_AB = protein_represetation(wv,seq_protein_A,seq_protein_B,maxlen,size)
  
    # creat label
    label = np.ones(len(represented_protein_AB))
     
    return represented_protein_AB,label   
 


   
def merged_DBN():
    # left model
    model_left = Sequential()
    model_left.add(Dense(2048, input_dim=sequence_len ,activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    
    
    # right model
    model_right = Sequential()
    model_right.add(Dense(2048,input_dim=sequence_len,activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))   
    model_right.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    # together
    merged = Concatenate([model_left, model_right])
  
    
    model = Sequential()
    model.add(merged)
    model.add(Dense(8, activation='relu',W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
   
    
    return model


def merged_DBN_functional(sequence_len):
    # left model
    model_left_input = keras.Input(shape=(sequence_len, ), name='left')
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
    model_right_input = keras.Input(shape=(sequence_len, ), name='right')
    model_right = Dense(2048,  activation='relu', kernel_regularizer=l2(0.01), name='right_dense1')(model_right_input)
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


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def max_min_avg_length(seq):
    length = []
    for string in seq:
        length.append(len(string))
    maxNum = max(length) #maxNum = 5
    minNum = min(length) #minNum = 1
    
    avg = averagenum(length)
    print('The longest length of protein is: '+str(maxNum))
    print('The shortest length of protein is: '+str(minNum))
    print('The avgest length of protein is: '+str(avg))

def get_shuffle(dataset,label,random_state):    
    #shuffle data
    np.random.seed(random_state)
    index = list(range(len(label)))
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label             

    
def read_traingingData(file_name):
    # read sample from a file
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                seq.append(line.split('\n')[0])
            i = i+1       
    return   seq 


def generate_RDPN(encoding, label, sequence_len, expected=True):
    import networkx as nx

    protsA = encoding[:, 0:sequence_len]
    protsB = encoding[:, sequence_len:sequence_len * 2]

    g1 = nx.Graph()
    g2 = nx.Graph()
    pos_edges = []
    neg_edges = []
    id = 0
    hash_to_id = dict()
    for i in range(len(protsA)):
        protA_enc = protsA[i, :]
        protA_hash = hash(protA_enc.tostring())
        protB_enc = protsB[i, :]
        protB_hash = hash(protB_enc.tostring())
        if protA_hash not in hash_to_id:
            g1.add_node(id, encoding=protA_enc)
            g2.add_node(id, encoding=protA_enc)
            hash_to_id[protA_hash] = id
            id += 1
        if protB_hash not in hash_to_id:
            g1.add_node(id, encoding=protB_enc)
            g2.add_node(id, encoding=protB_enc)
            hash_to_id[protB_hash] = id
            id += 1

        if label[i] == 1:
            pos_edges.append((hash_to_id.get(protA_hash),
                              hash_to_id.get(protB_hash)))
        else:
            neg_edges.append((hash_to_id.get(protA_hash),
                              hash_to_id.get(protB_hash)))
    g1.add_edges_from(pos_edges)
    g2.add_edges_from(neg_edges)
    if expected:
        degree_view = g1.degree()
        degree_sequence = [degree_view[node] for node in g1.nodes()]
        rewired_network = nx.expected_degree_graph(degree_sequence, seed=1234, selfloops=False)
    else:
        import graph_tool.all as gt
        d = nx.to_dict_of_lists(g1)
        edges = [(i, j) for i in d for j in d[i]]
        GT = gt.Graph(directed=False)
        GT.add_vertex(sorted(g1.nodes())[-1])
        GT.add_edge_list(edges)

        gt.random_rewire(GT, model="constrained-configuration", n_iter=100, edge_sweep=True)

        edges_new = list(GT.get_edges())
        edges_new = [tuple(x) for x in edges_new]
        rewired_network = nx.Graph()
        rewired_network.add_nodes_from(g1.nodes())
        rewired_network.add_edges_from(edges_new)
    edge_list = nx.generate_edgelist(rewired_network)
    protsA_rewired = []
    protsB_rewired = []
    labels_rewired = []
    idx = 0
    for edge in edge_list:
        encoding0 = g1.nodes[int(edge.split()[0])]['encoding']
        protsA_rewired.append(encoding0)
        encoding1 = g1.nodes[int(edge.split()[1])]['encoding']
        protsB_rewired.append(encoding1)
        labels_rewired.append(1)
        idx += 1

    edge_list1 = nx.generate_edgelist(g2)
    for edge in edge_list1:
        encoding0 = g2.nodes[int(edge.split()[0])]['encoding']
        protsA_rewired.append(encoding0)
        encoding1 = g2.nodes[int(edge.split()[1])]['encoding']
        protsB_rewired.append(encoding1)
        labels_rewired.append(0)
        idx += 1

    rewired_encoding = np.hstack((np.array(protsA_rewired),np.array(protsB_rewired)))
    rewired_labels = np.array(labels_rewired)
    return rewired_encoding, rewired_labels

#%%    
if __name__ == "__main__":   
   
    size =20
    maxlen  = 850   
    sequence_len = maxlen*20
    # load dictionary
  
    model_vec = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')
    
     # get training data 
 
    h5_file = h5py.File('dataset/11188/different_size_represented_data/size_20/swissProt_size_20_window_4_maxlen_850.h5','r')
    train_fea_protein_AB =  h5_file['trainset_x'][:]
    train_label = h5_file['trainset_y'][:]   
    h5_file.close()

    # shuffle
    train_fea_protein_AB,train_label = get_shuffle(train_fea_protein_AB,train_label,100)

    ########### test 1: ###############
    # shuffle labels
    #np.random.shuffle(train_label)
    ###################################

    ########## test 2: #################
    # rewire positive network: True=expected degrees, False: graph tool rewired
    train_fea_protein_AB, train_label = generate_RDPN(train_fea_protein_AB, train_label,
                                                      sequence_len, False)
    # y_train = utils.to_categorical(y_train)
    # print("rewired.")
    # X_train_left = X_train_rewired[:, 0:sequence_len]
    # X_train_right = X_train_rewired[:, sequence_len:sequence_len * 2]

    fea_protein_A = train_fea_protein_AB[:,0:sequence_len]
    fea_protein_B = train_fea_protein_AB[:,sequence_len:sequence_len*2]

    # label 
    Y = utils.to_categorical(train_label) 
    
    print('===========  begin to create model  =========')
    # feed data to model
    #model =  merged_DBN()
    model = merged_DBN_functional(sequence_len)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01,
                                  momentum=0.9,
                                  decay=0.001)
                           
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=[tf.keras.metrics.Precision()])
                           
    #model.fit([fea_protein_A, fea_protein_B], Y,
    #          batch_size = 64,
    #          nb_epoch = 25,
    #          verbose = 1)

    hist = model.fit(
        {'left': fea_protein_A, 'right': fea_protein_B},
        {'ppi_pred': Y},
        epochs=25,
        batch_size=64,
        verbose=1
    )

    #model.save('model/train_11188_test_5_special/train_11188_test_5_special.h5')

    print('===========  model created!  =========')
    
    #model = load_model('model/train_11188_test_5_special.h5')
    
    #  %%%%%%%%%%%%%%%%%%%%%%%%%     Celeg            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #scaler
    
    file_pro_A_Celeg = 'dataset/cross species dataset/Celeg_ProA.txt'
    file_pro_B_Celeg = 'dataset/cross species dataset/Celeg_ProB.txt'
    fea_protein_AB_Celeg,label_Celeg = get_test_dataset(model_vec.wv, file_pro_A_Celeg,file_pro_B_Celeg,maxlen,size)
    fea_protein_AB_Celeg = np.array(fea_protein_AB_Celeg)
    scaler = StandardScaler().fit(train_fea_protein_AB)
    
    fea_protein_AB_Celeg  = scaler.transform(fea_protein_AB_Celeg)
#    h5_file.create_dataset('Celeg_x', data = fea_protein_AB_Celeg)
#    h5_file.create_dataset('Celeg_y', data = label_Celeg)
#    
    #fea_protein_AB_Celeg = h5_file['Celeg_x'][:]
    fea_protein_A_Celeg = fea_protein_AB_Celeg[:,0:sequence_len]
    fea_protein_B_Celeg = fea_protein_AB_Celeg[:,sequence_len:sequence_len*2]
    predictions_test_Celeg = model.predict([fea_protein_A_Celeg, fea_protein_B_Celeg])  
    label_predict_test_Celeg_list = list(utils.categorical_probas_to_classes(predictions_test_Celeg) )
    accuracy_test_Celeg = 1.0*label_predict_test_Celeg_list.count(1)/len(label_predict_test_Celeg_list)
    print('\taccuracy_test_Celeg=%0.4f'% (accuracy_test_Celeg))
    
    
    #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         Ecoli            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    file_pro_A_Ecoli = 'dataset/cross species dataset/Ecoli_ProA.txt'
    file_pro_B_Ecoli = 'dataset/cross species dataset/Ecoli_ProB.txt'
    fea_protein_AB_Ecoli,label_Ecoli = get_test_dataset(model_vec.wv,file_pro_A_Ecoli,file_pro_B_Ecoli,maxlen,size)
    fea_protein_AB_Ecoli = np.array(fea_protein_AB_Ecoli)
    fea_protein_AB_Ecoli  = scaler.transform(fea_protein_AB_Ecoli)
#    h5_file.create_dataset('Ecoli_x', data = fea_protein_AB_Ecoli)
#    h5_file.create_dataset('Ecoli_y', data = label_Ecoli)
#    
    #fea_protein_AB_Ecoli = h5_file['Ecoli_x'][:]    
    fea_protein_A_Ecoli = fea_protein_AB_Ecoli[:,0:sequence_len]
    fea_protein_B_Ecoli = fea_protein_AB_Ecoli[:,sequence_len:sequence_len*2]
    # predict 
    predictions_test_Ecoli = model.predict([fea_protein_A_Ecoli, fea_protein_B_Ecoli])  
    label_predict_test_Ecoli_list = list(utils.categorical_probas_to_classes(predictions_test_Ecoli) )
    accuracy_test_Ecoli = 1.0*label_predict_test_Ecoli_list.count(1)/len(label_predict_test_Ecoli_list)
    print('\taccuracy_test_Ecoli=%0.4f'% (accuracy_test_Ecoli))
  
    #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         Hpylo            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    file_pro_A_Hpylo = 'dataset/cross species dataset/Hpylo_ProA.txt'
    file_pro_B_Hpylo = 'dataset/cross species dataset/Hpylo_ProB.txt'
    fea_protein_AB_Hpylo,label_Hpylo = get_test_dataset(model_vec.wv,file_pro_A_Hpylo,file_pro_B_Hpylo,maxlen,size)
    fea_protein_AB_Hpylo = np.array(fea_protein_AB_Hpylo)
    fea_protein_AB_Hpylo  = scaler.transform(fea_protein_AB_Hpylo)
#    h5_file.create_dataset('Hpylo_x', data = fea_protein_AB_Hpylo)
#    h5_file.create_dataset('Hpylo_y', data = label_Hpylo)
#    
    #fea_protein_AB_Hpylo = h5_file['Hpylo_x'][:]    
    fea_protein_A_Hpylo = fea_protein_AB_Hpylo[:,0:sequence_len]
    fea_protein_B_Hpylo = fea_protein_AB_Hpylo[:,sequence_len:sequence_len*2]
    # predict  
    predictions_test_Hpylo = model.predict([fea_protein_A_Hpylo, fea_protein_B_Hpylo])  
    label_predict_test_Hpylo_list = list(utils.categorical_probas_to_classes(predictions_test_Hpylo) )
    accuracy_test_Hpylo = 1.0*label_predict_test_Hpylo_list.count(1)/len(label_predict_test_Hpylo_list)
    print('\taccuracy_test_Hpylo=%0.4f'% (accuracy_test_Hpylo))
           

     #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         Hsapi            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    file_pro_A_Hsapi = 'dataset/cross species dataset/Hsapi_ProA.txt'
    file_pro_B_Hsapi = 'dataset/cross species dataset/Hsapi_ProB.txt'
    fea_protein_AB_Hsapi,label_Hsapi = get_test_dataset(model_vec.wv,file_pro_A_Hsapi,file_pro_B_Hsapi,maxlen,size)
    fea_protein_AB_Hsapi = np.array(fea_protein_AB_Hsapi)
    fea_protein_AB_Hsapi  = scaler.transform(fea_protein_AB_Hsapi)
#    h5_file.create_dataset('Hsapi_x', data = fea_protein_AB_Hsapi)
#    h5_file.create_dataset('Hsapi_y', data = label_Hsapi)
#    
    #fea_protein_AB_Hsapi = h5_file['Hsapi_x'][:]
    fea_protein_A_Hsapi = fea_protein_AB_Hsapi[:,0:sequence_len]
    fea_protein_B_Hsapi = fea_protein_AB_Hsapi[:,sequence_len:sequence_len*2]
    # predict  
    predictions_test_Hsapi = model.predict([fea_protein_A_Hsapi, fea_protein_B_Hsapi]) 
    label_predict_test_Hsapi_list = list(utils.categorical_probas_to_classes(predictions_test_Hsapi) )
    accuracy_test_Hsapi = 1.0*label_predict_test_Hsapi_list.count(1)/len(label_predict_test_Hsapi_list)
    print('\taccuracy_test_Hsapi=%0.4f'% (accuracy_test_Hsapi))
     #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         Mmusc            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    file_pro_A_Mmusc = 'dataset/cross species dataset/Mmusc_ProA.txt'
    file_pro_B_Mmusc = 'dataset/cross species dataset/Mmusc_ProB.txt'
    fea_protein_AB_Mmusc,label_Mmusc = get_test_dataset(model_vec.wv,file_pro_A_Mmusc,file_pro_B_Mmusc,maxlen,size)
    fea_protein_AB_Mmusc = np.array(fea_protein_AB_Mmusc)
    fea_protein_AB_Mmusc  = scaler.transform(fea_protein_AB_Mmusc)
#    h5_file.create_dataset('Mmusc_x', data = fea_protein_AB_Mmusc)
#    h5_file.create_dataset('Mmusc_y', data = label_Mmusc)  
#    
    #fea_protein_AB_Mmusc = h5_file['Mmusc_x'][:]

    fea_protein_A_Mmusc = fea_protein_AB_Mmusc[:,0:sequence_len]
    fea_protein_B_Mmusc = fea_protein_AB_Mmusc[:,sequence_len:sequence_len*2]
    # predict  
    predictions_test_Mmusc = model.predict([fea_protein_A_Mmusc, fea_protein_B_Mmusc])  #  两列
    label_predict_test_Mmusc_list = list(utils.categorical_probas_to_classes(predictions_test_Mmusc) )
    accuracy_test_Mmusc = 1.0*label_predict_test_Mmusc_list.count(1)/len(label_predict_test_Mmusc_list)
    print('\taccuracy_test_Mmusc =%0.4f'% (accuracy_test_Mmusc))
    

    
    
    
