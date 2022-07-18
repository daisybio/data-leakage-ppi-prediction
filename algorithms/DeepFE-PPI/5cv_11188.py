# -*- coding: utf-8 -*-

"""
Created on Wed Oct 24 09:54:33 2018

@author: yaoyu
"""

import os
import keras
from time import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, concatenate
from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
import numpy as np
#from keras.layers.core import Dense, Dropout#, Merge
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
#from keras.optimizers import SGD

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)
def plot(length):
    reversed_length =  sorted(length,reverse=True)
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
    maxNum = max(length) #maxNum = 5
    minNum = min(length) #minNum = 1
    
    avg = averagenum(length)
    
    print('The longest length of protein is: '+str(maxNum))
    print('The shortest length of protein is: '+str(minNum))
    print('The avgest length of protein is: '+str(avg))

def merged_DBN(sequence_len):
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
    #model.summary()
    
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
    
# define the function
def training_vis(hist,i,plot_dir,swm,be):
    loss = hist.history['loss']
    #val_loss = hist.history['val_loss']
    acc = hist.history['precision']
    #val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    #ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Traingng Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_precision')
    #ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision on Traingng Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + swm+be+'/round_'+str(i)+'.png')
    
    
def read_file(file_name):
    pro_swissProt = []
    with open(file_name, 'r') as fp:
        protein = ''
        for line in fp:
            if line.startswith('>sp|'):
                pro_swissProt.append(protein)
                protein = ''
            elif line.startswith('>sp|') == False:
                protein = protein+line.strip()
              
    return   pro_swissProt[1:]   

    
def get_res2vec_data():
    file_name = 'dataset/uniprot_sprot.fasta'
    pro_swissProt = read_file(file_name)
    
    return pro_swissProt
    

    
def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])

        token_dataset.append(seq)  
                
    return  token_dataset
    
def pandding_J(protein,maxlen):           
    padded_protein = copy.deepcopy(protein)   
    for i in range(len(padded_protein)):
        if len(padded_protein[i])<maxlen:
            for j in range(len(padded_protein[i]),maxlen):
                padded_protein[i].append('J')
    return padded_protein  
    

def protein_representation(wv,tokened_seq_protein,maxlen,size):  
    represented_protein  = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j]=='J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))    
    return represented_protein
    
def read_traingingData(file_name):
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                seq.append(line.split('\n')[0])
            i = i+1       
    return   seq   


              
def get_training_dataset(wv,  maxlen,size):

    file_1 = 'dataset/training and test dataset/positive/Protein_A.txt'
    file_2 = 'dataset/training and test dataset/positive/Protein_B.txt'
    file_3 = 'dataset/training and test dataset/negative/Protein_A.txt'
    file_4 = 'dataset/training and test dataset/negative/Protein_B.txt'
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
    tokened_token_pos_neg_seq_protein_A = pandding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = pandding_J(token_pos_neg_seq_protein_B,maxlen)
    # protein reprsentation
    feature_protein_A  = protein_representation(wv,tokened_token_pos_neg_seq_protein_A,maxlen,size)
    feature_protein_B  = protein_representation(wv,tokened_token_pos_neg_seq_protein_B,maxlen,size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A),np.array(feature_protein_B)))
    #  creat label
    label = np.ones(len(feature_protein_A))
    label[len(feature_protein_AB)//2:] = 0
   
    return feature_protein_AB,label
    #
    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                 
        os.makedirs(path)           
        print("---  new folder...  ---")
        print("---  OK  ---")
 
    else:
        print("---  There is this folder!  ---")


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
   
    # load dictionary
    model_wv = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')
    
  
    plot_dir = 'plot/11188/'
    
    sizes = [20]
    windows = [4]
    maxlens = [850]  # 550,650,750,850
    batch_sizes = [256] # 32,64,128,256
    nb_epoches = [45] 
    for size in sizes:
        for window in windows:
            for maxlen in maxlens:
                for batch_size in batch_sizes:
                    for nb_epoch in nb_epoches:
                        sequence_len = size*maxlen
                       
                        # get training data 
                        t_start = time()
                        #h5_file = h5py.File('dataset/11188/wv_swissProt_size_20_window_4_maxlen_'+str(maxlen)+'.h5','r')
                        #train_fea_protein_AB =  h5_file['trainset_x'][:]
                        #train_label = h5_file['trainset_y'][:]
                        #h5_file.close()
                        train_fea_protein_AB,train_label= get_training_dataset(model_wv.wv, maxlen,size)
                       
                        print('dataset is loaded')
                        swm = 'swissProt_size_'+str(size)+'_window_'+str(window)+'_maxlen_'+str(maxlen)
                       
#                        #scaler
                        scaler = StandardScaler().fit(train_fea_protein_AB)
                        train_fea_protein_AB = scaler.transform(train_fea_protein_AB)
#                    
                        db_dir= 'dataset/11188/different_size_represented_data/size_'+str(size)
                        mkdir(db_dir)
#                        # creat HDF5 file
                        h5_file = h5py.File(db_dir + '/'+swm+'.h5','w')
                        h5_file.create_dataset('trainset_x', data = train_fea_protein_AB)
                        h5_file.create_dataset('trainset_y', data = train_label)
                        h5_file.close()
#

                        Y = utils.to_categorical(train_label)
                        skf = StratifiedKFold(n_splits = 5,random_state= 20181031,shuffle= True)
                      
                        scores = []  
                        i = 0
                      
                        be = '_batch_size_'+str(batch_size)+'_nb_epoch_'+str(nb_epoch)
                        model_dir = 'model/dl/11188/'
                        result_dir = 'result/5cv/11188/'
                        mkdir(result_dir)
                        for (train_index, test_index) in skf.split(train_fea_protein_AB,train_label):
                            print("================")
                            print(test_index)
                            print(train_index)

                            ########### test 1: ###############
                            # shuffle labels
                            #y_train = np.argmax(Y[train_index], axis=1)
                            #print(f'y train first 10: {y_train[0:10]}')
                            #np.random.shuffle(y_train)
                            #print(f'y train first 10: {y_train[0:10]}')
                            #y_train = utils.to_categorical(y_train)
                            #print('shuffled!')
                            ###################################

                            ########## test 2: #################
                            # rewire positive network: true=expected deg, false=graph tool rewired
                            #y_train = np.argmax(Y[train_index], axis=1)
                            #X_train_rewired, y_train = generate_RDPN(train_fea_protein_AB[train_index],
                            #                                                               y_train,
                            #                                                  sequence_len, True)
                            #y_train = utils.to_categorical(y_train)
                            #print("rewired!")
                            #X_train_left = X_train_rewired[:, 0:sequence_len]
                            #X_train_right = X_train_rewired[:, sequence_len:sequence_len * 2]

                            X_train_left = train_fea_protein_AB[train_index][:,0:sequence_len]
                            X_train_right = train_fea_protein_AB[train_index][:,sequence_len:sequence_len*2]
                         
                            X_test_left = train_fea_protein_AB[test_index][:,0:sequence_len]
                            X_test_right = train_fea_protein_AB[test_index][:,sequence_len:sequence_len*2]
                    
                            # turn to np.array
                            X_train_left  = np.array(X_train_left)
                            X_train_right  = np.array(X_train_right)
                            
                            X_test_left  = np.array(X_test_left)
                            X_test_right  = np.array(X_test_right)
                                
                            # label
                            y_train = Y[train_index]
                            y_test = Y[test_index]


                            #model =  merged_DBN(sequence_len)
                            model = merged_DBN_functional(sequence_len)
                            sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
                            
                            model.compile(loss='categorical_crossentropy',
                                          optimizer=sgd,
                                          metrics=[tf.keras.metrics.Precision()])
                            # feed data into model
                            #hist = model.fit([X_train_left, X_train_right], y_train,
                            #          batch_size = batch_size,
                            #          nb_epoch = nb_epoch,
                            #          verbose = 1)
                            hist = model.fit(
                                {'left': X_train_left, 'right': X_train_right},
                                {'ppi_pred': y_train},
                                epochs=nb_epoch,
                                batch_size=nb_epoch,
                                verbose=1
                            )
                            
                            print('******   model created!  ******')
                            mkdir(model_dir + swm+be+'/')
                            mkdir(plot_dir + swm+be+'/')
                            training_vis(hist,i,plot_dir,swm,be)
                            model.save(model_dir + swm+be+'/round_'+str(i)+'.h5')
                    
                            predictions_test = model.predict([X_test_left, X_test_right]) 
                            
                            auc_test = roc_auc_score(y_test[:,1], predictions_test[:,1])
                            pr_test = average_precision_score(y_test[:,1], predictions_test[:,1])
                         
                            label_predict_test = utils.categorical_probas_to_classes(predictions_test)  
                            tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= utils.calculate_performace(len(label_predict_test), label_predict_test, y_test[:,1])
                            print(' ===========  test:'+str(i))
                            print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp_test,fp_test,tn_test,fn_test))
                            print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
                                  % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test, f1_score_test))
                            print('\tauc=%0.4f,pr=%0.4f'%(auc_test,pr_test))
                            scores.append([accuracy_test,precision_test, recall_test,specificity_test, MCC_test, f1_score_test, auc_test,pr_test]) 
                            
                            i=i+1
                            K.clear_session()
                            #tf.reset_default_graph()
                        
                        sc= pd.DataFrame(scores)   
                        sc.to_csv(result_dir+swm+be+'.csv')
                        scores_array = np.array(scores)
                        print(("accuracy=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100)))
                        print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100)))
                        print("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
                        print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
                        print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
                        print("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
                        print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
                        print("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
                      
                        print(time() - t_start)

                        with open(result_dir+'shuffled_5cv_'+swm+be+'.txt','w') as f:
                            f.write('accuracy=%.2f%% (+/- %.2f%%)' % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100))
                            f.write('\n')
                            f.write("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100))
                            f.write('\n')
                            f.write("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
                            f.write('\n')
                            f.write("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
                            f.write('\n')
                            f.write("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
                            f.write('\n')
                            f.write("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
                            f.write('\n')
                            f.write("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
                            f.write('\n')
                            f.write("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
                            f.write('\n')
                            f.write('\n')