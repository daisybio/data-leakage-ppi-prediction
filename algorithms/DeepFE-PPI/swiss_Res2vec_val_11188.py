# -*- coding: utf-8 -*-

"""
Created on Wed Oct 24 09:54:33 2018

@author: yaoyu
"""

import keras
import os
from time import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Concatenate, concatenate
from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
import numpy as np
#from keras.layers.core import Dense, Dropout, Merge
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
import psutil
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
#from keras.optimizers import SGD


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

def train_validation__vis(hist,i,plot_dir,db):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['precision']
    val_acc = hist.history['val_precision']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_precision')
    ax2.plot(val_acc,label='val_precision')
    ax2.set_xlabel('Epochs')
    
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + db+'/round_'+str(i)+'.png')

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
    
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def max_min_avg_length(pro_swissProt):
    length = []
    for i in range(len(pro_swissProt)):
        length.append(len(pro_swissProt[i]))
    maxNum = max(length) #maxNum = 5
    minNum = min(length) #minNum = 1
    index_max = length.index(maxNum)
    index_min = length.index(minNum)
    avg = averagenum(length)
    return index_max,index_min,avg
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
    # read sample from a file
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                seq.append(line.split('\n')[0])
            i = i+1       
    return   seq   


              
def get_training_dataset(wv,  maxlen,size):

    file_1 = 'dataset/training_and_test_dataset/positive/Protein_A.txt'
    file_2 = 'dataset/training_and_test_dataset/positive/Protein_B.txt'
    file_3 = 'dataset/training_and_test_dataset/negative/Protein_A.txt'
    file_4 = 'dataset/training_and_test_dataset/negative/Protein_B.txt'
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
    mem_hstack = getMemorystate()
    #  creat label
    label = np.ones(len(feature_protein_A))
    label[len(feature_protein_AB)//2:] = 0
   
    return feature_protein_AB,label,mem_hstack
    
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
    line = "Memory: %5s%% %6s/%s"%(phymem.percent,
                                   str(int(phymem.used/1024/1024))+"M",
                                   str(int(phymem.total/1024/1024))+"M") 
    return line     
    
def classify(size,window,maxlen,train_fea_protein_AB,train_label):
    time_start_classify = time()
    sg = 'swissProt_size_'+str(size)+'_window_'+str(window)   
    db = sg+'_maxlen_'+str(maxlen)
    #db_dir= 'dataset/11188/different_size_represented_data/size_'+str(size)
    plot_dir = "plot/11188/"
    result_dir = "result/11188/performance/"
    model_dir = "model/dl/11188/"
    
    mkdir(plot_dir + db)          
    #mkdir(result_dir + db) 
    mkdir(model_dir + db) 
   
    sequence_len = size*maxlen


    
    Y = utils.to_categorical(train_label)  
    skf = StratifiedKFold(n_splits = 5,random_state= 20181031,shuffle= True)
    
  
    scores = []  
    i = 0
    mem_cv = []
    for (train_index, test_index) in skf.split(train_fea_protein_AB,train_label):
        print("================")

        print(test_index)
        print(train_index)
        X_train, X_val, y_train, y_val = train_test_split(train_fea_protein_AB[train_index], Y[train_index],random_state= 20181031, test_size=0.1,shuffle= True)
        
        X_train_left = X_train[:,0:sequence_len]
        X_train_right = X_train[:,sequence_len:sequence_len*2]

        X_validation_left = X_val[:,0:sequence_len]
        X_validation_right = X_val[:,sequence_len:sequence_len*2]

        X_test_left = train_fea_protein_AB[:,0:sequence_len][test_index]
        X_test_right = train_fea_protein_AB[:,sequence_len:sequence_len*2][test_index]

        # turn to np.array
        X_train_left  = np.array(X_train_left)
        X_train_right  = np.array(X_train_right)
        
        X_test_left  = np.array(X_test_left)
        X_test_right  = np.array(X_test_right)
        
        X_validation_left  = np.array(X_validation_left)
        X_validation_right  = np.array(X_validation_right)
        # label
        y_test = Y[test_index]
              
        # feed data into model
        #model =  merged_DBN(sequence_len)
        model = merged_DBN_functional(sequence_len)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.01,
                                      momentum=0.9,
                                      decay=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=[tf.keras.metrics.Precision()])
        #hist = model.fit([X_train_left, X_train_right], y_train,
        #          validation_data=([X_validation_left,X_validation_right],y_val),
        #          batch_size = 128,
        #          nb_epoch = 45,
        #          verbose = 1)
        hist = model.fit(
            {'left': X_train_left, 'right': X_train_right},
            {'ppi_pred': y_train},
            validation_data=({'left': X_validation_left, 'right': X_validation_right}, {'ppi_pred': y_val}),
            epochs=45,
            batch_size=128,
            verbose=1
        )
        mem_cv.append('round '+str(i)+' '+getMemorystate()) 
        train_validation__vis(hist,i,plot_dir,db)
        print('******   model created!  ******')
        model.save(model_dir + db+'/round_'+str(i)+'.h5')

        predictions_test = model.predict([X_test_left, X_test_right]) 
        
        auc_test = roc_auc_score(y_test[:,1], predictions_test[:,1])
        pr_test = average_precision_score(y_test[:,1], predictions_test[:,1])
     
        label_predict_test = utils.categorical_probas_to_classes(predictions_test)  
        tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= utils.calculate_performace(len(label_predict_test), label_predict_test, y_test[:,1])
        print(db+'    test:'+str(i))
        print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp_test,fp_test,tn_test,fn_test))
        print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
              % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test, f1_score_test))
        print('\tauc=%0.4f,pr=%0.4f'%(auc_test,pr_test))
        scores.append([accuracy_test,precision_test, recall_test,specificity_test, MCC_test, f1_score_test, auc_test,pr_test]) 
        
        i=i+1
        K.clear_session()
        #tf.reset_default_graph()
    
    sc= pd.DataFrame(scores)   
    sc.to_csv(result_dir+'5cv_'+db+'_scores.csv')   
    scores_array = np.array(scores)
    print (db+'_5cv:')
    print(("accuracy=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100)))
    print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100)))
    print("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
    print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
    print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
    print("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
    print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
    print("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
    time_end_classify = time()
     
    # memory and time for classify
    print('Time of create db('+db+'):', time_end_classify - time_start_classify)
    with open('runInfo/11188_val/cv_mem_time.txt','a') as f:
        f.write('Time of cv('+db+'):'+str(time_end_classify - time_start_classify))
        f.write('\n')
        f.write(mem_cv[0])
        f.write('\n')
        f.write(mem_cv[1])
        f.write('\n')
        f.write(mem_cv[2])
        f.write('\n')
        f.write(mem_cv[3])
        f.write('\n')
        f.write(mem_cv[4])
        f.write('\n')
             
        
    with open(result_dir+'5cv_'+db+'.txt','w') as f:
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
               

        
def res2vec(sizes,windows,maxlens):
#    data = token(get_res2vec_data())   # token
    for size in sizes:
        for window in windows:
#            t_start = time()
#            model = Word2Vec(data,
#                             size = size,  
#                             min_count = 0,
#                             sg =1,
#                             window = window)
#            memInfo_wv = getMemorystate()
#            print('memInfo_wv '+getMemorystate())
#            print('Word2Vec model is created')
            sg = 'wv_swissProt_size_'+str(size)+'_window_'+str(window) 
#            print('Time of creat Word2Vec model ('+sg+'):', time() - t_start)
#            with open('runInfo/word2vec_mem_time.txt','a') as f:
#               f.write('Time of creat Word2Vec model ('+sg+'):'+str(time() - t_start))
#               f.write('\n')
#               f.write(memInfo_wv)
#               f.write('\n')
#               
#            model.save('H:/model/word2vec/wv_'+sg+'.model')
              # load dictionary
            model_wv = Word2Vec.load('model/word2vec/'+sg+'.model')
                          
            for maxlen in maxlens:
                #maxlen =800
                data_start_maxlen = time()
                
                # get training data  
                train_fea_protein_AB,train_label,mem_hstack= get_training_dataset(model_wv.wv, maxlen,size )
              
                #scaler
                scaler = StandardScaler().fit(train_fea_protein_AB)
                train_fea_protein_AB = scaler.transform(train_fea_protein_AB)
                train_fea_protein_AB = np.array(train_fea_protein_AB)
                data_end_maxlen = time()
                               
                db = sg+'_maxlen_'+str(maxlen)
                db_dir= 'dataset/11188/different_size_represented_data/size_'+str(size)
                mkdir(db_dir)            
                # creat HDF5 file
                h5_file = h5py.File(db_dir + '/'+db+'.h5','w')
                h5_file.create_dataset('trainset_x', data = train_fea_protein_AB)
                h5_file.create_dataset('trainset_y', data = train_label)
                h5_file.close()
                '''
                h5_file = h5py.File(db_dir + '/'+db+'.h5','r')
                train_fea_protein_AB =  h5_file['trainset_x'][:]
                train_label = h5_file['trainset_y'][:]
                '''
                
                # memory and time for creat data
                print('Time of create db('+db+'):', data_end_maxlen - data_start_maxlen)
                with open('runInfo/11188_val/db_mem_time.txt','a') as f:
                    f.write('Time of create db('+db+'):'+str(data_end_maxlen - data_start_maxlen))
                    f.write('\n')
                    f.write('hstack '+ mem_hstack)
                    f.write('\n')
                    
                print('classify:')
                #5cv
                classify(size,window,maxlen,train_fea_protein_AB,train_label)
                                                 
#%%  
if __name__ == "__main__":  
    sizes = [4,8,12,16,20,24]  
    windows = [4,8,16,32]
    maxlens = [550,650,750,850]
    print('**************************************')
    print('**************************************')
    print('res2vec:')
    train_fea_protein_AB = res2vec(sizes,windows,maxlens) 
    print('**************************************')   
    print('**************************************')
     
        
                                                      
                                    
                                
                
                
                
                
                
                
