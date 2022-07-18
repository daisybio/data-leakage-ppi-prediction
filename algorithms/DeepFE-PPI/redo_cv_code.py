# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:41:12 2018

@author: xal
"""
import numpy as np
import random

from time import time
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Concatenate, concatenate
from sklearn.metrics import roc_auc_score,average_precision_score
#from keras.layers.core import Dense, Dropout, Merge
import utils.tools as utils
from keras.regularizers import l2
import h5py
from keras import backend as K
import tensorflow as tf
import pandas as pd
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



if __name__ == "__main__":  
    t_start = time()
    h5_file = h5py.File('dataset/11188/wv_swissProt_size_20_window_4_maxlen_850.h5','r')
    train_fea_protein_AB =  h5_file['trainset_x'][:]
    X = np.array(train_fea_protein_AB)
    train_label = h5_file['trainset_y'][:]
    h5_file.close()
    
    num_pos = 11188//2
    
    num_neg = 11188//2
    X_pos = X[:num_pos]
    X_neg = X[-num_neg:]
    #shuffle
    random.seed(20181031)
    index_pos = [i for i in range(num_pos)]
    random.shuffle(index_pos) 
    X_pos = X_pos[index_pos]
    random.seed(20181031)
    index_neg =[i for i in range(num_neg)]
    random.shuffle(index_neg) 
    X_neg = X_neg[index_neg]


    # creat label
    label_pos = np.ones(len(X_pos))
    label_neg = np.zeros(len(X_neg))
   
    k = 5
    num_fold_pos = len(X_pos)//k
    num_fold_neg = len(X_neg)//k
    i = 0
    scores = []  
    for i in range(k):
        #print(i)
        i = 0
        # test data
        X_fold_pos_test = X_pos[i*num_fold_pos:(i+1)*num_fold_pos]
        X_fold_neg_test = X_neg[i*num_fold_neg:(i+1)*num_fold_neg]
                 
        # 测试序列的标签，分为正负                        
        Y_test = np.ones(len(X_fold_pos_test)).tolist()
        Y_test_neg = np.zeros(len(X_fold_neg_test)).tolist()
        
        X_test= np.vstack((X_fold_pos_test,X_fold_neg_test))
        X_test_left = X_test[:,0:len(X_test[0])//2]
        X_test_right = X_test[:,len(X_test[0])//2:len(X_test[0])] 
        Y_test.extend(Y_test_neg)
        Y_test = utils.to_categorical(Y_test)
        # train data
        
        X_fold_pos_before = X_pos[:i*num_fold_pos]
        X_fold_neg_before = X_neg[:i*num_fold_neg]
                              
        X_fold_pos_after = X_pos[(i+1)*num_fold_pos:]
        X_fold_neg_after = X_neg[(i+1)*num_fold_neg:]
                                 
                                 
        X_train_pos = np.vstack((np.array(X_fold_pos_before),np.array(X_fold_pos_after)))
        X_train_neg = np.vstack((np.array(X_fold_neg_before),np.array(X_fold_neg_after)))
        
        X_train = np.vstack((np.array(X_train_pos),np.array(X_train_neg)))
        X_train_left = X_train[:,0:len(X_train[0])//2]
        X_train_right = X_train[:,len(X_train[0])//2:len(X_train[0])]
        
        # 训练序列的标签，分为正负                        
        Y_train = np.ones(len(X_train_pos)).tolist()
        Y_train_neg = np.zeros(len(X_train_neg)).tolist()
        
        Y_train.extend(Y_train_neg)
        Y_train = np.array(Y_train)
        Y_train = utils.to_categorical(Y_train)
        
        #model =  merged_DBN(len(X_train[0])//2)
        model = merged_DBN_functional(len(X_train[0])//2)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=[tf.keras.metrics.Precision()])
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
        # feed data into model
        #hist = model.fit([X_train_left, X_train_right], Y_train,
        #          batch_size = 256,
        #          nb_epoch = 45,
        #          verbose = 1)
        hist = model.fit(
            {'left': X_train_left, 'right': X_train_right},
            {'ppi_pred': Y_train},
            epochs=45,
            batch_size=256,
            verbose=1
        )
        
        print('******   model created!  ******')
        print('******   model created!  ******')
        model.save('model/rewrite_cv/model.h5')
        predictions_test = model.predict([X_test_left, X_test_right]) 
        
        auc_test = roc_auc_score(Y_test[:,1], predictions_test[:,1])
        pr_test = average_precision_score(Y_test[:,1], predictions_test[:,1])
     
        label_predict_test = utils.categorical_probas_to_classes(predictions_test)  
        tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= utils.calculate_performace(len(label_predict_test), label_predict_test, Y_test[:,1])
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
    scores_array = np.array(scores)
            
    sc.to_csv('rewrite_cv.csv')             
#
    
    print(("accuracy=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100)))
    print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100)))
    print("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
    print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
    print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
    print("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
    print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
    print("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
    print(time() - t_start)
 