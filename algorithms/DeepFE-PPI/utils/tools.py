# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:09:17 2016

@author: sun

@type:sun's utils

categorical_probas_to_classes
to_categorical
plothistory
calculate_performace
plot_confusion_matrix
draw_roc
draw_pr
plot_embedding
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd
import math
  

def read_result_file(filename = 'result/result_file'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index % 3 == 0:
                rna = values[0]
            if index % 3 != 0:
                results.setdefault(rna, []).append(values)
                
                
            index = index + 1
    
    return results

def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )
        
def plot_figure():
    results = read_result_file('result/individual_acceptor_roc')
    
    Figure = plt.figure(figsize=(12, 15))
    #print "*******************************"
    #print results
    for key, values in results.iteritems():  
        rna = key
        #pdb.set_trace()
        labels = [int(float(val)) for val in values[0]]
        probability = [float(val) for val in values[1]]
        plot_roc_curve(labels, probability, rna)
    #plot_roc_curve(labels[1], probability[1], '')
    #plot_roc_curve(labels[2], probability[2], '')
    
    #title_type = 'stem cell circRNAs vs other circRNAs'
    title_type = 'ROC'
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(title_type)
    plt.legend(loc="lower right")
    plt.savefig('roc1.eps', format='eps') 
    #plt.show() 
  
def plot_parameter_bar(menMeans, xlabel):
    methodlabel = ['position', 'dependency', 'composition', 'diff1','diff2','diff3', 'seq', 'iDeep']
    
    #xval = [5, 10, 20, 30, 40, 50, 60,70]#, 80, 90]
    width = 0.10
    ind = np.arange(len(menMeans[0]))
    fig, ax = plt.subplots(figsize=(12,12))
    #pdb.set_trace()
    #plt.plot(xval,menMeans)
    rects1 = plt.barh(ind, menMeans[0], width, color='r')
    rects2 = plt.barh(ind +width, menMeans[1], width, color='g')
    rects3 = plt.barh(ind +2*width, menMeans[2], width, color='y')
    rects4 = plt.barh(ind+3*width, menMeans[3], width, color='b')
    rects5 = plt.barh(ind+4*width, menMeans[4], width, color='m')
    rects6 = plt.barh(ind+5*width, menMeans[5], width, color='c')
    #plt.title('stem cell circRNA vs other circRNA')
    ax.set_xlabel('AUC', fontsize=20)
    #plt.xlabel('Number of trees', fontsize=20)
    #ax.set_ylim([0.6, 0.75])
    ax.set_yticks(ind)
    ax.set_yticklabels(xlabel )
    #plt.margins(0.1)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0],  rects6[0]), ('region type', 'clip-cobinding', 'structure', 'motif', 'CNN sequence', 'iDeep'), 
              loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True)
    plt.tight_layout()
    
    plt.show()
    
def plot_parameter_bar():
    methodlabel = ['position', 'dependency', 'composition', 'diff1','diff2','diff3', 'seq', 'iDeep']
    
    #xval = [5, 10, 20, 30, 40, 50, 60,70]#, 80, 90]
    auc_HS3D_IE= [0.501126475694,0.500725086806,0.511351388889,0.499545138889,0.501341493056,0.499742708333,0.997405729167,0.93798828125]
    width = 0.10
    index = np.arange(len(menMeans[0]))
    fig, ax = plt.subplots(figsize=(12,12))
    #pdb.set_trace()
    #plt.plot(xval,menMeans)
    rects1 = plt.bar(index, menMeans[0], width, color='r')
    rects2 = plt.bar(index +width, menMeans[1], width, color='g')
    rects3 = plt.bar(index +2*width, menMeans[2], width, color='y')
    rects4 = plt.bar(index+3*width, menMeans[3], width, color='b')
    rects5 = plt.bar(index+4*width, menMeans[4], width, color='m')
    rects6 = plt.bar(index+5*width, menMeans[5], width, color='c')
    rects7 = plt.bar(index+6*width, menMeans[4], width, color='m')
    rects8 = plt.bar(index+7*width, menMeans[5], width, color='c')
    #plt.title('stem cell circRNA vs other circRNA')
    ax.set_xlabel('AUC', fontsize=20)
    #plt.xlabel('Number of trees', fontsize=20)
    #ax.set_ylim([0.6, 0.75])
    ax.set_yticks(ind)
    ax.set_yticklabels(xlabel )
    #plt.margins(0.1)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0],  rects6[0]), ('region type', 'clip-cobinding', 'structure', 'motif', 'CNN sequence', 'iDeep'), 
              loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True)
    plt.tight_layout()
    
    plt.show()

def plot_figure():
    results = read_result_file('result/roc')
    
    Figure = plt.figure(figsize=(12, 15))
    #print "*******************************"
    #print results
    for key, values in results.iteritems():  
        rna = key
        #pdb.set_trace()
        labels = [int(float(val)) for val in values[0]]
        probability = [float(val) for val in values[1]]
        plot_roc_curve(labels, probability, rna)
    #plot_roc_curve(labels[1], probability[1], '')
    #plot_roc_curve(labels[2], probability[2], '')
    
    #title_type = 'stem cell circRNAs vs other circRNAs'
    title_type = 'ROC'
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(title_type)
    plt.legend(loc="lower right")
    plt.savefig('roc1.eps', format='eps') 
    #plt.show() 

def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )
        
def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
    
def plothistory(history):
    if 'val_acc' in history.history.keys():
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #plt.axis([800, 1000, 0, 1])
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.axis([800, 1000, 0, 1])
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.title('Train acc/loss')
        plt.ylabel('acc/loss')
        plt.xlabel('epoch')
        plt.legend(['acc', 'loss'], loc='upper left')
        plt.show()
        
        
def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1        
                
                
    if (tp+fn) == 0:
        q9 = float(tn-fp)/(tn+fp + 1e-06)
    if (tn+fp) == 0:
        q9 = float(tp-fn)/(tp+fn + 1e-06)
    if  (tp+fn) != 0 and (tn+fp) !=0:
        q9 = 1- float(np.sqrt(2))*np.sqrt(float(fn*fn)/((tp+fn)*(tp+fn))+float(fp*fp)/((tn+fp)*(tn+fp)))
        
    Q9 = (float)(1+q9)/2
    accuracy = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    recall = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    ppv = float(tp)/(tp + fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    f1_score = float(2*tp)/(2*tp + fp + fn + 1e-06)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return tp,fp,tn,fn,accuracy, precision, sensitivity, recall, specificity, MCC, f1_score,Q9, ppv,npv
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def draw_roc(y_test, y_score):    
    # Compute ROC curve and ROC area for each class
    n_classes=y_score.shape[-1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num=0
    if n_classes<=1:
        fpr[0], tpr[0], _ = roc_curve(y_test[:,], y_score[:,])
        roc_auc[0] = auc(fpr[0], tpr[0])
        num=0
    else:    
        for i in range(n_classes):            
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            num=n_classes-1
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 10))
    
    #line-width
    lw = 2
    auc_score=roc_auc[num]*100
    plt.plot(fpr[num], tpr[num], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f%%)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def draw_pr(y_test, y_score):    
    # Compute ROC curve and ROC area for each class
    n_classes=y_score.shape[-1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    num=0
    if n_classes<=1:        
        precision[0], recall[0], _ = precision_recall_curve(y_test[:, ],y_score[:,])
        average_precision[0] = average_precision_score(y_test[:, ], y_score[:, ])
        num=0
    else:    
        for i in range(n_classes):           
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
            num=n_classes-1
    
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 10))
    
    #line-width
    lw = 2
    pr_score=average_precision[num]*100
    plt.plot(recall[i], precision[i], color='darkorange', lw=lw,
             label='Precision-recall curve (area = %0.2f%%)' % pr_score)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_embedding(X, y,title=None):
    #将数据归一化到0-1之间
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    df = pd.DataFrame(dict(x=X[:,0],y=X[:,1], label=y))
    groups = df.groupby('label')
    
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for name, group in groups:
        plt.scatter(group.x, group.y,c=plt.cm.Set1(name / 10.),label=name)
        #    plt.text(X[i, 0], X[i, 1], '.',
        #         color=plt.cm.Set1(labels[i] / 10.),
        #         fontdict={'weight': 'bold', 'size': 10})
    plt.xticks([]), plt.yticks([])
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
    
