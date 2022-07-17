from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class Conv3_3_2Dense_S(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        conv1 = layers.Conv1D(5, 20, activation='relu')
        pool1 = layers.MaxPooling1D(3)
        batchnorm1 = layers.BatchNormalization()
        conv2 = layers.Conv1D(5, 20, activation='relu')
        pool2 = layers.MaxPooling1D(3)
        batchnorm2 = layers.BatchNormalization()
        conv3 = layers.Conv1D(5, 20, activation='relu')
        pool3 = layers.MaxPooling1D(3)
        batchnorm3 = layers.BatchNormalization()
              
        input1 = Input(shape=(1166,24,), dtype=np.float32, name='protein1')
        protein1 = conv1(input1)
        protein1 = pool1(protein1)
        protein1 = batchnorm1(protein1)
        protein1 = conv2(protein1)
        protein1 = pool2(protein1)
        protein1 = batchnorm2(protein1)
        protein1 = conv3(protein1)
        protein1 = pool3(protein1)
        protein1 = batchnorm3(protein1)
      
        input2 = Input(shape=(1166,24,), dtype=np.float32, name='protein2')
        protein2 = conv1(input2)
        protein2 = pool1(protein2)
        protein2 = batchnorm1(protein2)
        protein2 = conv2(protein2)
        protein2 = pool2(protein2)
        protein2 = batchnorm2(protein2)
        protein2 = conv3(protein2)
        protein2 = pool3(protein2)
        protein2 = batchnorm3(protein2)
      
        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Flatten()(head)
        # head = layers.Dense(50, activation='relu')(head)
        # head = layers.BatchNormalization()(head)                
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)

        model = Model(inputs=[input1, input2], outputs=output)
        return model
