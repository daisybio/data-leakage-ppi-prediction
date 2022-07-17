from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class FC(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        dense1 = layers.Dense(1000, activation='relu')
        dropout1 = layers.Dropout(0.2)
        normalization1 = layers.BatchNormalization()
        dense2 = layers.Dense(1000, activation='relu')
        dropout2 = layers.Dropout(0.5)
        normalization2 = layers.BatchNormalization()
        dense3 = layers.Dense(1000, activation='relu')
        dropout3 = layers.Dropout(0.5)
        normalization3 = layers.BatchNormalization()
        dense4 = layers.Dense(1, activation='relu')
        dropout4 = layers.Dropout(0.5)
        normalization4 = layers.BatchNormalization()
        
        input1 = Input(shape=(1166,20,), dtype=np.float32, name='protein1')
        protein1 = dense1(input1)
        protein1 = dropout1(protein1)
        protein1 = normalization1(protein1)
        protein1 = dense2(protein1)
        protein1 = dropout2(protein1)
        protein1 = normalization2(protein1)
        protein1 = dense3(protein1)
        protein1 = dropout3(protein1)
        protein1 = normalization3(protein1)
        protein1 = dense4(protein1)
        protein1 = dropout4(protein1)
        protein1 = normalization4(protein1)
        
        input2 = Input(shape=(1166,20,), dtype=np.float32, name='protein2')
        protein2 = dense1(input2)
        protein2 = dropout1(protein2)
        protein2 = normalization1(protein2)
        protein2 = dense2(protein2)
        protein2 = dropout2(protein2)
        protein2 = normalization2(protein2)
        protein2 = dense3(protein2)
        protein2 = dropout3(protein2)
        protein2 = normalization3(protein2)
        protein2 = dense4(protein2)
        protein2 = dropout4(protein2)
        protein2 = normalization4(protein2)
        
        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Flatten()(head)
        head = layers.Dense(250, activation='relu')(head)
        head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(50, activation='relu')(head)
        head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(10, activation='relu')(head)
        head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        head = layers.Dropout(0.2)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
