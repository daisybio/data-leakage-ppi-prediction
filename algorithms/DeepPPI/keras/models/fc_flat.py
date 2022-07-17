from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class FC_Flat(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        flat = layers.Flatten()
        first_layer = layers.Dense(1000, activation='relu')
        #first_layer = layers.Dense(1000, activation='relu')
        first_dropout = layers.Dropout(0.2)
        first_normalization = layers.BatchNormalization()
        second_layer = layers.Dense(1000, activation='relu')
        second_dropout = layers.Dropout(0.5)
        second_normalization = layers.BatchNormalization()
        # third_layer = layers.Dense(1000, activation='relu')
        # third_dropout = layers.Dropout(0.5)
        # third_normalization = layers.BatchNormalization()
        
        input1 = Input(shape=(1166,20,), dtype=np.float32, name='protein1')
        protein1 = flat(input1)
        protein1 = first_layer(protein1)
        protein1 = first_dropout(protein1)
        protein1 = first_normalization(protein1)
        protein1 = second_layer(protein1)
        protein1 = second_dropout(protein1)
        protein1 = second_normalization(protein1)
        # protein1 = third_layer(protein1)
        # protein1 = third_dropout(protein1)
        # protein1 = third_normalization(protein1)
        
        input2 = Input(shape=(1166,20,), dtype=np.float32, name='protein2')
        protein2 = flat(input2)
        protein2 = first_layer(protein2)
        protein2 = first_dropout(protein2)
        protein2 = first_normalization(protein2)
        protein2 = second_layer(protein2)
        protein2 = second_dropout(protein2)
        protein2 = second_normalization(protein2)
        # protein2 = third_layer(protein2)
        # protein2 = third_dropout(protein2)
        # protein2 = third_normalization(protein2)
        
        head = layers.concatenate([protein1, protein2], axis=-1)
        # head = layers.Dense(200, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        # head = layers.Dense(200, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        head = layers.Dense(100, activation='relu')(head)
        head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        head = layers.Dropout(0.2)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
