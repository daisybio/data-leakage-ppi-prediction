from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class FC_Flat_NoDrop_NoNorm(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        flat = layers.Flatten()
        first_layer = layers.Dense(1000, activation='relu')
        second_layer = layers.Dense(200, activation='relu')
        
        input1 = Input(shape=(1166,20,), dtype=np.float32, name='protein1')
        protein1 = flat(input1)
        protein1 = first_layer(protein1)
        protein1 = second_layer(protein1)
        
        input2 = Input(shape=(1166,20,), dtype=np.float32, name='protein2')
        protein2 = flat(input2)
        protein2 = first_layer(protein2)
        protein2 = second_layer(protein2)
        
        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Dense(200, activation='relu')(head)
        head = layers.Dense(100, activation='relu')(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
