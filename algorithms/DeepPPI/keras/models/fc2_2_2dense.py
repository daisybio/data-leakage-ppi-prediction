from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class FC2_2_2Dense(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        input1 = Input(shape=(1166,20,), dtype=np.float32, name='protein1')
        protein1 = layers.Flatten()(input1)
        protein1 = layers.Dense(8, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)
        protein1 = layers.Dense(8, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)
        
        input2 = Input(shape=(1166,20,), dtype=np.float32, name='protein2')
        protein2 = layers.Flatten()(input2)
        protein2 = layers.Dense(8, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)
        protein2 = layers.Dense(8, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)
        
        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Dense(8, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
