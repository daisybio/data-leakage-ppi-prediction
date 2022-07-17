from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        lstm = layers.LSTM(128)

        input1 = Input(shape=(1166,20,), dtype=np.float32, name='protein1')
        protein1 = lstm(input1)

        input2 = Input(shape=(1166,20,), dtype=np.float32, name='protein2')
        protein2 = lstm(input2)

        head = layers.concatenate([protein1, protein2], axis=-1)
        #head = layers.Flatten()(head)
        head = layers.Dense(200, activation='relu')(head)
        head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(100, activation='relu')(head)
        head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        head = layers.Dropout(0.2)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
