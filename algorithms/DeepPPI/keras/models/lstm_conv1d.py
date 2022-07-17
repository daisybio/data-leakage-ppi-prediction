from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM_Conv1D(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        conv1 = layers.Conv1D(128,10, activation='relu', input_shape=(1166,20))
        pool = layers.MaxPooling1D(3)
        conv2 = layers.Conv1D(128,20, activation='relu')
        lstm = layers.LSTM(32, dropout=0.1, recurrent_dropout=0.5)

        input1 = Input(shape=(1166,20,), dtype=np.float32, name='protein1')
        protein1 = conv1(input1)
        protein1 = pool(protein1)
        protein1 = conv2(protein1)
        protein1 = lstm(protein1)

        input2 = Input(shape=(1166,20,), dtype=np.float32, name='protein2')
        protein2 = conv1(input2)
        protein2 = pool(protein2)
        protein2 = conv2(protein2)
        protein2 = lstm(protein2)

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
