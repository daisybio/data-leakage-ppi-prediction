from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM64x2_3Conv_3Dense(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        input1 = Input(shape=(None,20,), dtype=np.float32, name='protein1')
        protein1 = layers.Conv1D(10, 20, activation='relu')(input1)
        protein1 = layers.MaxPooling1D(3)(protein1)
        protein1 = layers.BatchNormalization()(protein1)
        protein1 = layers.Conv1D(10, 20, activation='relu')(protein1)
        protein1 = layers.MaxPooling1D(3)(protein1)
        protein1 = layers.BatchNormalization()(protein1)
        protein1 = layers.Conv1D(10, 20, activation='relu')(protein1)
        protein1 = layers.MaxPooling1D(3)(protein1)
        protein1 = layers.BatchNormalization()(protein1)
        protein1 = layers.LSTM(64, return_sequences=True)(protein1)
        protein1 = layers.LSTM(64)(protein1)

        input2 = Input(shape=(None,20,), dtype=np.float32, name='protein2')
        protein2 = layers.Conv1D(10, 20, activation='relu')(input2)
        protein2 = layers.MaxPooling1D(3)(protein2)
        protein2 = layers.BatchNormalization()(protein2)
        protein2 = layers.Conv1D(10, 20, activation='relu')(protein2)
        protein2 = layers.MaxPooling1D(3)(protein2)
        protein2 = layers.BatchNormalization()(protein2)
        protein2 = layers.Conv1D(10, 20, activation='relu')(protein2)
        protein2 = layers.MaxPooling1D(3)(protein2)
        protein2 = layers.BatchNormalization()(protein2)
        protein2 = layers.LSTM(64, return_sequences=True)(protein2)
        protein2 = layers.LSTM(64)(protein2)

        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Dense(50, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)

        model = Model(inputs=[input1, input2], outputs=output)
        return model
