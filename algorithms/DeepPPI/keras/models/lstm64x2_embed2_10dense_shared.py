from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM64x2_Embed2_10Dense_S(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        embed = layers.Embedding(21, 2, embeddings_initializer='glorot_uniform', mask_zero=True)
        lstm1 = layers.LSTM(64, return_sequences=True)
        lstm2 = layers.LSTM(64)
        
        input1 = Input(shape=(None,), dtype=np.float32, name='protein1')
        protein1 = embed(input1)
        protein1 = lstm1(protein1)
        protein1 = lstm2(protein1)

        input2 = Input(shape=(None,), dtype=np.float32, name='protein2')
        protein2 = embed(input2)
        protein2 = lstm1(protein2)
        protein2 = lstm2(protein2)

        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Dense(100, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(100, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(50, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(50, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(50, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)

        model = Model(inputs=[input1, input2], outputs=output)
        return model
