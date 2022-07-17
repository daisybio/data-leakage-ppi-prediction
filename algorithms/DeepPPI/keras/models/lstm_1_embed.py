from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM_1_Embed(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        #embed = layers.Embedding(21, 5, embeddings_initializer='glorot_uniform', mask_zero=True, input_length=1166)
        embed = layers.Embedding(21, 5, embeddings_initializer='glorot_uniform', mask_zero=True)
        lstm = layers.LSTM(64)

        input1 = Input(shape=(1166,), dtype=np.float32, name='protein1')
        protein1 = embed(input1)
        protein1 = lstm(protein1)

        input2 = Input(shape=(1166,), dtype=np.float32, name='protein2')
        protein2 = embed(input2)
        protein2 = lstm(protein2)

        head = layers.concatenate([protein1, protein2], axis=-1)
        #head = layers.Flatten()(head)
        # head = layers.Dense(200, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        # head = layers.Dense(100, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        # head = layers.Dense(50, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        head = layers.Dense(50, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
