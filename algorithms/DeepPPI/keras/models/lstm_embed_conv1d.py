from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM_Embed_Conv1D(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        embed = layers.Embedding(21, 5, embeddings_initializer='glorot_uniform', mask_zero=True)
        conv1 = layers.Conv1D(128,10, activation='relu')
        pool = layers.MaxPooling1D(3)
        conv2 = layers.Conv1D(128,20, activation='relu')
        lstm = layers.LSTM(32)

        input1 = Input(shape=(1166,), dtype=np.float32, name='protein1')
        protein1 = embed(input1)
        protein1 = conv1(input1)
        protein1 = pool(input1)
        protein1 = conv2(input1)
        protein1 = lstm(protein1)

        input2 = Input(shape=(1166,), dtype=np.float32, name='protein2')
        protein2 = embed(input2)
        protein2 = conv1(input2)
        protein2 = pool(input2)
        protein2 = conv2(input2)
        protein2 = lstm(protein2)

        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Dense(50, activation='relu')(head)
        head = layers.BatchNormalization()(head)                
        head = layers.Dense(25, activation='relu')(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
