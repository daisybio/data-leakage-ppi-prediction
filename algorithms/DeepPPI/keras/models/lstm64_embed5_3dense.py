from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LSTM64_Embed5_3Dense(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        input1 = Input(shape=(None,), dtype=np.float32, name='protein1')
        protein1 = layers.Embedding(21, 5, embeddings_initializer='glorot_uniform', mask_zero=True)(input1)
        protein1 = layers.LSTM(64)(protein1)

        input2 = Input(shape=(None,), dtype=np.float32, name='protein2')
        protein2 = layers.Embedding(21, 5, embeddings_initializer='glorot_uniform', mask_zero=True)(input2)
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
