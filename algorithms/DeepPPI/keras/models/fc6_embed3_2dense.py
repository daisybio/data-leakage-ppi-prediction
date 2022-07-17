from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class FC6_Embed3_2Dense(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        ## Protein 1
        input1 = Input(shape=(1166,), dtype=np.float32, name='protein1')
        #no mask
        protein1 = layers.Embedding(21, 3, embeddings_initializer='glorot_uniform')(input1)

        protein1 = layers.Dense(50, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)

        protein1 = layers.Dense(50, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)

        protein1 = layers.Dense(50, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)

        protein1 = layers.Dense(50, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)

        protein1 = layers.Dense(50, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)

        protein1 = layers.Dense(1, activation='relu')(protein1)
        protein1 = layers.BatchNormalization()(protein1)


        ## Protein 2
        input2 = Input(shape=(1166,), dtype=np.float32, name='protein2')
        #no mask
        protein2 = layers.Embedding(21, 3, embeddings_initializer='glorot_uniform')(input2)

        protein2 = layers.Dense(50, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)

        protein2 = layers.Dense(50, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)

        protein2 = layers.Dense(50, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)

        protein2 = layers.Dense(50, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)

        protein2 = layers.Dense(50, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)

        protein2 = layers.Dense(1, activation='relu')(protein2)
        protein2 = layers.BatchNormalization()(protein2)

        ## Head
        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Flatten()(head)
        #head = layers.Reshape((-1,))(head)
        
        head = layers.Dense(10, activation='relu')(head)
        head = layers.BatchNormalization()(head)

        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
