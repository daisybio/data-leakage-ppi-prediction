from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel


# FC flat, no embedding: 24 millions of parameters
# FC flat embedding, 2 body 100-unit layers: 2 millions of parameter
# FC flat embedding, 3 body 100/50/10-unit layers: 240.000 parameters (2018-11-21_03:34)
# FC flat embedding, 3 body 100/50/1-unit layers: 29.000 parameters (2018-11-21_03:58)

class FC_Flat_Embed_D5(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        #no mask
        embed = layers.Embedding(21, 5, embeddings_initializer='glorot_uniform', input_length=1166)
        #embed = layers.Embedding(21, 3, embeddings_initializer='glorot_uniform', mask_zero=True, input_length=1166)
        first_layer = layers.Dense(100, activation='relu')
        #first_dropout = layers.Dropout(0.2)
        first_normalization = layers.BatchNormalization()
        second_layer = layers.Dense(50, activation='relu')
        #second_dropout = layers.Dropout(0.5)
        second_normalization = layers.BatchNormalization()
        third_layer = layers.Dense(1, activation='relu')
        #third_dropout = layers.Dropout(0.5)
        third_normalization = layers.BatchNormalization()
        flat = layers.Flatten()

        input1 = Input(shape=(1166,), dtype=np.float32, name='protein1')
        protein1 = embed(input1)
        #protein1 = flat(protein1)
        protein1 = first_layer(protein1)
        #protein1 = first_dropout(protein1)
        protein1 = first_normalization(protein1)
        protein1 = second_layer(protein1)
        #protein1 = second_dropout(protein1)
        protein1 = second_normalization(protein1)
        protein1 = third_layer(protein1)
        #protein1 = third_dropout(protein1)
        protein1 = third_normalization(protein1)
        
        input2 = Input(shape=(1166,), dtype=np.float32, name='protein2')
        protein2 = embed(input2)
        #protein2 = flat(protein2)
        protein2 = first_layer(protein2)
        #protein2 = first_dropout(protein2)
        protein2 = first_normalization(protein2)
        protein2 = second_layer(protein2)
        #protein2 = second_dropout(protein2)
        protein2 = second_normalization(protein2)
        protein2 = third_layer(protein2)
        #protein2 = third_dropout(protein2)
        protein2 = third_normalization(protein2)
        
        head = layers.concatenate([protein1, protein2], axis=-1)
        head = flat(head)
        # head = layers.Dense(200, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        # head = layers.Dense(200, activation='relu')(head)
        # head = layers.Dropout(0.5)(head)
        # head = layers.BatchNormalization()(head)
        head = layers.Dense(10, activation='relu')(head)
        #head = layers.Dropout(0.5)(head)
        head = layers.BatchNormalization()(head)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        #head = layers.Dropout(0.2)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
