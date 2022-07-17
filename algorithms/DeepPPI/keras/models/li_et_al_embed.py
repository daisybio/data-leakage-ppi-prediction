from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import tensorflow as tf 

from models.abstract_model import AbstractModel

class LiEtAl(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        input1 = Input(shape=(1166,), dtype=np.float32, name='protein1')
        protein1 = layers.Embedding(21, 128, embeddings_initializer='glorot_uniform')(input1)
        #print("After embedding: {}".format(protein1._keras_shape))
        protein1 = layers.Reshape((1166, 128, 1))(protein1)
        #print("After reshape: {}".format(protein1._keras_shape))
        protein1 = layers.Conv2D(10, (10, 65), data_format='channels_last', activation='relu', kernel_initializer='glorot_uniform')(protein1)
        #print("After conv2d1:{}".format(protein1._keras_shape))
        protein1 = layers.MaxPooling2D((2, 1))(protein1)
        #print("After maxpool1:{}".format(protein1._keras_shape))
        protein1 = layers.Conv2D(10, (8, 1), activation='relu', kernel_initializer='glorot_uniform')(protein1)
        #print("After conv2d2:{}".format(protein1._keras_shape))
        protein1 = layers.MaxPooling2D((2, 1))(protein1)
        #print("After maxpool2:{}".format(protein1._keras_shape))
        protein1 = layers.Conv2D(1, (5, 1), activation='relu', kernel_initializer='glorot_uniform')(protein1)
        #print("After conv2d3:{}".format(protein1._keras_shape))
        protein1 = layers.MaxPooling2D((2, 1))(protein1)
        #print("After maxpool3:{}".format(protein1._keras_shape))
        protein1 = layers.Reshape((140, 64))(protein1)
        #print("After reshape:{}".format(protein1._keras_shape))
        protein1 = layers.LSTM(80)(protein1)
        #print("After LSTM:{}".format(protein1._keras_shape))

        input2 = Input(shape=(1166,), dtype=np.float32, name='protein2')
        protein2 = layers.Embedding(21, 128, embeddings_initializer='glorot_uniform')(input2)
        protein2 = layers.Reshape((1166, 128, 1))(protein2)
        protein2 = layers.Conv2D(10, (10, 65), data_format='channels_last', activation='relu', kernel_initializer='glorot_uniform')(protein2)
        protein2 = layers.MaxPooling2D((2, 1))(protein2)
        protein2 = layers.Conv2D(10, (8, 1), activation='relu', kernel_initializer='glorot_uniform')(protein2)
        protein2 = layers.MaxPooling2D((2, 1))(protein2)
        protein2 = layers.Conv2D(1, (5, 1), activation='relu', kernel_initializer='glorot_uniform')(protein2)
        protein2 = layers.MaxPooling2D((2, 1))(protein2)
        protein2 = layers.Reshape((140, 64))(protein2)
        protein2 = layers.LSTM(80)(protein2)

        head = layers.concatenate([protein1, protein2], axis=-1)
        head = layers.Dense(1)(head)
        output = layers.Activation(tf.nn.sigmoid)(head)
        
        model = Model(inputs=[input1, input2], outputs=output)
        return model
