import tensorflow as tf
import numpy as np



class GroupLinearLayer(tf.keras.layers.Layer):

    def __init__(self, units, nRIM):
        super(GroupLinearLayer, self).__init__()
        self.units = units
        self.nRIM = nRIM

    def build(self, input_shape):
        # input_shape = (batch, [time,] nRIM, din)
        self.w = self.add_weight(name='group_linear_layer',
                                 shape=(self.nRIM, int(input_shape[-1]), self.units),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        params = self.w
        out = tf.transpose(tf.matmul(tf.transpose(inputs, [1, 0, 2]), params), [1, 0, 2])
        return out