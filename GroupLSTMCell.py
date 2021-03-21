import tensorflow as tf

import numpy as np

class GroupLSTMCell(tf.keras.layers.Layer):

    def __init__(self, units, nRIM):
        super(GroupLSTMCell, self).__init__()
        self.units = units
        self.nRIM = nRIM

    @property
    def state_size(self):
        return (tf.TensorShape([self.nRIM, self.units]), tf.TensorShape([self.nRIM, self.units]))

    def build(self, input_shape):
        self.i2h_param = self.add_weight(name='group_lstm_i2h',
                                         shape=(self.nRIM, int(input_shape[-1]), self.units * 4),
                                         initializer='uniform',
                                         trainable=True)
        self.h2h_param = self.add_weight(name='group_lstm_h2h',
                                         shape=(self.nRIM, self.units, self.units * 4),
                                         initializer='uniform',
                                         trainable=True)

    def call(self, inputs, states):
        # inputs in shape [batch, nRIM, din]
        h, c = states
        preact_i = tf.transpose(tf.matmul(tf.transpose(inputs, [1, 0, 2]), self.i2h_param), [1, 0, 2])
        preact_h = tf.transpose(tf.matmul(tf.transpose(h, [1, 0, 2]), self.h2h_param), [1, 0, 2])
        preact = preact_i + preact_h

        new_cell = tf.tanh(preact[:, :, :self.units])
        gates = tf.sigmoid(preact[:, :, self.units:])
        input_gate = gates[:, :, :self.units]
        forget_gate = gates[:, :, self.units:(self.units * 2)]
        output_gate = gates[:, :, (self.units * 2):]

        c_t = tf.multiply(c, forget_gate) + tf.multiply(input_gate, new_cell)
        h_t = tf.multiply(output_gate, tf.tanh(c_t))
        return h_t, (h_t, c_t)