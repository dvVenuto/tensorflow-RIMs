import tensorflow as tf
import numpy as np
from tf_agents.networks import GroupLinearLayer
from tf_agents.networks import GroupLSTMCell
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

class RIMCell(tf.keras.layers.Layer):

    def __init__(self, units, nRIM, k,
                 num_input_heads, input_key_size, input_value_size, input_query_size, input_keep_prob,
                 num_comm_heads, comm_key_size, comm_value_size, comm_query_size, comm_keep_prob):
        super(RIMCell, self).__init__()
        self.units = units
        self.nRIM = nRIM
        self.k = k

        self.recurrent_activation = 'hard_sigmoid'
        self.use_bias = True
        self.kernel_initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        self.bias_initializer = 'zeros'
        self.unit_forget_bias = True
        self.kernel_regularizer = None
        self.recurrent_regularizer = None
        self.bias_regularizer = None
        self.kernel_constraint = None
        self.recurrent_constraint = None
        self.bias_constraint = None
        self.dropout = 0.
        self.recurrent_dropout = 0.

        self.num_input_heads = num_input_heads
        self.input_key_size = input_key_size
        self.input_value_size = input_value_size
        self.input_query_size = input_query_size
        self.input_keep_prob = input_keep_prob

        self.num_comm_heads = num_comm_heads
        self.comm_key_size = comm_key_size
        self.comm_value_size = comm_value_size
        self.comm_query_size = comm_query_size
        self.comm_keep_prob = comm_keep_prob

        self.activation="RIM"
        self.output_size = self.units

        assert input_key_size == input_query_size, 'input_key_size == input_query_size required'
        assert comm_key_size == comm_query_size, 'comm_key_size == comm_query_size required'

    @property
    def state_size(self):
        return (tf.TensorShape([self.nRIM, self.units]), tf.TensorShape([self.nRIM, self.units]))

    def build(self, input_shape):
        self.key = tf.keras.layers.Dense(units=self.num_input_heads * self.input_key_size, activation=None,
                                         use_bias=True)
        self.value = tf.keras.layers.Dense(units=self.num_input_heads * self.input_value_size, activation=None,
                                           use_bias=True)
        self.query = GroupLinearLayer.GroupLinearLayer(units=self.num_input_heads * self.input_query_size, nRIM=self.nRIM)
        self.input_attention_dropout = tf.keras.layers.Dropout(rate=1 - self.input_keep_prob)

        self.rnn_cell = GroupLSTMCell.GroupLSTMCell(units=self.units, nRIM=self.nRIM)

        self.key_ = GroupLinearLayer.GroupLinearLayer(units=self.num_comm_heads * self.comm_key_size, nRIM=self.nRIM)
        self.value_ = GroupLinearLayer.GroupLinearLayer(units=self.num_comm_heads * self.comm_value_size, nRIM=self.nRIM)
        self.query_ = GroupLinearLayer.GroupLinearLayer(units=self.num_comm_heads * self.comm_query_size, nRIM=self.nRIM)
        self.comm_attention_dropout = tf.keras.layers.Dropout(rate=1 - self.comm_keep_prob)
        self.comm_attention_output = GroupLinearLayer.GroupLinearLayer(units=self.units, nRIM=self.nRIM)


        self.built = True


    def call(self, inputs, states, training=False):
        # inputs of shape (batch_size, input_feature_size)

        # inputs of shape (batch_size, input_feature_size)
        hs, cs = states
        rnn_inputs, mask = self.input_attention_mask(inputs, hs, training=training)

        h_old = hs * 1.0
        c_old = cs * 1.0

        _, (h_rnnout, c_rnnout) = self.rnn_cell(rnn_inputs, (hs, cs))

        h_new = tf.stop_gradient(h_rnnout * (1 - mask)) + h_rnnout * mask

        h_comm = self.comm_attention(h_new, mask, training=training)

        h_update = h_comm * mask + h_old * (1 - mask)
        c_update = c_rnnout * mask + c_old * (1 - mask)

        out_state = tf.reshape(h_update, [tf.shape(inputs)[0], self.units * self.nRIM])

        #out_h = (h_update, c_update)


        out_h = [h_update, c_update]

        print(out_state.shape)

        print(h_update.shape)
        print(c_update.shape)
        #print(out_h.shape)

        return out_state, out_h

    def input_attention_mask(self, x, hs, training=False):
        # x of shape (batch_size, input_feature_size)
        # hs of shape (batch_size, nRIM, hidden_size = units)

        xx = tf.stack([x, tf.zeros_like(x)], axis=1)

        key_layer = self.key(xx)
        value_layer = self.value(xx)
        query_layer = self.query(hs)

        key_layer1 = tf.stack(tf.split(key_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
        value_layer1 = tf.stack(tf.split(value_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
        query_layer1 = tf.stack(tf.split(query_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
        value_layer2 = tf.reduce_mean(value_layer1, axis=1)

        attention_scores1 = tf.matmul(query_layer1, key_layer1, transpose_b=True) / np.sqrt(self.input_key_size)
        attention_scores2 = tf.reduce_mean(attention_scores1, axis=1)

        signal_attention = attention_scores2[:, :, 0]
        topk = tf.math.top_k(signal_attention, self.k)
        indices = topk.indices
        mesh = tf.meshgrid(tf.range(indices.shape[1]), tf.range(tf.shape(indices)[0]))[1]
        full_indices = tf.reshape(tf.stack([mesh, indices], axis=-1), [-1, 2])

        sparse_tensor = tf.sparse.SparseTensor(indices=tf.cast(full_indices, tf.int64),
                                               values=tf.ones(tf.shape(full_indices)[0]),
                                               dense_shape=[tf.shape(x)[0], self.nRIM])
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        mask_ = tf.sparse.to_dense(sparse_tensor)
        mask = tf.expand_dims(mask_, axis=-1)

        attention_prob = self.input_attention_dropout(tf.nn.softmax(attention_scores2, axis=-1), training=training)
        inputs = tf.matmul(attention_prob, value_layer2)
        inputs1 = inputs * mask
        return inputs1, mask

    def comm_attention(self, h_new, mask, training=False):
        # h_new of shape (batch_size, nRIM, hidden_size = units)
        # mask of shape (batch_size, nRIM, 1)

        comm_key_layer = self.key_(h_new)
        comm_value_layer = self.value_(h_new)
        comm_query_layer = self.query_(h_new)

        comm_key_layer1 = tf.stack(tf.split(comm_key_layer, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)
        comm_value_layer1 = tf.stack(tf.split(comm_value_layer, num_or_size_splits=self.num_comm_heads, axis=-1),
                                     axis=1)
        comm_query_layer1 = tf.stack(tf.split(comm_query_layer, num_or_size_splits=self.num_comm_heads, axis=-1),
                                     axis=1)

        comm_attention_scores = tf.matmul(comm_query_layer1, comm_key_layer1, transpose_b=True) / np.sqrt(
            self.comm_key_size)
        comm_attention_probs = tf.nn.softmax(comm_attention_scores, axis=-1)

        comm_mask_ = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_comm_heads, 1, 1])

        comm_attention_probs1 = self.comm_attention_dropout(comm_attention_probs * comm_mask_, training=training)
        context_layer = tf.matmul(comm_attention_probs1, comm_value_layer1)
        context_layer1 = tf.reshape(tf.transpose(context_layer, [0, 2, 1, 3]),
                                    [tf.shape(h_new)[0], self.nRIM, self.num_comm_heads * self.comm_value_size])

        comm_out = self.comm_attention_output(context_layer1) + h_new
        return comm_out

    def get_config(self):
        config = {
            'units':
                self.units * self.nRIM,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(LSTM_cell_test, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))


def _caching_device(rnn_cell):
  """Returns the caching device for the RNN variable.
  This is useful for distributed training, when variable is not located as same
  device as the training worker. By enabling the device cache, this allows
  worker to read the variable once and cache locally, rather than read it every
  time step from remote when it is needed.
  Note that this is assuming the variable that cell needs for each time step is
  having the same value in the forward path, and only gets updated in the
  backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If the
  cell body relies on any variable that gets updated every time step, then
  caching device will cause it to read the stall value.
  Args:
    rnn_cell: the rnn cell instance.
  """
  if context.executing_eagerly():
    # caching_device is not supported in eager mode.
    return None
  if not getattr(rnn_cell, '_enable_caching_device', False):
    return None
  # Don't set a caching device when running in a loop, since it is possible that
  # train steps could be wrapped in a tf.while_loop. In that scenario caching
  # prevents forward computations in loop iterations from re-reading the
  # updated weights.
  if control_flow_util.IsInWhileLoop(ops.get_default_graph()):
    logging.warn('Variable read device caching has been disabled because the '
                 'RNN is in tf.while_loop loop context, which will cause '
                 'reading stalled value in forward path. This could slow down '
                 'the training due to duplicated variable reads. Please '
                 'consider updating your code to remove tf.while_loop if '
                 'possible.')
    return None
  if (rnn_cell._dtype_policy.compute_dtype !=
      rnn_cell._dtype_policy.variable_dtype):
    logging.warn('Variable read device caching has been disabled since it '
                 'doesn\'t work with the mixed precision API. This is '
                 'likely to cause a slowdown for RNN training due to '
                 'duplicated read of variable for each timestep, which '
                 'will be significant in a multi remote worker setting. '
                 'Please consider disabling mixed precision API if '
                 'the performance has been affected.')
    return None
  # Cache the value on the device that access the variable.
  return lambda op: op.device

def _config_for_enable_caching_device(rnn_cell):
  """Return the dict config for RNN cell wrt to enable_caching_device field.
  Since enable_caching_device is a internal implementation detail for speed up
  the RNN variable read when running on the multi remote worker setting, we
  don't want this config to be serialized constantly in the JSON. We will only
  serialize this field when a none default value is used to create the cell.
  Args:
    rnn_cell: the RNN cell for serialize.
  Returns:
    A dict which contains the JSON config for enable_caching_device value or
    empty dict if the enable_caching_device value is same as the default value.
  """
  default_enable_caching_device = ops.executing_eagerly_outside_functions()
  if rnn_cell._enable_caching_device != default_enable_caching_device:
    return {'enable_caching_device': rnn_cell._enable_caching_device}
  return {}

def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.TensorShape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_nested(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)
