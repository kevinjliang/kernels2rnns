import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class LSTMCell_mod(LayerRNNCell):
  def __init__(self, num_units, gate_mod=None, ngram=False,
               no_feedback=False, use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True, layer_norm=False,
               activation=None, reuse=None, name=None, dtype=None, **kwargs):

    super(LSTMCell_mod, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)

    print("LSTM cell mode: {0}".format(gate_mod))

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._gate_mod = gate_mod
    self._ngram = ngram
    self._no_feedback = no_feedback
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._layer_norm = layer_norm
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    if self._ngram:
      self._kernel = self.add_variable(
          _WEIGHTS_VARIABLE_NAME,
          shape=[h_depth, 4 * self._num_units],
          initializer=self._initializer,
          partitioner=maybe_partitioner)
    else:
      self._kernel = self.add_variable(
          _WEIGHTS_VARIABLE_NAME,
          shape=[input_depth + h_depth, 4 * self._num_units],
          initializer=self._initializer,
          partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      
    if self._gate_mod in ["gated_linear", "linear"]:
      self._sigma2_f = self.add_variable("sigma2_f", shape=[self._num_units],
                                         initializer=tf.zeros_initializer)
      self._sigma2_f = tf.sigmoid(self._sigma2_f)
      
      self._sigma2_i = self.add_variable("sigma2_i", shape=[],
                                         initializer=tf.zeros_initializer)
      self._sigma2_i = tf.sigmoid(self._sigma2_i)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, must be 2-D, `[batch, input_size]`.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
      
    # No feedback, if desired; also, gcnn/cnn do not have feedback
    if self._no_feedback or self._gate_mod in ["gcnn", "cnn"]:
        m_prev = tf.zeros(m_prev.shape)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    if self._ngram:
      lstm_matrix = inputs + math_ops.matmul(m_prev, self._kernel)
    else:
      lstm_matrix = math_ops.matmul(
          array_ops.concat([inputs, m_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    elif self._gate_mod == "lstm":
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))
    elif self._gate_mod == "rkm_lstm":
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * j)
    elif self._gate_mod == "rkm_cifg":
      c = (sigmoid(f + self._forget_bias) * c_prev + (1 - sigmoid(f + self._forget_bias)) *j)
    elif self._gate_mod in ["gated_linear", "linear"]:
#      sigma2_f = 0.5
#      sigma2_i = 0.5
#      c = (sigma2_f * c_prev + sigma2_i * j)
      c = (self._sigma2_f * c_prev + self._sigma2_i * j)  
    elif self._gate_mod in ["gcnn", "cnn"]:
      sigma2_i = 1
      c = sigma2_i * j
    else:
      raise NotImplementedError("Invalid gate_mod: {0}".format(self._gate_mod))
      
    if self._layer_norm:
      c = tf.contrib.layers.layer_norm(c)
    
    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
      
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    elif self._gate_mod == "lstm":
      m = sigmoid(o) * self._activation(c)
    elif self._gate_mod in ["rkm_lstm", "rkm_cifg", "gated_linear", "gcnn"]:
      m = sigmoid(o) * c
    elif self._gate_mod in ["linear", "cnn"]:
      m = self._activation(c)
    else:
      raise NotImplementedError("Invalid gate_mod: {0}".format(self._gate_mod))  
      
    if self._num_proj is not None:
      m = math_ops.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(LSTMCell_mod, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

