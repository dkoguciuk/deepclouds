#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import tensorflow as tf

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell, LSTMStateTuple

from tensorflow.python.layers import base as base_layer

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class MyLSTMCell(_LayerRNNCell):
    """
    TODO (@dkoguciuk): Description.

    Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_out=None, proj_clip=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None):
        """Initialize the parameters for an LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            use_peepholes: bool, set True to enable diagonal/peephole connections.
            cell_clip: (optional) A float value, if provided the cell state is clipped
                by this value prior to the cell output activation.
            initializer: (optional) The initializer to use for the weight matrices.
            num_out: (optional) int, The output dimensionality for the cell.
                If None, output would be size of num_units.
            proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
                provided, then the projected values are clipped elementwise to within
                `[-proj_clip, proj_clip]`.
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of
                the training. Must set it manually to `0.0` when restoring from
                CudnnLSTM trained checkpoints.
            state_is_tuple: If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.  If False, they are concatenated
                along the column axis.  This latter behavior will soon be deprecated.
            activation: Activation function of the inner states.  Default: `tanh`.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.  If not `True`, and the existing scope already has
                the given variables, an error is raised.
            name: String, the name of the layer. Layers with the same name will
                share weights, but to avoid mistakes we require reuse=True in such
                cases.
            When restoring from CudnnLSTM-trained checkpoints, use
                `CudnnCompatibleLSTMCell` instead.
        """
        super(MyLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_out = num_out
        self._proj_clip = proj_clip
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

        if num_out:
            self._state_size = (
                LSTMStateTuple(num_out, num_units) if state_is_tuple else num_out + num_units)
            self._output_size = num_out
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units) if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        hidden_depth = self._num_units if self._num_out is None else self._num_out
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 4 * hidden_depth],
            initializer=self._initializer)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * hidden_depth],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self.h_depth],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self.h_depth],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self.h_depth],
                                               initializer=self._initializer)
    
        self.built = True

    def call(self, inputs, state):
        """Run one step of LSTM.
        Args:
            inputs: input Tensor, 2D, `[batch, num_units].
            state: if `state_is_tuple` is False, this must be a state Tensor,
                `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
                tuple of state Tensors, both `2-D`, with column sizes `c_state` and
                `m_state`.
        Returns:
            A tuple containing:
                - A `2-D, [batch, output_dim]`, Tensor representing the output of the
                    LSTM after reading `inputs` when previous state was `state`.
                    Here output_dim is:
                        num_out if num_out was set,
                        num_units otherwise.
                - Tensor(s) representing the new state of LSTM after reading `inputs` when
                    the previous state was `state`.  Same type and shape(s) as `state`.
        Raises:
            ValueError: If input size cannot be inferred from inputs via
                static shape inference.
        """
        num_out = self._num_units if self._num_out is None else self._num_out
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_out])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = math_ops.matmul(array_ops.concat([inputs, m_prev], 1), self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=1)

        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state
