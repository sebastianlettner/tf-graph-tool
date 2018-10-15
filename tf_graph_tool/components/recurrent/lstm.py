""" Wrapper for tensorflow lstm cell. """

import numpy as np
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.recurrent.binary_lstm.binary_lstm import BinaryLSTM
from tf_graph_tool.components.recurrent.binary_lstm.binary_stochastic_neuron.bsn_literals import *
from tf_graph_tool.components.component_literals import HIDDEN, OUTPUT


class LSTMCellConfig(RootConfig):

    """ Configuration for Lstm cells. """

    def __init__(self,
                 builder,
                 name,
                 scope,
                 num_units,
                 c_name,
                 h_name,
                 type,
                 seq_len_name,
                 tensorboard_verbosity=0,
                 binary=False,
                 stochastic_method=ROUND,
                 preprocessing_method=PASS_THROUGH_SIGMOID,
                 slope_tenor=tf.constant(1.0, name='slope_tensor'),
                 ):

        """

        Initializes object.

        Args:
            name(str): Inner scope of the variables .
            scope(str): Outer scope of the variables.
            num_units(int): Number of hidden units.
            c_name(str): Name of the placeholder component used for the cell state.
            h_name(str): Name of the placeholder component used for the hidden state.
            seq_len_name(str): Name of the placeholder for the sequence length.
            type(str): Defines the role of the component in the graph. Can be 'output' or 'hidden'.
            tensorboard_verbosity(int):
                0: No tensorboard.
                1: Only the activations are tracked.
                2: Activations and weights are tracked. This means the output operation and all the variables.
            binary(bool): Pass the states of the cell through a binary layer.
            Optional parameters if binary is true
                stochastic_method(str): Method for sampling input values after preprocessing.
                preprocessing_method(str): Method for mapping the input values to (0, 1).
                slope_tenor(constant tensor): Value for slope annealing trick.

        """

        self._name = name
        self._scope = scope
        self._num_units = num_units
        self._c_name = c_name
        self._h_name = h_name
        self._seq_len_name = seq_len_name
        self._binary = binary
        self._tensorboard_verbosity = tensorboard_verbosity
        if binary:
            self._stochastic_method = stochastic_method
            self._preprocessing_method = preprocessing_method
            self._slope_tensor = slope_tenor

        super(LSTMCellConfig, self).__init__(
            factory_request='lstm',
            type=type,
            builder=builder
        )
        super(LSTMCellConfig, self).add_node_to_nx_graph(self)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def num_units(self):
        return self._num_units

    @property
    def tensorboard_verbosity(self):
        return self._tensorboard_verbosity

    @property
    def binary(self):
        return self._binary

    @property
    def stochastic_method(self):
        return self._stochastic_method

    @property
    def preprocessing_method(self):
        return self._preprocessing_method

    @property
    def slope_tensor(self):
        return self._slope_tensor

    @property
    def c_name(self):
        return self._c_name

    @property
    def h_name(self):
        return self._h_name

    @property
    def seq_len_name(self):
        return self._seq_len_name


class LSTMCell(BaseComponent):

    """
        Implements a LSTM cell form tensorflow and wraps it to a component.
        Besides the output to the cell an additional op for running the state will be store in the neural networks
        dictionary. Key: name + '_state_out'. Also the sequence length placeholder is assumed share if there are
        multiple lstm's in the network.
    """

    def __init__(self, config, inputs):

        """

        Initializes object.

        Args:
            config(LSTMCellConfig): Configuration
            inputs(list): List of input tensor.
        """

        self._inputs = inputs
        self._config = config

        self._state_out, output, self._seq_len = self.lstm_cell

        super(LSTMCell, self).__init__(
            name=self._config.name,
            output=output,
            type=self._config.type,
            scope=self._config.scope,
            )

        self.interface = LSTMInterface(self)
        self.config.builder.compute_graph.recurrent_cells.append(self)

    @graph_component
    def lstm_cell(self):

        """

        Returns:

        """

        with tf.variable_scope(self.config.scope):

            if self.config.binary:
                lstm_cell = BinaryLSTM(
                    state_size=self.config.num_units,
                    stochastic_method=self.config.stochastic_method,
                    preprocessing_method=self.config.preprocessing_method,
                    slope_tenor=self.config.slope_tensor,
                    tf_graph=self.config.builder.compute_graph.tf_graph
                )
            else:
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_units,
                                                         state_is_tuple=True)

            c_in = self._config.builder.compute_graph.get_component(self.config.c_name)
            h_in = self._config.builder.compute_graph.get_component(self.config.h_name)
            seq_len = self._config.builder.compute_graph.get_component(self._config.seq_len_name)
            # remove the placeholders for the cell states from the input list
            self._inputs.remove(c_in)
            self._inputs.remove(h_in)
            self._inputs.remove(seq_len)
            inputs = tf.concat(self._inputs, axis=1)

            rnn_in = tf.reshape(inputs, tf.stack([-1, seq_len, inputs.shape[1]]))
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=rnn_in,
                initial_state=state_in,
                time_major=False,
                scope=self.config.name
            )
            lstm_c, lstm_h = lstm_state
            state_out = (lstm_c, lstm_h)
            rnn_out = tf.reshape(lstm_outputs, [-1, self.config.num_units])

            # only tensorboard stuff
            if self._config.tensorboard_verbosity == 0:
                return state_out, rnn_out, seq_len

            elif self._config.tensorboard_verbosity == 1:
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                summaries.append(tf.summary.histogram(self._config.name + 'activations', rnn_out))
                return state_out, rnn_out, seq_len

            elif self._config.tensorboard_verbosity == 2:
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                current_scope = self.config.builder.tf_graph.get_name_scope()
                if current_scope == '':
                    scope = self._config.scope + '/' + self._config.name
                else:
                    scope = current_scope + '/' + self._config.scope + '/' + self._config.name
                variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                summaries += [tf.summary.histogram(var.name, var) for var in variables]
                return state_out, rnn_out, seq_len

            else:
                raise ValueError("Unsupported tensorboard verbosity {}. Supported are 0, 1, 2.".format(
                    self._config.tensorboard_verbosity))

    @property
    def config(self):
        return self._config

    @property
    def state_out(self):
        return self._state_out

    def decorate_graph(self, compute_graph):

        """
        Add the output operation to the neural networks dictionary to make access easier.

        Args:
            compute_graph:

        Returns:

        """
        # assuming all lstm's in the network have the same sequence length
        compute_graph.inputs['seq_len'] = self._seq_len
        if self.config.type == HIDDEN:
            # if the component is recurrent add an additional op to compute the cell states
            compute_graph.hidden[self.name + '_state_out'] = self.state_out
            compute_graph.hidden[self.name] = self.output
        elif self.type == OUTPUT:
            # if the component is recurrent add an additional op to compute the cell states
            compute_graph.outputs[self.name + '_state_out'] = self.state_out
            compute_graph.outputs[self.name] = self.output
        else:
            raise ValueError('Unknown component type: {}'.format(self.type))


class LSTMInterface(object):
    """ Interface to an lstm cell. """

    def __init__(self, lstm_cell):

        """
        Initializes object.
        Args:
            lstm_cell(LSTMCell): The cell.
        """
        self._lstm_cell = lstm_cell

    def get_state_names(self):
        """

        Returns:
            names(tuple): Names of the cell states: (c, h)
        """

        return self._lstm_cell.config.c_name, self._lstm_cell.config.h_name

    def get_run_states_name(self):
        """

        Returns:
            op_name(str): Name of the op to run the states.
        """
        return self._lstm_cell.config.name + '_state_out'

    def convert_out_to_in(self, state_tuple):
        """

        Args:
            state_tuple(tuple): The state tuple (c, h)

        Returns:
            output(dict): key are the names of the states and values are the states.
        """
        c_state = state_tuple[0]
        h_state = state_tuple[1]
        return {self.get_state_names()[0]: c_state, self.get_state_names()[1]: h_state}

    def get_name(self):
        """

        Returns:
            name(str): of the cell
        """
        return self._lstm_cell.config.name

    def get_recurrent_init(self):

        """

        Returns:
            initial_states(dict): keys are the names of the cell states and values the initial state values (zeros)
        """
        return {self.get_state_names()[0]: np.zeros((1, self._lstm_cell.config.num_units)),
                self.get_state_names()[1]: np.zeros((1, self._lstm_cell.config.num_units))}
