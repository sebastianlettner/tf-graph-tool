import tensorflow as tf
from binary_stochastic_neuron.bsn_builder import BsnBuilder as Bsn_layer
from binary_stochastic_neuron import bsn_literals


class BinaryLSTM(tf.contrib.rnn.BasicLSTMCell):

    """ A basic long short-term memory with binary cell and hidden state. """

    def __init__(self,
                 state_size,
                 stochastic_method,
                 preprocessing_method,
                 slope_tenor,
                 tf_graph,
                 **kwargs):

        """

        Initializes object.

        Args:
            state_size(int): Size of the hidden state and cell state respectively.
            stochastic_method(str): Method for sampling input values after preprocessing.
            preprocessing_method(str): Method for mapping the input values to (0, 1).
            slope_tenor(constant tensor): Value for slope annealing trick.
            **kwargs:

        """
        kwargs['state_is_tuple'] = True  # Force state is tuple.
        self._cell_name = 'binary_lstm'
        self._state_size = state_size
        self._stochastic_method = stochastic_method
        self._preprocessing_method = preprocessing_method
        self._slope_tensor = slope_tenor
        self._tf_graph = tf_graph

        super(BinaryLSTM, self).__init__(state_size, **kwargs)  # create an lstm cell

    def __call__(self, inputs, state):

        """

        Overwriting the call function from the BasicLSTMCell.

        Args:
            inputs: `2-D` tensor with shape `[batch_size, input_size]`.
            state: An `LSTMStateTuple` of state tensors, each shaped
                  `[batch_size, self.state_size]`.

        Returns:
            A pair containing the new hidden state, and the new state a
            `LSTMStateTuple`
        """

        output, next_state = super(BinaryLSTM, self).__call__(inputs, state)
        with tf.variable_scope(self._cell_name):

            binary_cell_state = Bsn_layer.bsn_layer(next_state[0],
                                                    stochastic_method=self.stochastic_method,
                                                    preprocessing_method=self.preprocessing_method,
                                                    tf_graph=self._tf_graph,
                                                    slope_tensor=self._slope_tensor,
                                                    loss_op_name='loss_by_example',
                                                    name='binary_layer',
                                                    stochastic_gradient_estimator=bsn_literals.STRAIGHT_THROUGH)
            binary_hidden_state = Bsn_layer.bsn_layer(next_state[1],
                                                      stochastic_method=self.stochastic_method,
                                                      preprocessing_method=self.preprocessing_method,
                                                      tf_graph=self._tf_graph,
                                                      slope_tensor=self._slope_tensor,
                                                      loss_op_name='loss_by_example',
                                                      name='binary_layer',
                                                      stochastic_gradient_estimator=bsn_literals.STRAIGHT_THROUGH)

        return binary_hidden_state, tf.nn.rnn_cell.LSTMStateTuple(binary_cell_state, binary_hidden_state)

    @property
    def stochastic_method(self):
        return self._stochastic_method

    @stochastic_method.setter
    def stochastic_method(self, stochastic_method):
        self._stochastic_method = stochastic_method

    @property
    def preprocessing_method(self):
        return self._preprocessing_method

    @preprocessing_method.setter
    def preprocessing_method(self, preprocessing_method):
        self._preprocessing_method = preprocessing_method

    @property
    def slope_tensor(self):
        return self._slope_tensor

    @slope_tensor.setter
    def slope_tensor(self, slope_tensor):
        self._slope_tensor = slope_tensor
