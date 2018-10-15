""" Base Module. Implements an interface for graph components. """


class BaseComponent(object):

    """ Abstract Interface for Graph Components. """

    def __init__(self,
                 name,
                 output,
                 type,
                 scope=None,
                 ):

        """

        Args:
            scope(): Scope of the component e.g. 'Encoder'. If scope is None no scope will be created
            name(): Name of the component e.g 'Layer1'
            output(tensor): The output tensor of the component. This tensor will be used as input for the next component.

        """

        self._name = name
        self._output = output
        self._type = type
        self._scope = scope

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        return self._output

    @property
    def type(self):
        return self._type

    @property
    def scope(self):
        return self._scope

    def decorate_graph(self, compute_graph):
        raise NotImplementedError
