""" Decorator for tensorflow variable scope """

import functools
import tensorflow as tf


def double_wrap(function):
    """

    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.

    Args:
        function(func): The decorated function

    Returns:
        decorator(func)

    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrap: function(wrap, *args, **kwargs)
    return decorator


@double_wrap
def graph_component_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.

    Args:
        function(): The decorated function
        scope(str): The variable scope
        *args:
        **kwargs:

    Returns:
        decorator(func)

    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def graph_component(function):

    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    Args:
        function(func): The decorated function

    Returns:

    """
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def use_network_graph(function):

    """

    Args:
        function:

    Returns:

    """
    @functools.wraps(function)
    def decorator(self):
        with self.tf_graph.as_default():
            return function(self)
    return decorator
