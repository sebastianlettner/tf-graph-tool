from distutils.core import setup


setup(
    name='tf_graph_tool',
    version='1.0',
    packages=[
        'tf_graph_tool',
        'tf_graph_tool/example',
        'tf_graph_tool/builder',
        'tf_graph_tool/util',
        'tf_graph_tool/components',
        'tf_graph_tool/components/convolution',
        'tf_graph_tool/components/dense',
        'tf_graph_tool/components/losses',
        'tf_graph_tool/components/placeholder',
        'tf_graph_tool/components/recurrent',
        'tf_graph_tool/components/trainer',
        'tf_graph_tool/components/recurrent/binary_lstm',
        'tf_graph_tool/components/recurrent/binary_lstm/binary_stochastic_neuron'

    ],
    long_description=open('README.md').read()
)