import datetime
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)


def now():
    """Return the current time."""
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')

def lrelu(x, leak=0.2, name="lrelu"):
    """A leaky Rectified Linear Unit."""
    return tf.maximum(x, leak * x)

def nameop(op, name):
    """Give the current op this name, so it can be retrieved in another session."""
    op = tf.identity(op, name=name)
    return op

def tbn(name):
    """Get a tensor of the given name from the graph."""
    return tf.compat.v1.get_default_graph().get_tensor_by_name(name)

def obn(name):
    """Get an object of the given name from the graph."""
    return tf.compat.v1.get_default_graph().get_operation_by_name(name)

def get_all_node_names():
    """Get a list of all the node names in the current graph."""
    return [n.name for n in tf.get_default_graph().as_graph_def().node]

