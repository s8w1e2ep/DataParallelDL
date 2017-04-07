from cStringIO import StringIO
from numpy.lib import format
import cPickle
import numpy as np
import zlib

def preprocess(target):
    s_out = numpy_serialize(target)
    #dec_out = zlib.compress(s_out)
    #return dec_out
    return s_out

def deprocess(target, graph_shape):
    #c_out = zlib.decompress(target)
    #des_out = numpy_deserialize(c_out, graph_shape)
    #return des_out
    return numpy_deserialize(target, graph_shape)

def numpy_serialize(un_serialized_object):
    flat_arr = to_one_dim(un_serialized_object)
    serialized_object = flat_arr.tostring()
    return serialized_object

def numpy_deserialize(serialized_object, graph_shape):
    flat_arr = np.fromstring(serialized_object, dtype=np.float32)
    un_serialized_object = expand_dim(flat_arr, graph_shape)
    return un_serialized_object

def cPickle_serialize(un_serialized_object):
    serialized_object = cPickle.dumps(un_serialized_object)
    return serialized_object

def cPickle_deserilize(serialized_object):
    un_serialized_object = cPickle.loads(serialized_object)
    return un_serialized_object

def to_one_dim(matrix_list):
    flat_arr = np.empty([0,0], dtype=np.float32)
    for arr in matrix_list:
        flat_arr = np.append(flat_arr, arr.ravel())
    return flat_arr

def expand_dim(flat_arr, graph_shape):
    matrix_list = list()
    start_idx = 0
    for shape in graph_shape:
        length = 1
        for dim in shape:
            length *= dim
        expand_arr = np.reshape(flat_arr[start_idx:start_idx+length], shape)
        matrix_list.append(expand_arr)
        start_idx = start_idx + length
    return matrix_list
