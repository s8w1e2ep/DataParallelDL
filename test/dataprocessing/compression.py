from cStringIO import StringIO
from numpy.lib import format
import cPickle
import numpy as np
import zlib
import time

data_type = np.float16

def preprocess(target):
    s_out = numpy_serialize(target)
    return s_out

def deprocess(target, graph_shape):
    return numpy_deserialize(target, graph_shape)

def numpy_serialize(un_serialized_object):
    ts1 = time.time()
    flat_arr = to_one_dim(un_serialized_object)
    te1 = time.time()
    ts2 = time.time()
    serialized_object = flat_arr.tostring()
    te2 = time.time()
    print "flatten:%1.5fs\tserialize:%1.5fs" % (te1-ts1, te2-ts2)
    return serialized_object

def numpy_deserialize(serialized_object, graph_shape):
    flat_arr = np.fromstring(serialized_object, dtype=data_type)
    un_serialized_object = expand_dim(flat_arr, graph_shape)
    return un_serialized_object

def to_one_dim(matrix_list):
    flat_arr = np.empty([0,0], dtype=data_type)
    elapsed_time = 0
    for arr in matrix_list:
        ts = time.time()
        arr = arr.astype(data_type)
        te = time.time()
        flat_arr = np.append(flat_arr, arr.ravel())
        elapsed_time += (te-ts)
    print "type conversion : %fs" % elapsed_time
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
