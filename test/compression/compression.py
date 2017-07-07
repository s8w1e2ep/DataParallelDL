from numpy.lib import format
import numpy as np
import zlib
import time

data_type = np.float32
zcomp = False

def preprocess(target):
    s_out = numpy_serialize(target)
    if zcomp:
        ts = time.time()
        c_out = zlib.compress(s_out)
        te = time.time()
        print "compression elapsed time : %fs" % (te-ts) 
        return c_out
    else:
        return s_out

def deprocess(target, graph_shape):
    if zcomp:
        ts = time.time()
        dec_out = zlib.decompress(target)
        te = time.time()
        print "decompression elapsed time : %fs" % (te-ts)
        des_out = numpy_deserialize(dec_out, graph_shape)
        return des_out
    else:
        des_out = numpy_deserialize(target, graph_shape)
        return des_out

def numpy_serialize(un_serialized_object):
    flat_arr = to_one_dim(un_serialized_object)
    serialized_object = flat_arr.tostring()
    return serialized_object

def numpy_deserialize(serialized_object, graph_shape):
    flat_arr = np.fromstring(serialized_object, dtype=data_type)
    un_serialized_object = expand_dim(flat_arr, graph_shape)
    return un_serialized_object

def to_one_dim(matrix_list):
    flat_arr = np.empty([0,0], dtype=data_type)
    elap_time = 0
    for arr in matrix_list:
        ts = time.time()
        arr = arr.astype(data_type)
        te = time.time()
        elap_time += (te-ts)
        flat_arr = np.append(flat_arr, arr.ravel())
    print "casting compression elapsed time : %fs" % elap_time
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
