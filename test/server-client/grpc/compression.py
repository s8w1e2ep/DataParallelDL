from cStringIO import StringIO
from numpy.lib import format
import cPickle
import numpy as np
import zlib

def compress(target):
    s_out = numpy_serialize(target)
    dec_out = zlib.compress(s_out)
    return dec_out

def decompress(target):
    c_out = zlib.decompress(target)
    des_out = numpy_deserialize(c_out)
    return des_out

def numpy_serialize(un_serialized_object):
    un_serialized_object = to_one_dim(un_serialized_object)
    serialized_object = un_serialized_object.tostring()
    return serialized_object

def numpy_deserialize(serialized_object):
    un_serialized_object = np.fromstring(serialized_object, dtype=np.float32)
    un_serialized_object = expand_dim(un_serialized_object)
    return un_serialized_object

def cPickle_serialize(un_serialized_object):
    serialized_object = cPickle.dumps(un_serialized_object)
    return serialized_object

def cPickle_deserilize(serialized_object):
    un_serialized_object = cPickle.loads(serialized_object)
    return un_serialized_object

def to_one_dim(matrix_list): 
    arr_grad = np.reshape(matrix_list[0], 78400)
    arr_grad = np.append(arr_grad, matrix_list[1])
    arr_grad = np.append(arr_grad, np.reshape(matrix_list[2], 1000))
    arr_grad = np.append(arr_grad, matrix_list[3])
    return arr_grad

def expand_dim(arr):
    matrix_list = list()
    matrix_list.append(np.reshape(arr[0:78400],(784, 100)))
    matrix_list.append(arr[78400:78500])
    matrix_list.append(np.reshape(arr[78500:79500], (100, 10)))
    matrix_list.append(arr[79500:79510])
    return matrix_list
