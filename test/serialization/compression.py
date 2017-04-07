from cStringIO import StringIO
from numpy.lib import format
import cPickle

def from_string(s):
    f = StringIO(s)
    arr = format.read_array(f)
    return arr 

def to_string(arr):
    f = StringIO()
    format.write_array(f, arr)
    s = f.getvalue()
    return s

def compress(uncompressed_model):
    compressed_model = cPickle.dumps(uncompressed_model)
    return compressed_model

def decompress(compressed_model):
    decompressed_model = cPickle.loads(compressed_model)
    return decompressed_model
