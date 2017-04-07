import numpy as np 
import cPickle as pickle
import tensorflow as tf
import TensorGraph as tg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

batch_size = 200

def training(data, label, valid_dataset, valid_labels):
    #gradients = tensorgraph.get_gradients(data, label)
    for epoch in range(10):
        for i in range(len(data)/batch_size):
            x = i * batch_size
            tensorgraph.optimize(data[x:x+batch_size], label[x:x+batch_size])
        print("Valid accuracy: %.1f%%" % accuracy(tensorgraph.predict(valid_dataset), valid_labels))
    return tensorgraph.get_parameters()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def open_dataset(start, length, pickle_file='/home/rche/data/notMNIST.pickle'):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        train_dataset, train_labels = reformat(train_dataset, train_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        test_dataset, test_labels = reformat(test_dataset, test_labels)
    return train_dataset[start:start+length], train_labels[start:start+length], valid_dataset, valid_labels, test_dataset, test_labels

def reformat(dataset, labels):
    image_size = 28
    num_labels = 10
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_dataset(0,10000)
tensorgraph = tg.TensorGraph()

times = 10

def cPickle_test(data):
    for i in range(times):
        pickle.loads(pickle.dumps(data))
    
def built_in_test(data):
    result = None
    for i in range(times):
        result = np.fromstring(data.tostring(), dtype=data.dtype)
    return result

def compress_test(data):
    result = None
    for i in range(times):
        result = zlib.decompress(zlib.compress(data))
    return result

gradients = training(train_dataset,train_labels, valid_dataset, valid_labels)
arr_grad = np.reshape(gradients[0], 78400)
arr_grad = np.append(arr_grad, gradients[1])
arr_grad = np.append(arr_grad, np.reshape(gradients[2], 1000))
arr_grad = np.append(arr_grad, gradients[3])

tStart = time.time()
a = built_in_test(arr_grad)
tEnd = time.time()
print "%fsec" % (tEnd - tStart)

import sys
import zlib
print all(i == j for (i, j) in zip(a, arr_grad))
s_object = arr_grad.tostring()
print type(s_object)
print sys.getsizeof(s_object)
print len(s_object)

tStart = time.time()
compress_test(s_object)
tEnd = time.time()
print "%fsec" % (tEnd - tStart)
c_out = zlib.compress(s_object)
print sys.getsizeof(c_out)

tStart = time.time()
cPickle_test(arr_grad)
tEnd = time.time()
print "%fsec" % (tEnd - tStart)

s_object = pickle.dumps(arr_grad)
print type(s_object)
print sys.getsizeof(s_object)
print len(s_object)
