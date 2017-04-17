import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import tensorflow as tf
import compression as comp
#from MNIST_Cnn import MNIST_CNN as CNN
from CIFAR10_CNN import CIFAR10_CNN as CNN
from Ann import ANN
from StopWatch import StopWatch
from thrift_conn import init_conn
from dataset import open_cifar10_dataset
from dataset import open_mnist_dataset

def gpu_split(worker_num):
    proportion = 1. / (worker_num+1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=proportion)
    config = tf.ConfigProto(gpu_options=gpu_options)
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class ComputingNode:
    def __init__(self, cn_id, cluster_spec, start, length, path=None, debug=0, fname='../log/cn{}_profiling.log'):
        self.id = cn_id
        self.batch_size = 200
        self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels = open_cifar10_dataset(start,length)
        #self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels = open_mnist_dataset(start,length)
        gpu_config = gpu_split(len(cluster_spec['cn']))
        self.graph = CNN(gpu_config)
        self.graph_shape = self.graph.get_configure()
        self.ps = init_conn(cluster_spec['ps'][0]['IP'], cluster_spec['ps'][0]['Port']) 
        self.num_epochs = 3
        self.sw = StopWatch()
        self.status = {'GlobalStep':0, 'LocalStep':0,'Hit':0}

    def run(self):
        import time
        if not len(self.train_dataset) % self.batch_size == 0:
            raise ValueError('Batch size error')
        all_batch_data = [self.train_dataset[x:x+self.batch_size] for x in xrange(0, len(self.train_dataset), self.batch_size)]
        all_batch_label = [self.train_labels[x:x+self.batch_size] for x in xrange(0, len(self.train_labels), self.batch_size)]
        self.update_parameters()
        for step in range(self.num_epochs):
            self.training(all_batch_data, all_batch_label)
            if step % 1 == 0:
                self.validating()
        self.sw.present()
        print "Local step : %d" % self.status['LocalStep']
        print "Hit count : %d" % self.status['Hit']
        print "Hit rate : %f" % (1000. * self.status['Hit'] / self.status['LocalStep'] * 0.001)
        

    def training(self, all_batch_data, all_batch_label):
        for i in range(len(all_batch_data)):
            # compute the graidents
            self.sw.reset()
            gradients = self.graph.get_gradients(all_batch_data[i], all_batch_label[i])
            self.sw.accumulate('compute_gradients')
            self.update_parameters()
            self.sw.reset()
            self.apply_gradients(gradients)
            self.sw.accumulate('apply_gradients')
            # update the gradients to the ps
            self.upload_parameters()

    def validating(self):
        print("Valid accuracy: %.1f%%" % accuracy(self.graph.predict(self.valid_dataset), self.valid_labels))

    def testing(self):
        print("Test accuracy: %.1f%%" % accuracy(self.graph.predict(self.test_dataset), self.test_labels))

    def upload_parameters(self):
        model = self.graph.get_parameters()
        self.sw.reset()
        text = comp.preprocess(model)
        self.sw.accumulate('preprocess')
        self.status['GlobalStep'] = self.ps.upload(text)
        #self.ps.upload(text)
        self.sw.accumulate('upload')
    
    def apply_gradients(self, gradients):
        self.status['LocalStep'] += 1
        self.graph.put_gradients(gradients)

    def update_parameters(self):
        # sync with ps
        gStatus = self.ps.getGlobalStatus()
        if gStatus == self.status['GlobalStep']:
            self.status['Hit'] += 1
        else:
            self.sw.reset()
            text = self.ps.download()
            self.sw.accumulate('download')
            model = comp.deprocess(text, self.graph_shape)
            self.sw.accumulate('deprocess')
            self.graph.put_parameters(model)
            self.sw.accumulate('put para')

def accuracy(predictions, labels):
    if labels.ndim == 1:
        return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])
    else:
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def open_file(fname):
    existed = os.path.isfile(fname)
    f = open(fname, 'w') if existed else open(fname, 'a')
    return f
