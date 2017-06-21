import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import tensorflow as tf
import compression as comp
from CIFAR10_CNN import CIFAR10_CNN as CNN
from StopWatch import StopWatch
from thrift_conn import init_conn
from dataset import open_cifar10_dataset
from dataset import open_mnist_dataset
import threading
import external
from cluster_specification import cluster_spec

def gpu_split(worker_num):
    proportion = 1. / (worker_num+1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=proportion)
    config = tf.ConfigProto(gpu_options=gpu_options)
    return config

class ComputingNode:
    def __init__(self, cn_id, start, length):
        self.id = cn_id
        self.batch_size = 200
        self.num_epochs = 10
        self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels = open_cifar10_dataset(start,length)
        gpu_config = gpu_split(len(cluster_spec['cn']))
        self.tensorgraph = CNN(gpu_config)
        self.tensorgraph_shape = self.tensorgraph.get_configure()

        # establish connection with parameter server to acquire store service
        self.ps = init_conn(cluster_spec['ps'][0]['IP'], cluster_spec['ps'][0]['Port'])
        self.sw = StopWatch()

    def run(self):
        if not len(self.train_dataset) % self.batch_size == 0:
            raise ValueError('Batch size error')
        all_batch_data = [self.train_dataset[x:x+self.batch_size] for x in xrange(0, len(self.train_dataset), self.batch_size)]
        all_batch_label = [self.train_labels[x:x+self.batch_size] for x in xrange(0, len(self.train_labels), self.batch_size)]
        self.update_parameters()
        for step in range(self.num_epochs):
            self.training(all_batch_data, all_batch_label)
            if step % 1 == 0:
                self.validating()
        self.terminate()

    def terminate(self):
        self.sw.present()

    def training(self, all_batch_data, all_batch_label):
        for i in range(len(all_batch_data)):
            # compute the graidents
            self.sw.reset()
            gradients = self.tensorgraph.get_gradients(all_batch_data[i], all_batch_label[i])
            self.sw.accumulate('compute_gradients')
            # update the gradients to the ps
            self.upload_gradients(gradients)
            self.update_parameters()

    def validating(self):
        print("Valid accuracy: %.1f%%" % accuracy(self.tensorgraph.predict(self.valid_dataset), self.valid_labels)),
        print "\tLoss : " , self.tensorgraph.get_loss(self.valid_dataset, self.valid_labels)

    def testing(self):
        print("Test accuracy: %.1f%%" % accuracy(self.tensorgraph.predict(self.test_dataset), self.test_labels))

    def upload_gradients(self, grads):
        self.sw.reset()
        text = comp.preprocess(grads)
        self.sw.accumulate('preprocess')
        self.ps.upload(self.id, text)
        self.sw.accumulate('upload_gradients')

    def update_parameters(self):
        self.sw.reset()
        text = self.ps.download()
        self.sw.accumulate('download')
        model = comp.deprocess(text, self.tensorgraph_shape)
        self.sw.accumulate('deprocess')
        self.tensorgraph.put_parameters(model)
        self.sw.accumulate('put para')

def accuracy(predictions, labels):
    if labels.ndim == 1:
        return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])
    else:
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

if __name__ == '__main__':
    import argparse
    import timeit
    parser = argparse.ArgumentParser(description='Argument Checker')
    parser.add_argument("-w", "--worker", type=int, help="worker id", default=-1)
    parser.add_argument("-s", "--size", type=int, help="training data size", default=10000)
    args = parser.parse_args()

    if args.worker >= 0:
        start = args.worker * args.size
        length = args.size
        cn_id = args.worker
        cn_node = ComputingNode(cn_id, start, length)
        elapsed_time = timeit.Timer(cn_node.run).timeit(number=1)
        print "cn_node %d : %f sec" % ((cn_id), elapsed_time)
