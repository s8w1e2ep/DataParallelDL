import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import tensorflow as tf
import compression as comp
from CIFAR10_CNN import CIFAR10_CNN as CNN
from StopWatch import StopWatch
from thrift_conn import init_conn
from thrift_conn import init_receiver
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

class Handler(object):
    def __init__(self):
        self.status = -1
        self.model = None

    def forward(self, newstatus, newmodel):
        self.status = newstatus
        self.model = newmodel
    
    def getStatus(self):
        return self.status

    def getModel(self):
        return self.model

def receive(ip, port, handler):
    receiver = init_receiver(ip, port, handler)
    print 'Start receiver service(%s:%d)' % (ip, port)
    receiver.serve()

class ComputingNode:
    def __init__(self, cn_id, start, length, receive_service=True, uploading_in_background=True):
        self.id = cn_id
        self.batch_size = 200
        self.num_epochs = 3
        self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels = open_cifar10_dataset(start,length)
        gpu_config = gpu_split(len(cluster_spec['cn']))
        self.tensorgraph = CNN(gpu_config)
        self.tensorgraph_shape = self.tensorgraph.get_configure()

        # establish connection with parameter server to acquire store service
        self.ps = init_conn(cluster_spec['ps'][0]['IP'], cluster_spec['ps'][0]['Port'])

        if receive_service:
            # start a model receiver service
            self.service_handler = Handler()
            service = threading.Thread(target = receive, args=(cluster_spec['cn'][cn_id]['IP'], cluster_spec['cn'][cn_id]['Port'], self.service_handler))
            service.daemon = True
            service.start()
            self.update_parameters = self.update_parameters_opt
        else:
            self.update_parameters = self.update_parameters_ori

        # switch between origin or optimized mode for uploading parameters
        self.lock = threading.Lock()
        if uploading_in_background:
            self.upload_parameters = self.upload_parameters_bg
        else:
            self.upload_parameters = self.upload_parameters_ori

        self.sw = StopWatch()
        self.status = {'GlobalStep':-1, 'LocalStep':0,'LocalHit':0, 'RemoteHit':0}

    def run(self):
        if not len(self.train_dataset) % self.batch_size == 0:
            raise ValueError('Batch size error')
        all_batch_data = [self.train_dataset[x:x+self.batch_size] for x in xrange(0, len(self.train_dataset), self.batch_size)]
        all_batch_label = [self.train_labels[x:x+self.batch_size] for x in xrange(0, len(self.train_labels), self.batch_size)]
        del self.train_dataset
        del self.train_labels
        self.ps.notifyToStart(self.id)
        self.update_parameters()
        for step in range(self.num_epochs):
            self.training(all_batch_data, all_batch_label)
            if step % 1 == 0:
                self.validating()
        self.terminate()

    def terminate(self):
        self.sw.present()
        print "Hit count : %d(%d+%d)" % (self.status['LocalHit']+self.status['RemoteHit'], self.status['LocalHit'], self.status['RemoteHit'])
        print "Hit rate : %f" % (1000. * (self.status['LocalHit']+self.status['RemoteHit']) / self.status['LocalStep'] * 0.001)

    def training(self, all_batch_data, all_batch_label):
        for i in range(len(all_batch_data)):
            # compute the graidents
            self.sw.reset()
            gradients = self.tensorgraph.get_gradients(all_batch_data[i], all_batch_label[i])
            self.sw.accumulate('compute_gradients')
            self.update_parameters()
            self.sw.reset()
            self.apply_gradients(gradients)
            self.sw.accumulate('apply_gradients')
            # update the gradients to the ps
            self.upload_parameters()
            self.sw.accumulate('upload_parameters')

    def validating(self):
        print("Valid accuracy: %.1f%%" % accuracy(self.tensorgraph.predict(self.valid_dataset), self.valid_labels))

    def testing(self):
        print("Test accuracy: %.1f%%" % accuracy(self.tensorgraph.predict(self.test_dataset), self.test_labels))

    def upload_parameters_ori(self):
        self.lock.acquire()
        model = self.tensorgraph.get_parameters()
        text = comp.preprocess(model)
        self.status['GlobalStep'] = self.ps.upload(self.id, text)
        self.lock.release()

    def upload_parameters_bg(self):
        upload_bg = threading.Thread(target=self.upload_parameters_ori)
        upload_bg.start()

    def apply_gradients(self, gradients):
        self.status['LocalStep'] += 1
        self.tensorgraph.put_gradients(gradients)

    def update_parameters_opt(self):
        # sync with ps
        try:
            gStatus = self.ps.getGlobalStatus()
        except:
            gStatus = -1
        if gStatus == self.status['GlobalStep']:
            self.status['LocalHit'] += 1
            return
        if gStatus == self.service_handler.getStatus():
            self.status['RemoteHit'] += 1
            model = comp.deprocess(self.service_handler.getModel(), self.tensorgraph_shape)
            self.tensorgraph.put_parameters(model)
            return
        self.sw.reset()
        try:
            text = self.ps.download()
            self.sw.accumulate('download')
            model = comp.deprocess(text, self.tensorgraph_shape)
            self.sw.accumulate('deprocess')
        except:
            del self.ps
            self.ps = init_conn(cluster_spec['ps'][0]['IP'], cluster_spec['ps'][0]['Port'])
            return
        self.sw.reset()
        self.tensorgraph.put_parameters(model)
        self.sw.accumulate('put para')

    def update_parameters_ori(self):
        self.sw.reset()
        try:
            text = self.ps.download()
            self.sw.accumulate('download')
            model = comp.deprocess(text, self.tensorgraph_shape)
            self.sw.accumulate('deprocess')
        except:
            del self.ps
            self.ps = init_conn(cluster_spec['ps'][0]['IP'], cluster_spec['ps'][0]['Port'])
            return
        self.tensorgraph.put_parameters(model)
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
