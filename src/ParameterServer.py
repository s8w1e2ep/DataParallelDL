import sys
from SimpleXMLRPCServer import *
import socket
import fcntl 
import struct
import compression as comp
import tensorflow as tf
#from MNIST_Cnn import MNIST_CNN as CNN
from CIFAR10_CNN import CIFAR10_CNN as CNN
from Ann import ANN
import thriftpy
from thriftpy.rpc import make_server
import threading

def check_size(model):
    print "The total parameter size is %d bytes" % len(model)

def gpu_configure():
    config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class Dispatcher(object):
    def __init__(self, graph):
        self.update_count = 0
        self.model = comp.preprocess(graph.get_parameters())

    def upload(self, u_parameters):
        self.model = u_parameters
        self.update_count += 1
        return self.update_count

    def download(self):
        return self.model

    def getGlobalStatus(self):
        return self.update_count

def init_server(ip, port, graph):
    weightsync_thrift = thriftpy.load("weightsync.thrift", module_name="weightsync_thrift")
    requestHandler = Dispatcher(graph)
    server = make_server(weightsync_thrift.WeightSync, requestHandler, ip, port)
    return server

class ParameterServer(threading.Thread):
    def __init__(self, ps_id, cluster_spec):
        self.ip = cluster_spec['ps'][ps_id]['IP']
        self.port = cluster_spec['ps'][ps_id]['Port']
        config = gpu_configure()
        self.graph = CNN(config)
        check_size(comp.preprocess(self.graph.get_parameters()))
        self.server = init_server(self.ip, self.port, self.graph) 

    def run(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()
