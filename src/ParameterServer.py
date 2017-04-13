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

def check_size(model):
    print "The total parameter size is %d bytes" % len(model)

def gpu_configure():
    config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class Dispatcher(object):
    def __init__(self):
        self.update_count = 0
        config = gpu_configure()
        self.graph = CNN(config)
        check_size(comp.preprocess(self.graph.get_parameters()))
        self.model = comp.preprocess(self.graph.get_parameters())

    def upload(self, u_parameters):
        self.model = u_parameters
        self.update_count += 1
        return True

    def download(self):
        return self.model

    def get_updateCount(self):
        return self.update_count

def init_server(ip, port):
    weightsync_thrift = thriftpy.load("weightsync.thrift", module_name="weightsync_thrift")
    server = make_server(weightsync_thrift.WeightSync, Dispatcher(), ip, port)
    return server

class ParameterServer:
    def __init__(self, ps_id, cluster_spec):
        self.ip = cluster_spec['ps'][ps_id]['IP']
        self.port = cluster_spec['ps'][ps_id]['Port']
        self.server = init_server(self.ip, self.port) 

    def run(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()
