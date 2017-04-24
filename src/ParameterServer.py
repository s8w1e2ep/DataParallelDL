import compression as comp
import tensorflow as tf
#from MNIST_Cnn import MNIST_CNN as CNN
from CIFAR10_CNN import CIFAR10_CNN as CNN
from Ann import ANN
import threading
from thrift_conn import init_server

def check_size(model):
    print "The total parameter size is %d bytes" % len(model)

def gpu_configure():
    config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class Dispatcher(object):
    def __init__(self, graph):
        self.update_count = 0
        self.model = comp.preprocess(graph.get_parameters())
        self.lock = threading.Lock()

    def upload(self, u_parameters):
        self.lock.acquire()
        self.model = u_parameters
        self.update_count += 1
        self.lock.release()
        return self.update_count

    def download(self):
        self.lock.acquire()
        model = self.model
        self.lock.release()
        return model

    def getGlobalStatus(self):
        return self.update_count

class ParameterServer(threading.Thread):
    def __init__(self, ps_id, cluster_spec):
        self.ip = cluster_spec['ps'][ps_id]['IP']
        self.port = cluster_spec['ps'][ps_id]['Port']
        config = gpu_configure()
        self.graph = CNN(config)
        check_size(comp.preprocess(self.graph.get_parameters()))
        handler = Dispatcher(self.graph)
        self.server = init_server(self.ip, self.port, handler)
        super(ParameterServer, self).__init__() 

    def run(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()
