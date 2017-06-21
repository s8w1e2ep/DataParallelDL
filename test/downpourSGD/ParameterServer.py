import compression as comp
import tensorflow as tf
from CIFAR10_CNN import CIFAR10_CNN as CNN
from thrift_conn import init_server
from cluster_specification import cluster_spec
import threading

def check_size(model):
    print "The total parameter size is %d bytes" % len(model)

def gpu_configure():
    config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class Dispatcher(object):
    def __init__(self, cnn_graph):
        self.cnn_graph = cnn_graph
        self.cnn_graph_shape = self.cnn_graph.get_configure()
        self.update_count = 0
        self.lock = threading.Lock()

    def upload(self, cnid, u_grads):
        self.lock.acquire()
        grads = comp.deprocess(u_grads, self.cnn_graph_shape)
        self.cnn_graph.put_gradients(grads)
        self.update_count += 1
        self.lock.release()
        return self.update_count

    def download(self):
        self.lock.acquire()
        model = self.cnn_graph.get_parameters()
        text = comp.preprocess(model)
        self.lock.release()
        return text

    def getGlobalStatus(self):
        return self.update_count

class ParameterServer(threading.Thread):
    def __init__(self, ps_id):
        self.ip = cluster_spec['ps'][ps_id]['IP']
        self.port = cluster_spec['ps'][ps_id]['Port']
        self.service_list = list()
        self.service_list.append(self.start_store_service)

        # training model
        config = gpu_configure()
        self.cnn_graph = CNN(config)
        compressed_parameters = comp.preprocess(self.cnn_graph.get_parameters())
        check_size(compressed_parameters)

        # handle for store service
        handler = Dispatcher(self.cnn_graph)
        self.server = init_server(self.ip, self.port, handler)

        super(ParameterServer, self).__init__()

    def start_store_service(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()

    def run(self):
        for service in self.service_list:
            service()

if __name__ == '__main__':
    ps_node = ParameterServer(0)
    ps_node.run()
