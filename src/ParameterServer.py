import compression as comp
import tensorflow as tf
from CIFAR10_CNN import CIFAR10_CNN as CNN
import Queue
import threading
from thrift_conn import init_server
import flowpredictor as fprd
from cluster_specification import cluster_spec
import logging

def check_size(model):
    print "The total parameter size is %d bytes" % len(model)

def gpu_configure():
    config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class Dispatcher(object):
    def __init__(self, shared_training_model, shared_mes_queue):
        self.update_count = 0
        self.model = shared_training_model
        self.mes_queue = shared_mes_queue
        self.lock = threading.Lock()

    def notifyToStart(self, cnid):
        mes = {'mes_type':'notify', 'mes_content':cnid}
        self.__pass_to_queue(mes)

    def upload(self, cnid, u_parameters):
        self.lock.acquire()
        mes = {'mes_type':'prepare_to_train', 'mes_content':cnid}
        self.__pass_to_queue(mes)
        self.model = u_parameters
        self.update_count += 1
        status = self.update_count
        self.lock.release()
        return status

    def non_blocking_upload(self, cnid, u_parameters):
        self.lock.acquire()
        mes = {'mes_type':'prepare_to_train', 'mes_content':cnid}
        self.__pass_to_queue(mes)
        self.model = u_parameters
        self.update_count += 1
        status = self.update_count
        self.lock.release()
        return

    def download(self):
        self.lock.acquire()
        model = self.model
        self.lock.release()
        return model

    def getGlobalStatus(self):
        self.lock.acquire()
        status = self.update_count
        self.lock.release()
        return status

    def getGlobalModel(self):
        self.lock.acquire()
        model = self.model
        self.lock.release()
        return model

    def getUploadRecord(self):
        mes = {'mes_type':'terminate', 'mes_content':None}
        self.__pass_to_queue(mes)
        self.mes_queue.join()
        return

    def __pass_to_queue(self, mes):
        self.mes_queue.put(mes)

logging.basicConfig(level=logging.INFO)
class ParameterServer(threading.Thread):
    def __init__(self, ps_id, predict_service=True):
        self.ip = cluster_spec['ps'][ps_id]['IP']
        self.port = cluster_spec['ps'][ps_id]['Port']
        self.service_list = list()
        if predict_service:
            self.service_list.append(self.start_predict_service)
        self.service_list.append(self.start_store_service)

        # training model
        config = gpu_configure()
        with CNN(config) as cnn_graph:
            compressed_parameters = comp.preprocess(cnn_graph.get_parameters())
            check_size(compressed_parameters)
            self.model = compressed_parameters

        # message queue used between store service and predict service
        self.mes_queue = Queue.Queue(maxsize=0)

        # handle for store service
        handler = Dispatcher(self.model, self.mes_queue)
        self.server = init_server(self.ip, self.port, handler)
        # handle for predict service
        self.predictor = fprd.Predictor([len(cluster_spec['cn'])* 4, 10, len(cluster_spec['cn'])], cluster_spec, handler, self.mes_queue)

        super(ParameterServer, self).__init__()

    def start_store_service(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()

    def start_predict_service(self):
        self.predictor.start()

    def run(self):
        for service in self.service_list:
            service()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Argument Checker')
    parser.add_argument("-p", "--predict", type=bool, help="enable predict service", default=False)
    args = parser.parse_args()

    ps_node = ParameterServer(0, args.predict)
    ps_node.run()
