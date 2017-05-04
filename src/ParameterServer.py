import compression as comp
import tensorflow as tf
#from MNIST_Cnn import MNIST_CNN as CNN
from CIFAR10_CNN import CIFAR10_CNN as CNN
from Ann import ANN
import Queue
import threading
from thrift_conn import init_server
import flowpredictor as fprd

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
        self.lock.release()
        return self.update_count

    def download(self):
        self.lock.acquire()
        model = self.model
        self.lock.release()
        return model

    def getGlobalStatus(self):
        return self.update_count

    def getUploadRecord(self):
        mes = {'mes_type':'show', 'mes_content':None}
        self.__pass_to_queue(mes)
        self.mes_queue.join()
        return

    def __pass_to_queue(self, mes):
        self.mes_queue.put(mes)
        pass

class ParameterServer(threading.Thread):
    def __init__(self, ps_id, cluster_spec):
        self.ip = cluster_spec['ps'][ps_id]['IP']
        self.port = cluster_spec['ps'][ps_id]['Port']

        # training model
        config = gpu_configure()
        self.tensorgraph = CNN(config)
        check_size(comp.preprocess(self.tensorgraph.get_parameters()))
        self.model = comp.preprocess(self.tensorgraph.get_parameters())

        # message queue used between store service and predict service
        self.mes_queue = Queue.Queue(maxsize=0)

        # handle for store service
        handler = Dispatcher(self.model, self.mes_queue)
        self.server = init_server(self.ip, self.port, handler)
        # handle for predict service
        self.predictor = fprd.Predictor([len(cluster_spec['cn'])* 4, 5, len(cluster_spec['cn'])], cluster_spec, self.model, self.mes_queue)

        super(ParameterServer, self).__init__()

    def start_store_service(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()

    def start_predict_service(self):
        self.predictor.start()

    def run(self):
        self.start_predict_service()
        self.start_store_service()

if __name__ == "__main__":
    cluster_spec = {'ps':[{'IP':'127.0.0.1', 'Port':8888}],'cn':[{'IP':'127.0.0.1','Port':60000},{'IP':'127.0.0.1','Port':60001}]}
    ps_node = ParameterServer(0, cluster_spec)
    ps_node.run()
