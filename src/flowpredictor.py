import tensorflow as tf
import threading
import time
import numpy as np
from thrift_conn import init_sender

def gpu_configure():
    config = tf.ConfigProto(device_count = {'GPU': 0})
    return config

class Policy:
    def __init__(self, input_dim, hidden_neurons, output_dim):
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            self.variable = self.create_variable(input_dim, hidden_neurons, output_dim)
            self.inputset = self.create_placeholder(input_dim, output_dim)
            self.model_op = self.create_model()
            self.init_op = tf.initialize_all_variables()
        config = gpu_configure()
        self.session = tf.Session(config=config, graph=self.graph)
        self.init_parameters()
        
    def init_parameters(self, path=None):
        if path is None:
            self.session.run(self.init_op)
        else:
            self.load(path)

    def create_variable(self, input_dim, hidden_neurons, output_dim):
        # Variables.
        weights1 = tf.Variable(tf.truncated_normal([input_dim, hidden_neurons]))
        biases1 = tf.Variable(tf.zeros([hidden_neurons]))
        weights2 = tf.Variable(tf.truncated_normal([hidden_neurons, output_dim]))
        biases2 = tf.Variable(tf.zeros([output_dim]))
        
        return weights1, biases1, weights2, biases2

    def create_placeholder(self, input_dim, output_dim):
        # define input and output dimension
        tf_dataset = tf.placeholder(tf.float32, shape=(None, input_dim))
        tf_labels = tf.placeholder(tf.float32, shape=(None, output_dim))
    
        return tf_dataset, tf_labels
    
    def create_model(self):
       
        weights1, biases1, weights2, biases2 = self.variable
        tf_dataset, tf_labels = self.inputset
        
        # Training computation.
        tf_hidden = tf.matmul(tf_dataset, weights1) + biases1
        tf_activate = tf.nn.relu(tf_hidden)
        logits = tf.matmul(tf_activate, weights2) + biases2 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))
        
        # Optimizer.
        optimization = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        # action
        predict = tf.nn.softmax(logits)
        return loss, optimization, predict
    
    def train(self, data, label):
        tf_data, tf_label = self.inputset
        operation = self.model_op[1]
        feed_dict = {tf_data : data, tf_label : label}
        self.session.run(operation, feed_dict=feed_dict)

    def predict(self, data):
        tf_data = self.inputset[0]
        operation = self.model_op[2]
        feed_dict = {tf_data : data}
        return self.session.run(operation, feed_dict=feed_dict)

queue_size = 3

class Predictor(threading.Thread):
    def __init__(self, dim, cluster_spec, store_service_handler, shared_mes_queue):
        self.dim = dim
        self.cluster_spec = cluster_spec
        self.policy = Policy(dim[0], dim[1], dim[2])
        # create a copy of store_service_handler in local
        self.store_service_handler = store_service_handler
        # a shared mes_queue
        self.mes_queue = shared_mes_queue
        # a record queue
        self.arrive_queue = [[0 for i in range(dim[2])] for j in range(queue_size)]
        self.arrive_record = [0 for i in range(dim[2])]
        self.elapsed_record = [0. for i in range(dim[2])]
        self.state = self.__format_data()
        self.pnext = np.argmax(self.policy.predict(np.expand_dims(self.state, axis=0)))
        self.history = list()
        self.conn_table = dict()
        super(Predictor, self).__init__()
        self.daemon = True

    def __format_data(self):
        ts_data = normalize(self.arrive_record)
        # different training data input
        et_data = self.elapsed_record
        #et_data = stdev_normalize(self.elapsed_record)
        #et_data = softmax(self.elapsed_record)
        self.__record_pop()
        self.__record_append(ts_data)
        mergedata = []
        for item in self.arrive_queue:
            mergedata = mergedata + item
        mergedata = mergedata + et_data 
        return mergedata

    def prepare_to_train(self, cn_id):
        current_time = time.time()
        data = self.state 
        eData = np.expand_dims(data, axis=0)
        label = np.zeros(self.dim[2])
        label[cn_id] = 1
        eLabel = np.expand_dims(label, axis=0)
        self.policy.train(eData, eLabel)
        self.history.append((data, label))
        # update records
        self.elapsed_record[cn_id] = current_time - self.arrive_record[cn_id]
        self.arrive_record[cn_id] = current_time
        self.state = self.__format_data()

    def __record_append(self, item):
        self.arrive_queue.append(item)

    def __record_pop(self):
        ret = self.arrive_queue.pop(0)
        del ret

    def notify(self, cn_id):
        self.arrive_record[cn_id] = time.time()
        try:
            self.conn_table[cn_id] = init_sender(self.cluster_spec['cn'][cn_id]['IP'], self.cluster_spec['cn'][cn_id]['Port'])
        except Exception as e:
            print e
            time.sleep(2)
            print "reconnect to (%s:%d)" % (self.cluster_spec['cn'][cn_id]['IP'], self.cluster_spec['cn'][cn_id]['Port'])
            self.conn_table[cn_id] = init_sender(self.cluster_spec['cn'][cn_id]['IP'], self.cluster_spec['cn'][cn_id]['Port'])

    def guess(self):
        return np.argmax(self.policy.predict(np.expand_dims(self.state, axis=0)))

    def show(self):
        self.batch_train()

    def batch_train(self):
        input_set_size = len(self.history)
        data = [item[0] for item in self.history]
        labels = [item[1] for item in self.history]
        for i in range(1000):
            for offset in xrange(0, input_set_size, 100):
                self.policy.train(data[offset:offset+100], labels[offset:offset+100])
            if i % 100 == 0:
                self.__testing(data, labels)

    def __testing(self, data, labels):
        predictions = self.policy.predict(data)
        print "Test accuracy: %.1f%%" % accuracy(predictions, labels)

    def run(self):
        while True:
            mes = self.mes_queue.get()
            try:
                if mes['mes_type'] == 'prepare_to_train':
                    self.prepare_to_train(mes['mes_content'])
                    try:
                        cnid = self.guess()
                        self.conn_table[cnid].forward(self.store_service_handler.getGlobalStatus(), self.store_service_handler.getGlobalModel())
                    except Exception as e:
                        pass
                elif mes['mes_type'] == 'notify':
                    self.notify(mes['mes_content'])
                elif mes['mes_type'] == 'show':
                    self.show()
            except Exception as e:
                pass
            self.mes_queue.task_done()

def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()

def stdev_normalize(x):
    # (x-Mean)/Deviation
    stdev = np.std(x)
    mean = np.mean(x)
    stdev = 1 if stdev == 0 else stdev
    return map(lambda i: (i-mean) * 100 / stdev * 0.01, x)

def normalize(x):
    # (timeseries-timeseries.min())/(timeseries.max()-timeseries.min())
    baseline = min(x)
    last_finish = max(x)
    diff = last_finish - baseline
    diff = 1 if diff == 0 else diff
    return map(lambda i: (i-baseline) * 100 / diff * 0.01, x)
