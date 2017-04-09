import tensorflow as tf
from abc import ABCMeta, abstractmethod
import numpy as np

class TensorGraph:
    __metaclass__= ABCMeta
    def __init__(self, config=None):
        self.session = tf.Session(config=config)
        self.parameters = self.create_variable()
        self.input_placeholders = self.create_input_placeholder()
        self.parameter_placeholders = self.create_parameter_placeholder()
        self.graph_op = self.create_graph()
        self.init_parameters()

    @abstractmethod
    def create_variable(self):
        pass

    @abstractmethod
    def create_input_placeholder(self):
        pass

    @abstractmethod
    def create_parameter_placeholder(self):
        pass

    @abstractmethod
    def create_graph(self):
        pass

    def init_parameters(self, path=None):
        if path is None:
            #self.session.run(tf.initialize_all_variables())
            self.session.run(tf.global_variables_initializer())
        else:
            self.load(path)

    def get_parameters(self):
        op = tf.global_variables()
        paras = self.session.run(op)
        return paras[:len(self.parameters)]

    def predict(self, data):
        prediction =self.graph_op[5]
        test_data = self.input_placeholders[0]
        feed_dict = dict()
        feed_dict[test_data] = data
        return self.session.run(prediction, feed_dict=feed_dict)

    def optimize(self, data, label):
        optimization =self.graph_op[4]
        t_data = self.input_placeholders[0]
        t_label = self.input_placeholders[1]
        feed_dict = dict()
        feed_dict[t_data] = data
        feed_dict[t_label] = label
        self.session.run(optimization, feed_dict=feed_dict)

    def get_gradients(self, data, label):
        compute_gradients = self.graph_op[1]
        t_data = self.input_placeholders[0] 
        t_label = self.input_placeholders[1]
        feed_dict = dict()
        feed_dict[t_data] = data
        feed_dict[t_label] = label
        gradient_pairs = self.session.run(compute_gradients, feed_dict=feed_dict)
        gradients = list()
        for gradient in gradient_pairs:
            gradients.append(gradient[0])
        return gradients

    def put_gradients(self, gradients):
        apply_gradients =self.graph_op[2]
        feed_dict = dict()
        for i, gradient_name in enumerate(self.parameter_placeholders):
            feed_dict[gradient_name] = gradients[i]
        self.session.run(apply_gradients, feed_dict=feed_dict)
   
    def put_parameters(self, parameters):
        assign_parameters = self.graph_op[3]
        feed_dict = dict()
        for i, parameter_name in enumerate(self.parameter_placeholders):
            feed_dict[parameter_name] = parameters[i]
        self.session.run(assign_parameters, feed_dict=feed_dict)

    def get_configure(self):
        paras = self.get_parameters()
        graph_shape = [para.shape for para in paras]
        return graph_shape

    def __exit__(self):
        self.session.close()
