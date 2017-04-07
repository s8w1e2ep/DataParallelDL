import tensorflow as tf

image_size = 28
num_labels = 10
num_hidden_neurons = 100

class TensorGraph:
    def __init__(self, config=None):
        self.session = tf.Session(config=config)
        self.parameters = self.create_variable()
        self.input_placeholders = self.create_input_placeholder()
        self.parameter_placeholders = self.create_parameter_placeholder()
        self.loss, self.gradient_pairs, self.apply_gradients, self.assign_parameters, self.optimization, self.prediction = self.create_graph()
        self.init_parameters(path=None)

    def create_variable(self):
        
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_neurons]))
        biases1 = tf.Variable(tf.zeros([num_hidden_neurons]))
        weights2 = tf.Variable(tf.truncated_normal([num_hidden_neurons, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        return weights1, biases1, weights2, biases2

    def create_input_placeholder(self):
        tf_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
        tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        tf_test_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))

        return tf_dataset, tf_labels, tf_test_dataset

    def create_parameter_placeholder(self):
        w1_space = tf.placeholder(tf.float32, shape=(image_size * image_size, num_hidden_neurons))
        b1_space = tf.placeholder(tf.float32, shape=(num_hidden_neurons))
        w2_space = tf.placeholder(tf.float32, shape=(num_hidden_neurons, num_labels))
        b2_space = tf.placeholder(tf.float32, shape=(num_labels))

        return w1_space, b1_space, w2_space, b2_space

    def create_graph(self):
        tf_dataset, tf_labels, tf_test_dataset = self.input_placeholders
        weights1, biases1, weights2, biases2 = self.parameters
        w1_space, b1_space, w2_space, b2_space = self.parameter_placeholders

        tf_train_h_dataset = tf.matmul(tf_dataset, weights1) + biases1
        tf_train_h_dataset = tf.nn.relu(tf_train_h_dataset)
        logits = tf.matmul(tf_train_h_dataset, weights2) + biases2
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

        optimizer = tf.train.GradientDescentOptimizer(0.5)
        gradient_pairs = optimizer.compute_gradients(loss)
        optimization = optimizer.minimize(loss)
        prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)

        apply_gradients = [weights1.assign(weights1-w1_space),
                          weights2.assign(weights2-w2_space),
                          biases1.assign(biases1-b1_space),
                          biases2.assign(biases2-b2_space)]
        assign_parameters = [weights1.assign(w1_space),
                            weights2.assign(w2_space),
                            biases1.assign(b1_space),
                            biases2.assign(b2_space)]

        return loss, gradient_pairs, apply_gradients, assign_parameters, optimization, prediction

    def init_parameters(self, path):
        if path is None:
            self.session.run(tf.initialize_all_variables())
        else:
            self.load(path)

    def get_parameters(self):
        # tensorflow APIs in different version
        variables = tf.global_variables()
        #variables = tf.GraphKeys.GLOBAL_VARIABLES()
        #variables = tf.all_variables()
        return self.session.run(variables)

    def predict(self, data):
        _, _, test_data = self.input_placeholders
        feed_dict = dict()
        feed_dict[test_data] = data
        return self.session.run(self.prediction, feed_dict=feed_dict)

    def optimize(self, data, label):
        t_data, t_label, _ = self.input_placeholders
        feed_dict = dict()
        feed_dict[t_data] = data
        feed_dict[t_label] = label
        self.session.run(self.optimization, feed_dict=feed_dict)

    def get_gradients(self, data, label):
        t_data, t_label, _ = self.input_placeholders
        feed_dict = dict()
        feed_dict[t_data] = data
        feed_dict[t_label] = label
        return self.session.run(self.gradient_pairs, feed_dict=feed_dict)

    def put_gradients(self, gradients):
        feed_dict = dict()
        for i, gradient_name in enumerate(self.parameter_placeholders):
            feed_dict[gradient_name] = gradients[i]
        self.session.run(self.apply_gradients, feed_dict=feed_dict)
   
    def put_parameters(self, parameters):
        feed_dict = dict()
        for i, parameter_name in enumerate(self.parameter_placeholders):
            feed_dict[parameter_name] = parameters[i]
        self.session.run(self.assign_parameters, feed_dict=feed_dict)
 
    def __exit__(self):
        self.session.close()

if __name__ == '__main__':
    tg = TensorGraph()
    print tg.get_parameters()
    
