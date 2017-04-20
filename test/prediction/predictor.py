import tensorflow as tf

class predictor:
    def __init__(self, input_dim, hidden_neurons, output_dim):
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.inputset = self.create_placeholder(input_dim, output_dim)
        self.model_op = self.create_model(input_dim, hidden_neurons, output_dim)
        self.session = tf.Session()
        self.init_parameters()
        
    def init_parameters(self, path=None):
        if path is None:
            #self.session.run(tf.initialize_all_variables())
            self.session.run(tf.global_variables_initializer())
        else:
            self.load(path)   
 
    def create_placeholder(self, input_dim, output_dim):
        # define input and output dimension
        tf_dataset = tf.placeholder(tf.float32, shape=(None, input_dim))
        tf_labels = tf.placeholder(tf.float32, shape=(None, output_dim))
    
        return tf_dataset, tf_labels
    
    def create_model(self, input_dim, hidden_neurons, output_dim):
    
        tf_dataset, tf_labels = self.inputset
            
        # Variables.
        weights1 = tf.Variable(tf.truncated_normal([input_dim, hidden_neurons]))
        biases1 = tf.Variable(tf.zeros([hidden_neurons]))
        weights2 = tf.Variable(tf.truncated_normal([hidden_neurons, output_dim]))
        biases2 = tf.Variable(tf.zeros([output_dim]))
    
        # Training computation.
        tf_hidden = tf.matmul(tf_dataset, weights1) + biases1
        tf_activate = tf.nn.relu(tf_hidden)
        logits = tf.matmul(tf_activate, weights2) + biases2 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))
    
        # Optimizer.
        optimization = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    
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

if __name__ == "__main__":
    prd = predictor(8, 3 ,4)
    tds = [0.2, 0.1, 0.3, -2.7, 0.2, 0.1, 0.3, -2.7]
    tls = [0, 1, 0, 0]

    print prd.predict([tds])
    for i in range(1000):
        prd.train([tds], [tls])
    import numpy as np
    print np.argmax(prd.predict([tds]))
    print prd.predict([tds])
