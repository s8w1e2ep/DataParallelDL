import tensorflow as tf
from TensorGraph import TensorGraph

image_size = 28
num_labels = 10
num_hidden_neurons = 100

class ANN(TensorGraph):    
    def create_variable(self):    
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_neurons]))
        biases1 = tf.Variable(tf.zeros([num_hidden_neurons]))
        weights2 = tf.Variable(tf.truncated_normal([num_hidden_neurons, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))
        
        return weights1, biases1, weights2, biases2

    def create_input_placeholder(self):
        tf_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
        tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

        return tf_dataset, tf_labels

    def create_parameter_placeholder(self):
        w1_space = tf.placeholder(tf.float32, shape=(image_size * image_size, num_hidden_neurons))
        b1_space = tf.placeholder(tf.float32, shape=(num_hidden_neurons))
        w2_space = tf.placeholder(tf.float32, shape=(num_hidden_neurons, num_labels))
        b2_space = tf.placeholder(tf.float32, shape=(num_labels))

        return w1_space, b1_space, w2_space, b2_space

    def create_graph(self):
        tf_dataset, tf_labels = self.input_placeholders
        weights1, biases1, weights2, biases2 = self.parameters
        w1_space, b1_space, w2_space, b2_space = self.parameter_placeholders

        tf_train_h_dataset = tf.matmul(tf_dataset, weights1) + biases1
        tf_train_h_dataset = tf.nn.relu(tf_train_h_dataset)
        logits = tf.matmul(tf_train_h_dataset, weights2) + biases2
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

        optimizer = tf.train.GradientDescentOptimizer(0.5)
        compute_gradients = optimizer.compute_gradients(loss)
        optimization = optimizer.minimize(loss)
        prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_dataset, weights1) + biases1), weights2) + biases2)

        apply_gradients = [weights1.assign(weights1-w1_space),
                          weights2.assign(weights2-w2_space),
                          biases1.assign(biases1-b1_space),
                          biases2.assign(biases2-b2_space)]
        assign_parameters = [weights1.assign(w1_space),
                            weights2.assign(w2_space),
                            biases1.assign(b1_space),
                            biases2.assign(b2_space)]

        return loss, compute_gradients, apply_gradients, assign_parameters, optimization, prediction

if __name__ == '__main__':
    ann = Ann()
    print ann.graph_op
    print type(ann.graph_op) 
