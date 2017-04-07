import tensorflow as tf
from TensorGraph import TensorGraph

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def parameter_placeholder(shape):
    return tf.placeholder(tf.float32, shape=shape)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class MNIST_CNN(TensorGraph):
    def create_variable(self):
        variable = (weight_varible([5, 5, 1, 32]),
                    bias_variable([32]), 
                    weight_varible([5,5, 32, 64]),
                    bias_variable([64]),
                    weight_varible([3136, 1024]),
                    bias_variable([1024]),
                    weight_varible([1024, 10]),
                    bias_variable([10]))
        return variable

    def create_input_placeholder(self):
        tf_dataset = tf.placeholder(tf.float32, [None, 784])
        tf_labels = tf.placeholder(tf.float32, [None, 10])
    
        return [tf_dataset, tf_labels]

    def create_parameter_placeholder(self):
        para_placeholder = list()
        para_placeholder.append(parameter_placeholder([5, 5, 1, 32]))
        para_placeholder.append(parameter_placeholder([32]))
        para_placeholder.append(parameter_placeholder([5, 5, 32, 64]))
        para_placeholder.append(parameter_placeholder([64]))
        para_placeholder.append(parameter_placeholder([3136, 1024]))
        para_placeholder.append(parameter_placeholder([1024]))
        para_placeholder.append(parameter_placeholder([1024, 10]))
        para_placeholder.append(parameter_placeholder([10]))

        return para_placeholder

    def create_graph(self):
        keep_prob = 0.5
        learning_rate = 0.0001
        tf_dataset, tf_labels = self.input_placeholders
        variables = self.parameters
        para_ph = self.parameter_placeholders
        x_image = tf.reshape(tf_dataset, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, variables[0]) + variables[1])
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, variables[2]) + variables[3])
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, variables[4]) + variables[5])
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, variables[6]) + variables[7])
        loss = -tf.reduce_sum(tf_labels * tf.log(y_conv))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimization = optimizer.minimize(loss)
        compute_gradients = optimizer.compute_gradients(loss)
        prediction = y_conv
        apply_gradients = []
        for i in range(len(variables)):
            apply_gradients.append(variables[i].assign(variables[i] - para_ph[i]))

        assign_parameters = []
        for i in range(len(variables)):
            assign_parameters.append(variables[i].assign(para_ph[i]))

        return (loss, compute_gradients, apply_gradients, assign_parameters, optimization, prediction)

if __name__ == '__main__':
    cnn = MNSIT_CNN()
    
