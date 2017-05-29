import tensorflow as tf
from TensorGraph import TensorGraph

dtype = tf.float32

def weight_varible(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev, dtype=dtype)
    return tf.Variable(initial)

def bias_variable(shape, value):
    initial = tf.constant(value, shape=shape, dtype=dtype)
    return tf.Variable(initial)

def parameter_placeholder(shape):
    return tf.placeholder(dtype, shape=shape)

def compute_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean

class CIFAR10_CNN(TensorGraph):
    def create_variable(self):
        variable = (weight_varible([5, 5, 3, 64], 5e-2),
                    bias_variable([64], 0.0), 
                    weight_varible([5, 5, 64, 64], 5e-2),
                    bias_variable([64], 0.1),
                    weight_varible([2304, 384], 0.04),
                    bias_variable([384], 0.1),
                    weight_varible([384, 192], 0.04),
                    bias_variable([192], 0.1),
                    weight_varible([192, 10], 1./192.0),
                    bias_variable([10], 0.0))
        return variable

    def create_input_placeholder(self):
        tf_dataset = tf.placeholder(dtype, [None, 32, 32, 3])
        tf_labels = tf.placeholder(dtype, [None])
        return [tf_dataset, tf_labels]

    def create_parameter_placeholder(self):
        para_placeholder = (parameter_placeholder([5, 5, 3, 64]),
                            parameter_placeholder([64]), 
                            parameter_placeholder([5, 5, 64, 64]),
                            parameter_placeholder([64]),
                            parameter_placeholder([2304, 384]),
                            parameter_placeholder([384]),
                            parameter_placeholder([384, 192]),
                            parameter_placeholder([192]),
                            parameter_placeholder([192, 10]),
                            parameter_placeholder([10]))
        return para_placeholder

    def create_graph(self):
        lr = 1e-3
        variables = self.parameters
        images, labels = self.input_placeholders
        batch_size = batch_size = tf.shape(images)[0] 
        para_ph = self.parameter_placeholders

        resize_images = tf.image.resize_images(images, [24, 24])
        conv = tf.nn.conv2d(resize_images, variables[0], [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, variables[1])
        conv1 = tf.nn.relu(pre_activation)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

        conv = tf.nn.conv2d(norm1, variables[2], [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, variables[3])
        conv2 = tf.nn.relu(pre_activation)
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        reshape = tf.reshape(pool2, [batch_size, -1])
        local3 = tf.nn.relu(tf.matmul(reshape, variables[4]) + variables[5])
        local4 = tf.nn.relu(tf.matmul(local3, variables[6]) + variables[7])
        logits = tf.add(tf.matmul(local4, variables[8]), variables[9])
        loss = compute_loss(logits, labels)

        optimizer = tf.train.GradientDescentOptimizer(lr)
        compute_gradient_op = optimizer.compute_gradients(loss)
        optimization = optimizer.minimize(loss)

        var = tf.global_variables()
        grads = [(para_ph[i], var[i]) for i in range(len(variables))]
        apply_gradient_op = optimizer.apply_gradients(grads)
        assign_parameters = [variables[i].assign(para_ph[i]) for i in range(len(variables))]

        return (loss, compute_gradient_op, apply_gradient_op, assign_parameters, optimization, logits)
