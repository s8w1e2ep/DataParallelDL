import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")

sess = tf.InteractiveSession()


# conv layer-1
x1 = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x1, [-1, 28, 28, 1])
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
x2 = tf.placeholder(tf.float32, [None, 14, 14, 32])
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(x2, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
x3 = tf.placeholder(tf.float32, [None, 7, 7, 64])
W_fc1 = weight_varible([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(x3, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
x4 = tf.placeholder(tf.float32, [None, 1024])
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(x4, keep_prob)

# output layer: softmax
x5 = tf.placeholder(tf.float32, [None, 1024])
W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(x5, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    batch = mnist.train.next_batch(50)
    start = time.time()
    output_1 = h_pool1.eval(feed_dict = {x1: batch[0]})
    output_2 = h_pool2.eval(feed_dict={x2: output_1})
    output_3 = h_fc1.eval(feed_dict={x3: output_2})
    output_4 = h_fc1_drop.eval(feed_dict={x4: output_3, keep_prob: 0.5})
    output_5 = y_pred.eval(feed_dict={x5: output_4, y_: batch[1]})
    end = time.time()
    print "elapsed time : %fsec" % (end-start) 
