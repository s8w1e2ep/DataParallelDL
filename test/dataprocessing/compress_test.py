import compression as comp
from dataset import open_cifar10_dataset
from CIFAR10_CNN import CIFAR10_CNN as CNN
import time
import numpy as np

cnn_graph = CNN()
cnn_shape = cnn_graph.get_configure()
tdataset, tlabels, _, _, _, _ = open_cifar10_dataset(0, 100)

gradients = cnn_graph.get_gradients(tdataset, tlabels)

for i in range(10):
    print "[Learning]epoch %d......" % i
    cnn_graph.optimize(tdataset, tlabels)
parameters = cnn_graph.get_parameters()

print "type of gradients : ", gradients[0].dtype
print "shape of gradients",
for layer in gradients:
    print layer.shape,
print ""
tStart = time.time()
comp.preprocess(gradients)
tEnd = time.time()
print "compress grad time : %fs" % (tEnd - tStart)

print "type of parameters : ", parameters[0].dtype
print "shape of parameters",
for layer in parameters:
    print layer.shape,
print ""
tStart = time.time()
comp.preprocess(parameters)
tEnd = time.time()
print "compress para time : %fs" % (tEnd - tStart)

flat_grad = comp.to_one_dim(gradients)
import matplotlib.pyplot as plt
def draw_grad(X):
    x_val = np.arange(len(X))
    plt.figure(figsize=(8,4))
    plt.plot(x_val, X,label="value",color="red",linewidth=1)
    plt.ylabel("value")
    plt.legend()
    plt.show()
draw_grad(flat_grad)
