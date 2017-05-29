import thriftpy
import compression as comp
from CIFAR10_CNN import CIFAR10_CNN as CNN
import time

test_thrift = thriftpy.load("benchmarkmessage.thrift", module_name="test_thrift")

from thriftpy.rpc import make_client


cnn_graph = CNN()
compressed_parameters = comp.preprocess(cnn_graph.get_parameters())
model = compressed_parameters

client = make_client(test_thrift.Helloworld, '127.0.0.1', 6000)
ts = time.time()
for i in xrange(10):
    client.upload(model)
te = time.time()
print "elapsed time : %fs" % (te-ts)
