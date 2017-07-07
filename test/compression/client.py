import thriftpy
import time
test_thrift = thriftpy.load("benchmarkmessage.thrift", module_name="test_thrift")

from thriftpy.rpc import make_client
client = make_client(test_thrift.Helloworld, '127.0.0.1', 6000)

import compression as comp
import sys
sys.path.append('../../src')
from CIFAR10_CNN import CIFAR10_CNN as CNN


cnn_graph = CNN()
text = comp.preprocess(cnn_graph.get_parameters())
print len(text)

ts = time.time()
ret = client.echo(text)
te = time.time()
print "networking elapsed time : %fs" % (te-ts)

model = comp.deprocess(ret, cnn_graph.get_configure())
