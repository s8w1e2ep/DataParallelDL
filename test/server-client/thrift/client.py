import TensorGraph as tg
import thriftpy
import compression as comp
import c1

test_thrift = thriftpy.load("benchmarkmessage.thrift", module_name="test_thrift")

from thriftpy.rpc import make_client

#client = make_client(test_thrift.Helloworld, '127.0.0.1', 6000)

tensorgraph = tg.TensorGraph()
s_ret = comp.preprocess(tensorgraph.get_parameters())

paras = tensorgraph.get_parameters()
graph_shape = [para.shape for para in paras]
ret = comp.deprocess(s_ret, graph_shape)

print ret[3]

#import time
#tStart = time.time()
#for i in range(1000):
#    print i
#    ret = client.upload(s_ret)
#tEnd = time.time()
#print "elapsed time : %f sec" % (tEnd-tStart)
#
#if ret == s_ret:
#    print "success"
#else:
#    print "failed"
