import xmlrpclib

client = xmlrpclib.ServerProxy("http://localhost:8000/")

import TensorGraph as tg
import compression as comp

tensorgraph = tg.TensorGraph()
s_ret = comp.numpy_serialize(tensorgraph.get_parameters())

import time
tStart = time.time()
for i in range(1000):
    print i
    ret = client.upload(xmlrpclib.Binary(s_ret)).data
tEnd = time.time()
print "elapsed time : %f sec" % (tEnd-tStart)

if ret == s_ret:
    print "success"
else:
    print "failed"
