import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import signal, psutil
import timeit
import multiprocessing
import ComputingNode as cn
import ParameterServer as ps
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from pyspark import SparkContext, SparkConf


def init_conn(ip, port):
    import thriftpy
    from thriftpy.rpc import make_client
    weightsync_thrift = thriftpy.load("weightsync.thrift", module_name="weightsync_thrift")
    client = make_client(weightsync_thrift.WeightSync, ip, port)
    return client

def ps_job(ps_id, cluster_spec):
    ps_node = ps.ParameterServer(ps_id, cluster_spec)
    ps_node.start()

def cn_job(cn_id, cluster_spec, start, length):
    cn_node = cn.ComputingNode(cn_id, cluster_spec, start, length)
    elapsed_time = timeit.Timer(cn_node.run).timeit(number=1)
    print "cn_node %d : %f sec" % ((cn_id), elapsed_time)
    #cn_node.run()
    return 'sucess'

if __name__ == '__main__':

    cluster_spec = dict()
    # 4 machine 
    cluster_spec[4] = {'ps':[{'IP':'127.0.0.1', 'Port':8888}],
                    'cn':[{'IP':'127.0.0.1','Port':60000},
                          {'IP':'127.0.0.1','Port':60001},
                          {'IP':'127.0.0.1','Port':60002},
                          {'IP':'127.0.0.1','Port':60003}]}
    # 2 machine
    cluster_spec[2] = {'ps':[{'IP':'127.0.0.1', 'Port':8888}],'cn':[{'IP':'127.0.0.1','Port':60000},{'IP':'127.0.0.1','Port':60001}]}
    # 1 machine
    cluster_spec[1] = {'ps':[{'IP':'127.0.0.1', 'Port':8888}],'cn':[{'IP':'127.0.0.1','Port':60000}]}
    
    machine_num = int(sys.argv[1])
    ps_num = len(cluster_spec[machine_num]['ps'])
    cn_num = len(cluster_spec[machine_num]['cn'])
    ps_processes = list()
    cn_processes = list()

    # create parameter servers
    for i in range(ps_num):
        ps_job(i, cluster_spec[machine_num])

    # create computing nodes
    training_set_size = 20000
    length = training_set_size / cn_num
    conf = SparkConf().setAppName('paralleltest').setMaster('localcluster').set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=conf)
    x = sc.parallelize(range(cn_num))
    x.map(lambda x: cn_job(x, cluster_spec[machine_num], x*length, length)).collect()
