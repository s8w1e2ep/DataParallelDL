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

def init_conn(ip, port):
    import thriftpy
    from thriftpy.rpc import make_client
    weightsync_thrift = thriftpy.load("weightsync.thrift", module_name="weightsync_thrift")
    client = make_client(weightsync_thrift.WeightSync, ip, port)
    return client

def sig_handler(signum, frame):
    kill_child_processes()

def kill_child_processes():
    ps = init_conn('127.0.0.1', 8888)
    print "update count : %d" % ps.getGlobalStatus()
    parent_pid = os.getpid()
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signal.SIGINT)

def soft_exit(signum, frame):
    sys.exit(0)

def ps_job(ps_id, cluster_spec):
    signal.signal(signal.SIGINT, soft_exit)
    ps_node = ps.ParameterServer(ps_id, cluster_spec)
    ps_node.run()

def cn_job(cn_id, cluster_spec, start, length):
    signal.signal(signal.SIGINT, soft_exit)
    cn_node = cn.ComputingNode(cn_id, cluster_spec, start, length)
    elapsed_time = timeit.Timer(cn_node.run).timeit(number=1)
    print "cn_node %d : %f sec" % ((cn_id), elapsed_time)
    #cn_node.run()

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
        process = multiprocessing.Process(target=ps_job, args=(i, cluster_spec[machine_num]))
        process.start()
        ps_processes.append(process)

    # create computing nodes
    training_set_size = 40000
    length = training_set_size / cn_num
    for i in range(cn_num):
        process = multiprocessing.Process(target=cn_job, args=(i, cluster_spec[machine_num], i*length, length))
        process.start()
        cn_processes.append(process)

    signal.signal(signal.SIGINT, sig_handler)

    # wait for training is done
    for i in range(cn_num):
        cn_processes[i].join() 
    kill_child_processes()
