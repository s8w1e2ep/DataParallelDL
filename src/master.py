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
from thrift_conn import init_conn
from cluster_specification import cluster_spec
import argparse

def sig_handler(signum, frame):
    kill_child_processes()

def kill_child_processes():
    ps = init_conn(cluster_spec['ps'][0]['IP'], cluster_spec['ps'][0]['Port'])
    print "update count : %d" % ps.getGlobalStatus()
    ps.getUploadRecord()
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

def ps_job(ps_id, predict_service):
    signal.signal(signal.SIGINT, soft_exit)
    ps_node = ps.ParameterServer(ps_id, predict_service)
    ps_node.run()

def cn_job(cn_id, start, length, predict_service, background_uploading):
    signal.signal(signal.SIGINT, soft_exit)
    cn_node = cn.ComputingNode(cn_id, start, length, predict_service, background_uploading)
    elapsed_time = timeit.Timer(cn_node.run).timeit(number=1)
    print "cn_node %d : %f sec" % ((cn_id), elapsed_time)
    #cn_node.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Checker')
    parser.add_argument("-p", "--predict", type=bool, help="enable predict service", default=False)
    parser.add_argument("-b", "--background", type=bool, help="upload parameters in background", default=False)
    args = parser.parse_args()

    ps_num = len(cluster_spec['ps'])
    cn_num = len(cluster_spec['cn'])
    ps_processes = list()
    cn_processes = list()

    # create parameter servers
    for i in range(ps_num):
        process = multiprocessing.Process(target=ps_job, args=(i,args.predict))
        process.start()
        ps_processes.append(process)

    # create computing nodes
    training_set_size = 20000
    length = training_set_size / cn_num
    for i in range(cn_num):
        process = multiprocessing.Process(target=cn_job, args=(i, i*length, length, args.predict, args.background))
        process.start()
        cn_processes.append(process)

    signal.signal(signal.SIGINT, sig_handler)

    # wait for training is done
    for i in range(cn_num):
        cn_processes[i].join() 
    kill_child_processes()
