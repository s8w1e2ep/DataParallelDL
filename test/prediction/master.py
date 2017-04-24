import os
import sys
import signal, psutil
import time
import numpy as np
import multiprocessing
from thrift_conn import init_conn
from thrift_conn import init_server
from predictor import predictor
from random import *
from math import sqrt
import thriftpy
from thriftpy.rpc import make_server
from thriftpy.rpc import make_client

def sig_handler(signum, frame):
    kill_child_processes()

def kill_child_processes():
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

class Dispatcher(object):
    def __init__(self, dim):
        self.dim = dim
        self.predictor = predictor(dim[0], dim[1], dim[2])
        stime = time.time()
        self.arrive_record = [stime for i in range(dim[2])]
        self.elapsed_record = [0. for i in range(dim[2])]
        self.pnext = np.argmax(self.predictor.predict([self._format_data()]))
        self.record = list()
        self.hit = 0
        self.count = 0
        import threading
        self.lock = threading.Lock()

    def _format_data(self):
        ts_data = normalize(self.arrive_record)
        et_data = stdev_normalize(self.elapsed_record)
        mergedata = ts_data + et_data
        return mergedata

    def ping(self, cn_id):
        self.lock.acquire()
        current_time = time.time()
        self.count += 1
        data = self._format_data()
        label = np.zeros(self.dim[2])
        label[cn_id] = 1
        self.predictor.train([data], [label])
        self.record.append(([data], [label]))
        
        self.elapsed_record[cn_id] = current_time - self.arrive_record[cn_id]
        self.arrive_record[cn_id] = current_time
        if self.pnext == cn_id:
            self.hit += 1
        
        dat = self._format_data()
        self.pnext = np.argmax(self.predictor.predict([data]))
        self.lock.release()

    def getHitRate(self):
        for component in self.record:
            print component[0],'\t',component[1]
        return float(self.hit) / self.count

def ps_job(ps_id, cluster_spec, dim): 
    signal.signal(signal.SIGINT, soft_exit)
    ping_thrift = thriftpy.load("ping.thrift", module_name="ping_thrift")
    handler = Dispatcher(dim) 
    server = make_server(ping_thrift.PingService, handler,  '127.0.0.1', 6000)
    print "start Server(%s:%d)" % ('127.0.0.1', 6000)
    server.serve()

def cn_job(cn_id, cluster_spec, times):
    signal.signal(signal.SIGINT, soft_exit)
    ping_thrift = thriftpy.load("ping.thrift", module_name="ping_thrift")
    conn = make_client(ping_thrift.PingService, '127.0.0.1', 6000)
    for i in range(times):
        compute_pi(100000)
        conn.ping(cn_id)

def fib(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return fib(n-1)+fib(n-2)

def compute_pi(n):
    inside=0
    for i in range(0,n):
        x=random()
        y=random()
        if sqrt(x*x+y*y)<=1:
            inside+=1
    pi=4*inside/n
    return pi

def stdev_normalize(x):
   # (x - Mean) / Deviation
    stdev = np.std(x)
    mean = np.mean(x)
    stdev = 1 if stdev == 0 else stdev
    print mean, stdev
    return map(lambda i: (i-mean) * 100 / stdev * 0.01, x)

def normalize(x):
    # (timeseries-timeseries.min())/(timeseries.max()-timeseries.min())
    baseline = min(x)
    last_finish = max(x)
    diff = last_finish - baseline
    diff = 1 if diff == 0 else diff
    return map(lambda i: (i-baseline) * 100 / diff * 0.01, x)

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
        process = multiprocessing.Process(target=ps_job, args=(i, cluster_spec[machine_num], [cn_num *2 ,10 , cn_num]))
        process.start()
        ps_processes.append(process)

    time.sleep(2)

    # create computing nodes
    for i in range(cn_num):
        process = multiprocessing.Process(target=cn_job, args=(i, cluster_spec[machine_num], int(sys.argv[2])))
        process.start()
        cn_processes.append(process)

    signal.signal(signal.SIGINT, sig_handler)

    # wait for training is done
    for i in range(cn_num):
        cn_processes[i].join()

    ping_thrift = thriftpy.load("ping.thrift", module_name="ping_thrift")
    conn = make_client(ping_thrift.PingService, '127.0.0.1', 6000)
    print conn.getHitRate()
    kill_child_processes()
