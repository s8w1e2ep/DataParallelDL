import threading
from thrift_conn import init_server
import os
import signal
import sys

def signal_handler(signal, frame):
    sys.exit()

class Dispatcher(object):
    def __init__(self, sharedstatus, sharedmodel):
        self.lock = threading.Lock()
        self.status = sharedstatus
        self.model = sharedmodel

    def upload(self, model):
        try:
            self.status += 1
            self.model.assign(model)
            a = self.status
        except Exception as e:
            print e.message
        return a

    def getGlobalStatus(self):
        return self.status

class Integer(object) :
    def __init__(self, val=0) :
        self._val = int(val)
    def __add__(self, val) :
        if type(val) == Integer :
            return Integer(self._val + val._val)
        return self._val + val
    def __iadd__(self, val) :
        self._val += val
        return self
    def __str__(self) :
        return str(self._val)
    def __int__(self) :
        return int(self._val)
    def __repr__(self) :
        return self._val

class String(object) :
    def __init__(self, val) :
        self._val = str(val)
    def __str__(self) :
        return str(self._val)
    def __repr__(self) :
        return self._val
    def assign(self, val):
        self._val = val
    def get(self):
        return self._val

class ParameterServer(threading.Thread):
    def __init__(self):
        self.ip = "127.0.0.1"
        self.port = 50001
        self.status = Integer(0)
        self.model = String("12345")
        handler = Dispatcher(self.status, self.model)
        self.server = init_server(self.ip, self.port, handler)
        super(ParameterServer, self).__init__() 

    def run(self):
        print 'Start parameter server(%s)' % self.ip
        self.server.serve()

if __name__ == "__main__":
    ps = ParameterServer()
    ps.daemon = True
    ps.start()
  
    signal.signal(signal.SIGINT, signal_handler)
    import time
    while 1:
        print str(ps.status), ps.model
        time.sleep(1)
