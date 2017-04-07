import time
import logging

FORMAT = '%(message)s'

class Timer(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format=FORMAT)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs *1000
        print self.secs
        return self.secs
