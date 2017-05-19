import time

class StopWatch:
    def __init__(self):
        self.stamp = time.time()
        self.StampSet = dict()

    def accumulate(self, key, refresh=True):
        if key in self.StampSet:
            self.StampSet[key] += time.time() - self.stamp
        else:
            self.StampSet[key] = time.time() - self.stamp
        if refresh:
            self.reset()    

    def reset(self):
        self.stamp = time.time()

    def present(self):
        print "\n"
        for key in iter(self.StampSet):
            print "%-20s\t%-2.5fsec" % (key, self.StampSet[key])
