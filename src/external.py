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
