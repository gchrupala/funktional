import sys
from contextlib import contextmanager

training = False


@contextmanager
def context(**kwargs):
    current = dict((k, getattr(sys.modules[__name__], k)) for k in kwargs)
    for k,v in kwargs.items():
        setattr(sys.modules[__name__], k, v)
    yield
    for k,v in current.items():
        setattr(sys.modules[__name__], k, v)
        
