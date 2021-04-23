from contextlib import contextmanager
from threading import Lock


@contextmanager
def locked(lock: Lock):
    _locked = False
    try:
        _locked = lock.acquire(blocking=False)
        yield _locked
    finally:
        if _locked:
            lock.release()
