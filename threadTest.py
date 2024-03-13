import time, threading

WAIT_SECONDS = 1

def foo():
    print(time.ctime())
    threading.Timer(WAIT_SECONDS, foo).start()
    
foo()
