import time
import datetime as dt

def sleep(s):
    time.sleep(s)

def timestamp():
    return time.time_ns()

def stamp(msg):
    print('{}:\t{}'.format(msg, timestamp()))

def display_format(ts):
    date = dt.datetime.fromtimestamp(int(ts/1e9))
    tstr = date.strftime('%m.%d %H:%M:%S')
    tstr += ':{:03d}'.format(int(ts%1e9/1e6))
    return tstr
    
def ts_he_to_unix(ts):
    ts = '17:14:17:066'
    
