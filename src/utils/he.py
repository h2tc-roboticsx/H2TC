import os, json
import pytz
from datetime import datetime
from enum import Enum


'''
Constants and Enumeration

'''

class Hand(Enum):
    '''
    Enumeration for hands: LEFT and RIGHT

    '''
    
    LEFT = 0,
    RIGHT = 1

    @classmethod
    def all(cls):
        ''' return all enumerations defined in Hand '''
        
        return (cls.LEFT, cls.RIGHT)

# raw file names for LEFT and RIGHT hands
RAW = {
    Hand.LEFT : 'P1L.csv',
    Hand.RIGHT : 'P1R.csv'
}

# output file names for LEFT and RIGHT hands
FORMAT_FILENAMES = {
    Hand.LEFT : 'left_hand_pose.csv',
    Hand.RIGHT : 'right_hand_pose.csv'
}

# meta data file names for LEFT and RIGHT hands
META = {
    Hand.LEFT : 'P1LMeta.json',
    Hand.RIGHT : 'P1RMeta.json'
}

FPS = 120 # frame per second for HE


'''
Formating

'''

def format(fdir, hand):
    '''
    Formate data and remove unnecessary entries

    Returns:
        list: [[timestamp, hand pose], # header
               [values]] # values of a frame

    Args:
        fdir (string): the directory of HE data
        hand (Hand): LEFT or RIGHT hand to process

    '''
    
    # a detailed explanation is in https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md/#gloves-hands-pose
    fpath = os.path.join(fdir, RAW[hand]) # path to the raw data of the hand
    # load raw data to a list
    with open(fpath, 'r') as f: frames = f.readlines()
    # remove all space ' ' from the raw data
    frames = [frame.replace(' ', '') for frame in frames]
    # separate headers by ','
    headers = frames[0].split(',')
    ts_idx = headers.index('Timecode(device)') # the column index of the device timecode
    data_start_idx = headers.index('hand_x') # the column index from which the hand pose data presents

    # create headers for the formatted data
    # timestamp + hand pose data headers (as raw file)
    data = [['timestamp'] + headers[data_start_idx:]]

    # load meta data (json)
    meta = open(os.path.join(fdir, META[hand]), 'r')
    meta = json.load(meta)
    # get recording date from meta data
    date = meta['startDate']

    # iterate every row in the raw file from the second row to exclude the header
    for frame in frames[1:]:
        # separate data for each header by ','
        frame = frame.split(',')
        # convert HE timestamp to UNIX timestamp
        ts = ts_to_unix(date, frame[ts_idx])
        # concatenate timestamp and hand pose data as a list
        data += [[ts] + frame[data_start_idx:]]
    return data


def ts_to_unix(date, ts):
    '''
    Convert HE timestamp to UNIX timestamp

    Returns:
    int: UNIX timestamp in nanoseconds
    
    Args:
    date (string): string of the recording date
    ts (string): HE timestamp string in a form of HH:mm:ss:{FPS numbering}

    '''
    
    # year, month, day from date
    y, m, d = [int(d) for d in date.split('-')]
    # hour, minute, second, FPS numbering
    h, min, s, ss = int(ts[:-7]), int(ts[-7:-5]), int(ts[-5:-3]), int(ts[-3:])
    # create datetime instance from the parsed date
    dt = datetime(y, m, d, h, min, s)
    # the data was recorded in a timezone of Shanghai
    tz = pytz.timezone('Asia/Shanghai')
    # set the timezone of datetime to Shanghai
    dt = tz.localize(dt)
    # convert FPS numbering to nanoseconds
    # FPS numbering / FPS * 1e9
    ts = int((dt.timestamp() + ss/FPS) * 1e9)
    return ts


'''
Verification

'''

def verify_frame_integrity(take_path, hand, length, tolerance):
    '''
    Raise an exception if the frames drop more than a threshold
    
    Returns:
        None
    
    Args:
        take_path (string): path to the take directory
        hand (Hand): 
        length (float): 
        tolerance (float): 
    
    '''
    
    total = length * FPS
    fname = os.path.join(take_path, RAW[hand])
    with open(fname, 'r') as f: frames = f.readlines()
    num_frames = len(frames)-1
    drop_rate = float(total - num_frames) / total
    if abs(drop_rate) > tolerance:
        raise Exception("Hand {} drop {:.2%} frames exceeding the tolerance {:.2%}"
                        .format(hand, drop_rate, tolerance))
