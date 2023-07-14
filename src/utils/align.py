import pandas as pd
import numpy as np
from datetime import datetime
from bisect import bisect_left
import random
import json
import os
import glob


def align(opt_path, rgbd0_path, rgbd1_path, rgbd2_path, event_path, opti_paths, he_paths):
    '''
    This function is to be called by process.py to align timestamps of different streams
    INPUT:
        opt_path: output path for saving the alignment.json file
        rgbd0_path: rgbd0 camera timestamp file path
        rgbd1_path: rgbd1 camera timestamp file path
        rgbd2_path: rgbd2 camera timestamp file path
        event_path: event camera timestamp file path
        opti_paths: optitrack timestamp file paths (the number of files equal to the number of tracked objects)
        he_paths: hand engine timestamp file paths (left and right hand files)
    '''

    # read timestamps from the csv file of each data stream
    rgbd0_tss = pd.read_csv(rgbd0_path)['nanoseconds'].values.tolist()
    rgbd1_tss = pd.read_csv(rgbd1_path)['nanoseconds'].values.tolist()
    rgbd2_tss = pd.read_csv(rgbd2_path)['nanoseconds'].values.tolist()
    
    event_tss = pd.read_csv(event_path)['nanoseconds'].values.tolist()
    
    he_tss = {}
    for path in he_paths:
        tss = pd.read_csv(path)['timestamp'].values.tolist()
        hand = path.split('/')[-1].split('.')[0]
        he_tss[hand] = tss
        
    opti_tss = {}
    for path in opti_paths:
        obj = path.split('/')[-1].split('.')[0]
        tss = pd.read_csv(path)['timestamp'].values.tolist()
        opti_tss[obj] = tss

 
    # call the sub-function to align timestamps of mutiple data streams       
    align_multimodal_with_ts(opt_path, rgbd0_tss, rgbd1_tss, rgbd2_tss, event_tss, he_tss, opti_tss)
    

def find_nearest(sorted_array, value, threshold=16666667*4):
    '''
    Find the nearest timestamp in another modality of a given timestamp.
    Threshold is set to 1/60*1000000000 nano seconds
    INPUT:
        sorted_array: sorted timestamp array
        value: the query timestamp, which is to compare against all values in the sorted_array to find its nearest timestamp
        threshold: threshold between two compared timestamps. There must exist a nearest timestamp, but this timestamp can't
        be larger or smaller than the query timestamp by a threshold.
    OUTOUT:
        return the nearest timestamp if found, otherwise return None
    '''

    # the following is the implementation of a standard binary search algorithm for finding the 
    # nearest neighbor
    if value >= sorted_array[-1]:
        if (value - sorted_array[-1]) > threshold:
            return None
        else:
            return sorted_array[-1]
    elif value <= sorted_array[0]:
        if (sorted_array[0] - value) > threshold:
            return None
        else:
            return sorted_array[0]
            
    pos = bisect_left(sorted_array, value)   

    before = sorted_array[pos - 1]
    after = sorted_array[pos]

    after_diff = after - value
    before_diff = value - before

    if after_diff < before_diff and after_diff < threshold:
        return after
    elif before_diff <= after_diff and before_diff < threshold:
        return before
    else:
        return None
        
def align_multimodal_with_ts(save_fpath, rgbd0_tss, rgbd1_tss, rgbd2_tss, event_tss, he_tss, opti_tss):
    '''
    This is sub-function, which will be called by the function align().
    This function aligns the timestamps of multiple data streams
    INPUT:
        save_fpath: save path of the alignment.json file
        rgbd0_tss: timestamp array of the rgbd0 camera
        rgbd1_tss: timestamp array of the rgbd1 camera
        rgbd2_tss: timestamp array of the rgbd2 camera
        event_tss: timestamp array of the event camera
        he_tss: timestamp array of the hand engine 
        opti_tss: timestamp array of the optitrack data
    '''


    frame_tss = {}
    # get the only timestamps from the array
    # rgbd0_tss = rgbd0_tss[1:]
    # rgbd1_tss = rgbd1_tss[1:]
    # rgbd2_tss = rgbd2_tss[1:]

    # sort the timestamp array
    rgbd0_tss = sorted(rgbd0_tss)
    rgbd1_tss = sorted(rgbd1_tss)
    rgbd2_tss = sorted(rgbd2_tss)
    event_tss = sorted(event_tss)

    for k, v in he_tss.items():
        he_tss[k] = sorted(v)

    for k, v in opti_tss.items():
        opti_tss[k] = sorted(v)

    # start to align
    for frame_idx, curr_rgbd0_ts in enumerate(rgbd0_tss):
        curr_tss = {'rgbd0': curr_rgbd0_ts}
        curr_tss['rgbd1'] = find_nearest(rgbd1_tss, curr_rgbd0_ts)
        curr_tss['rgbd2'] = find_nearest(rgbd2_tss, curr_rgbd0_ts)
        curr_tss['event'] = find_nearest(event_tss, curr_rgbd0_ts)
        for hand_idx, tss in he_tss.items():
            curr_tss[hand_idx] = find_nearest(tss, curr_rgbd0_ts)

        for opti_idx, tss in opti_tss.items():
            curr_tss[opti_idx] = find_nearest(tss, curr_rgbd0_ts)

        frame_tss[frame_idx] = curr_tss
    # save the alignment file - alignment.json
    with open(save_fpath, 'w') as f:
        json.dump(frame_tss, f, indent=4)
