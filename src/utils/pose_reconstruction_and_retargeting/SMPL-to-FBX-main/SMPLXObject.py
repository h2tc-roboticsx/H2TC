import numpy as np
import glob
import pickle
import os

from typing import Dict
from typing import Tuple

from PathFilter import PathFilter

class SMPLXObjects(object):
    joints = [
     "pelvis"
    ,"left_hip"
    ,"right_hip"
    ,"spine1"

    ,"left_knee"
    ,"right_knee"
    ,"spine2"

    ,"left_ankle"
    ,"right_ankle"
    ,"spine3"

    ,"left_foot"
    ,"right_foot"
    ,"neck"

    ,"left_collar"
    ,"right_collar"

    ,"head"
    ,"left_shoulder"
    ,"right_shoulder"

    ,"left_elbow"
    ,"right_elbow"
    ,"left_wrist"
    ,"right_wrist",
    
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
    
    
    ]
    def __init__(self, read_path):
        self.files = {}

        # For AIST naming convention
        #paths = PathFilter.filter(read_path, dance_genres=["gBR"],  dance_types=["sBM"], music_IDs=["0"])
        paths = PathFilter.filter(read_path, dance_genres=None,  dance_types=None, music_IDs=None)
        for path in paths:
            path = path.replace("\\","/")
            filename = path.split("/")[-1]
            
            data = np.load(path)
            smpl_body = data['pose_body'] # seq_len*63
            smpl_hand = data['pose_hand'] # seq_len*[45+45]
            smpl_pose = np.concatenate((data['root_orient'], smpl_body,smpl_hand),axis=1) # n * (1*3 + 21*3 + 15 + 15)
            self.files[filename] = {"smpl_poses":smpl_pose,
                                    # "smpl_trans":data["smpl_trans"] / (data["smpl_scaling"][0]*100)}
                                    "smpl_trans":data["trans"],
                                    "smpl_orient":data['root_orient']}
        self.keys = [key for key in self.files.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx:int) -> Tuple[str, Dict]:
        key = self.keys[idx]
        return key, self.files[key]
