import numpy as np
import glob
import pickle
import os

from typing import Dict
from typing import Tuple

from PathFilter import PathFilter

class MixamoObjects(object):
    joints = [
    "mixamorig:Hips"
    ,"mixamorig:LeftUpLeg"
    ,"mixamorig:RightUpLeg"
    ,"mixamorig:Spine"

    ,"mixamorig:LeftLeg"
    ,"mixamorig:RightLeg"
    ,"mixamorig:Spine1"

    ,"mixamorig:LeftFoot"
    ,"mixamorig:RightFoot"
    ,"mixamorig:Spine2"

    ,"mixamorig:LeftToeBase"
    ,"mixamorig:RightToeBase"
    ,"mixamorig:Neck"

    ,"mixamorig:LeftToe_End"
    ,"mixamorig:RightToe_End"

    ,"mixamorig:Head"
    ,"mixamorig:LeftShoulder"
    ,"mixamorig:RightShoulder"

    ,"mixamorig:LeftArm"
    ,"mixamorig:RightArm"
    ,"mixamorig:LeftForeArm"
    ,"mixamorig:RightForeArm",
    
    "mixamorig:LeftHand",
    "mixamorig:RightHand",
    
    'mixamorig:LeftHandIndex1',
    'mixamorig:LeftHandIndex2',
    'mixamorig:LeftHandIndex3',
    'mixamorig:LeftHandMiddle1',
    'mixamorig:LeftHandMiddle2',
    'mixamorig:LeftHandMiddle3',
    'mixamorig:LeftHandPinky1',
    'mixamorig:LeftHandPinky2',
    'mixamorig:LeftHandPinky3',
    'mixamorig:LeftHandRing1',
    'mixamorig:LeftHandRing2',
    'mixamorig:LeftHandRing3',
    'mixamorig:LeftHandThumb1',
    'mixamorig:LeftHandThumb2',
    'mixamorig:LeftHandThumb3',
    
    'mixamorig:RightHandIndex1',
    'mixamorig:RightHandIndex2',
    'mixamorig:RightHandIndex3',
    'mixamorig:RightHandMiddle1',
    'mixamorig:RightHandMiddle2',
    'mixamorig:RightHandMiddle3',
    'mixamorig:RightHandPinky1',
    'mixamorig:RightHandPinky2',
    'mixamorig:RightHandPinky3',
    'mixamorig:RightHandRing1',
    'mixamorig:RightHandRing2',
    'mixamorig:RightHandRing3',
    'mixamorig:RightHandThumb1',
    'mixamorig:RightHandThumb2',
    'mixamorig:RightHandThumb3'
    
    
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
