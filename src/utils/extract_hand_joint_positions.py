from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json

from src.utils.optitrack import get_ts_t_matrix
from argparse import ArgumentParser



'''
default finger bone lengths
users can specify customized bone lengths to adapt to their specific scenarios and needs
'''
finger_long = {}
finger_long['thumb'] = [None, 0.25, 0.11, 0.06]
finger_long['index'] = [0.34, 0.15, 0.08, 0.06]
finger_long['middle'] = [0.33, 0.15, 0.10, 0.07]
finger_long['ring'] = [0.31, 0.13, 0.10, 0.06]
finger_long['pinky'] = [0.3, 0.08, 0.06, 0.06]

for k, v in finger_long.items():
	finger_long[k] = [vv if vv is not None else None for vv in v]


def save_joint_position_to_json(filepath, data):
	with open(filepath, 'w') as f:
		json.dump(data, f)


def extract_left_hand_joint_positions(opti_t_matrix, records):
	'''
	Extract the joint positions of the left hand
	INPUT:
		opti_t_matrix: optitrack's 4x4 converted transformation matrix
		records: hand engine euler angle data
	'''

	
	# rotation for aligning righted-handed coordinate system with the our throw-catch zone coordinate system
	rotX = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # x -180
	rotY = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]) # y -90
	
	# start position to plot the hand
	pos_hand = np.dot(opti_t_matrix, np.array([0,0,0,1]))[:3]
	
	# list that stores the joint positions calculated using forward kinematics
	joint_positions = [pos_hand.tolist()]
	# use forward kinematics to plot the hand
	for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
		if finger == 'thumb': n_start = 1 # the first element of thumb bone length list is None, so starting from 1
		else: n_start = 0
		n = 4

		t_finger = []
		for i in range(n_start, n):
			r = R.from_euler('XYZ', [records['{}_{:0>2d}_x'.format(finger, i)].values[0],
									 records['{}_{:0>2d}_y'.format(finger, i)].values[0],
									 records['{}_{:0>2d}_z'.format(finger, i)].values[0]], degrees=True)
			r = r.as_matrix()
			t_r = np.concatenate([np.concatenate([r, np.array([0, 0, 0]).reshape(3, 1)], axis=1),
								np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
			t_p = np.eye(4)
			t_p[0, -1] = finger_long[finger][i]
			t = np.matmul(t_r, t_p)
			t_finger.append(t)
			pos_finger = np.array([0, 0, 0, 1])

			# inverse
			for t in t_finger[::-1]:
				pos_finger = np.dot(t, pos_finger)

			# add rotation to the finger joints			
			pos_finger = np.dot(rotX, pos_finger)
			pos_finger = np.dot(rotY, pos_finger)
			# apply the converted optitrack transformation matrix (already expressed in the our throw-catch zone system) 
			# to obtain the correct x,y,z positions of the joint
			pos_finger = np.dot(opti_t_matrix, pos_finger)[:3]
			
			# add this joint position to the list
			joint_positions.append(pos_finger.tolist())

	return joint_positions

			
def extract_right_hand_joint_positions(opti_t_matrix, records):
	'''
	Extract the joint positions of the right hand
	INPUT:
		opti_t_matrix: optitrack's 4x4 converted transformation matrix
		records: hand engine's euler angle data
	'''

	# rotation for aligning the coordinate system with the our throw-catch zone coordinate system
	rotY = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]) # y 90

	# start position to plot the hand
	pos_hand = np.dot(opti_t_matrix, np.array([0,0,0,1]))[:3]

	# list that stores the joint positions calculated using forward kinematics
	joint_positions = [pos_hand.tolist()]

	# use forward kinematics to plot the hand
	for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
		if finger == 'thumb': n_start = 1 # the first element of thumb bone length list is None, so starting from 1
		else: n_start = 0
		n = 4
		
		t_finger = []
		for i in range(n_start, n):

			r = R.from_euler('XYZ', [records['{}_{:0>2d}_x'.format(finger, i)].values[0],
									 records['{}_{:0>2d}_y'.format(finger, i)].values[0],
									 records['{}_{:0>2d}_z'.format(finger, i)].values[0]], degrees=True)
			r = r.as_matrix()
			t_r = np.concatenate([np.concatenate([r, np.array([0, 0, 0]).reshape(3, 1)], axis=1),
								np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
			t_p = np.eye(4)
			t_p[0, -1] = -finger_long[finger][i]
			t = np.matmul(t_r, t_p)
			t_finger.append(t)
			pos_finger = np.array([0, 0, 0, 1])

			# inverse
			for t in t_finger[::-1]:
				pos_finger = np.dot(t, pos_finger)

			# apply rotation to the finger joint for coordinate system alignment
			pos_finger = np.dot(rotY, pos_finger)
			
			# apply the converted optitrack transformation matrix (already expressed in the our throw-catch zone system) 
			# to obtain the correct x,y,z positions of the joint
			pos_finger = np.dot(opti_t_matrix, pos_finger)[:3]

			# add this joint position
			joint_positions.append(pos_finger.tolist())

	return joint_positions

def main(data_root, local_sys_id, rotate_right_hand, rotate_left_hand):
	'''
	This function extracts the joint positions of both hands based on captured euler angles
	and defined bone lengths (users can use the default bone lengths or specify their customized
	bone lengths at the top of this script)
	INPUT:
		data_root: data root of the current take
        local_sys_id: the id of the origin (ids should be either 0, 1, or 2)
	rotate_right_hand: whether to rotate the right hand or not. -1 means not to rotate. Other options 
                           are 90 and 180, which is to rotate 90 or 180 degrees
        rotate_left_hand: whether to rotate the left hand or not. -1 means not to rotate. Other option is 45,
                           which is to rotate 45 degrees
	'''

	with open(os.path.join(data_root, 'processed/alignment.json'), 'r') as f:
		frame_tss_dict = json.load(f)

	# get transformation matrix of both hands
	opti_df = pd.read_csv(os.path.join(data_root, 'raw/optitrack.csv'), delimiter=' ')
	right_hand_ts_t_matrix = get_ts_t_matrix(opti_df, 116, local_sys_id, 'global', rotate_right_hand=rotate_right_hand)
	left_hand_ts_t_matrix = get_ts_t_matrix(opti_df, 117, local_sys_id, 'global', rotate_left_hand=rotate_left_hand)

	# get euler angle data of the right hand
	right_records = pd.read_csv(os.path.join(data_root, 'processed/right_hand_pose.csv'))
	right_records = right_records.rename(columns=dict([(k, k.lstrip()) for k in right_records.columns]))

	# get euler angle data of the left hand
	left_records = pd.read_csv(os.path.join(data_root, 'processed/left_hand_pose.csv'))
	left_records = left_records.rename(columns=dict([(k, k.lstrip()) for k in left_records.columns]))

	# dictionary to store joint positions calculated using forward kenematics
	# key is the timestamp, and value is a list of joint positions
	left_hand_joint_position_dict = {}
	right_hand_joint_position_dict = {}
	
 
	for frame_idx, aligned_tss in frame_tss_dict.items():
    
		# extract left hand joint positions
		if aligned_tss['left_hand_pose'] != None and aligned_tss['sub1_left_hand_motion'] != None:
			curr_left_hand_data = left_records.loc[left_records['timestamp'] == aligned_tss['left_hand_pose']]
			curr_left_hand_t_matrix = np.array(left_hand_ts_t_matrix[aligned_tss['sub1_left_hand_motion']])
			curr_left_hand_joint_positions = extract_left_hand_joint_positions(curr_left_hand_t_matrix, curr_left_hand_data)
			left_hand_joint_position_dict[aligned_tss['left_hand_pose']] = curr_left_hand_joint_positions

		# extract right hand joint positions
		if aligned_tss['right_hand_pose'] != None and aligned_tss['sub1_right_hand_motion'] != None:
			curr_right_hand_data = right_records.loc[right_records['timestamp'] == aligned_tss['right_hand_pose']]
			curr_right_hand_t_matrix = np.array(right_hand_ts_t_matrix[aligned_tss['sub1_right_hand_motion']])
			curr_right_hand_joint_positions = extract_right_hand_joint_positions(curr_right_hand_t_matrix, curr_right_hand_data)
			right_hand_joint_position_dict[aligned_tss['right_hand_pose']] = curr_right_hand_joint_positions


		
		save_joint_position_to_json(os.path.join(data_root, 'processed', 'left_hand_joint_positions.json'), left_hand_joint_position_dict)	
		save_joint_position_to_json(os.path.join(data_root, 'processed', 'right_hand_joint_positions.json'), right_hand_joint_position_dict)


if __name__ == '__main__':
	argparser = ArgumentParser()
	argparser.add_argument('--data_root', type=str, 
	                       help='the root folder path of the target take')
	
	args = argparser.parse_args()

	take = args.data_root.split('/')[-1]

	# determine the coordinate system ID according the take ID
	# check the document (https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md/#the-coordinate-system-id) for more details about our coordinate system 
	if int(take) < 2889:
	    local_sys_id = '0'
	elif int(take) < 9789:
	    local_sys_id = '1'
	else:
	    local_sys_id = '2'

	# the right hand in takes from 520-1699 needs to apply an additional rotation
	# 90 degrees along the Y axis for takes from 520-1559 
	# and 180 degrees along the Y axis for takes from 1560-1699
	# check the document (https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md/#note) for more details about the extra transformation
	if 519 < int(take) and int(take) <= 1559:
	    rotate_right_hand = 90
	elif 1559 < int(take) and int(take) <= 1699:
	    rotate_right_hand = 180
	else:
	    rotate_right_hand = -1

	# the left hand in takes from 1040-1559 needs to apply an additional rotation 
	# 45 degrees along the Y axis 
	if 1040 <= int(take) and int(take) <= 1559:
	    rotate_left_hand = 45
	else:
	    rotate_left_hand = -1

	main(args.data_root, local_sys_id, rotate_right_hand, rotate_left_hand)






