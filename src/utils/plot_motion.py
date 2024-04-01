from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json

from src.utils.optitrack import get_ts_t_matrix

'''
defined finger bone length
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


def plot_dots(ax, x, y, z):
	'''
	Plot the trajectory of object with markers using dot
	INPUT:
		ax: canvas ax
		x: x position of the object
		y: y position of the object
		z: z position of the object
	'''
	ax.scatter(x, y, z)


def plot_lines(ax, start_point, end_point, color):
	'''
	Plot bone between two finger joints as a line
	INPUT:
		ax: canvas ax
		start_point: start point of the line
		end_point: end point of the line
		color: color used for this line
	'''
	x = np.array([start_point[0], end_point[0]])
	y = np.array([start_point[1], end_point[1]])
	z = np.array([start_point[2], end_point[2]])

	ax.plot(x, y, z, color=color)



def plot_left_hand(ax, opti_t_matrix, records, color_list):
	'''
	Plot left hand
	INPUT:
		ax: canvas ax
		opti_t_matrix: optitrack's 4x4 converted transformation matrix
		records: hand engine euler angle data
		color_list: a list of color to be used for plotting the hand
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

		pos_finger_0 = pos_hand
		t_finger = []
		for i, color in zip(range(n_start, n), color_list):
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

			# plot the bone that connects this joint and the last one
			plot_lines(ax, pos_finger_0, pos_finger, color)
			pos_finger_0 = pos_finger

	return joint_positions

			
def plot_right_hand(ax, opti_t_matrix, records, color_list):
	'''
	Plot right hand
	INPUT:
		ax: canvas ax
		opti_t_matrix: optitrack's 4x4 converted transformation matrix
		records: hand engine's euler angle data
		color_list: a list of color to be used for plotting the hand
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

		pos_finger_0 = pos_hand
		t_finger = []
		for i, color in zip(range(n_start, n), color_list):

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

			# plot the bone connecting this joint and the last one 
			plot_lines(ax, pos_finger_0, pos_finger, color)
			pos_finger_0 = pos_finger
			
	return joint_positions

def plot_hand_pose_and_motion(data_root, local_sys_id, rotate_right_hand, rotate_left_hand):
	'''
	This function will be called by process.py to plot both hands and object's trajectory if 
	the object is attached with markers
	INPUT:
		data_root: data root of the current take
        local_sys_id: the id of the origin (ids should be either 0, 1, or 2)
		rotate_right_hand: whether to rotate the right hand or not. -1 means not to rotate. Other options 
                           are 90 and 180, which is to rotate 90 or 180 degrees
        rotate_left_hand: whether to rotate the left hand or not. -1 means not to rotate. Other option is 45,
                           which is to rotate 45 degrees
	'''


	save_dir = os.path.join(data_root, 'processed/hand_motion')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

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

	# color lists for plotting the right and left hands
	right_color = ['red', 'green', 'blue', 'yellow']
	left_color = ['cyan', 'purple', 'black', 'pink']


	# get object's trajectory data if the object was attached with markers during data collection
	object_stream_name = None
	all_stream_names = list(frame_tss_dict['0'].keys())
	for s in all_stream_names:
		if s.endswith('_motion') and 'sub' not in s:
			object_stream_name = s
			break

	if object_stream_name != None:
		print(object_stream_name)
		object_records = pd.read_csv(os.path.join(data_root, 'processed', object_stream_name+'.csv'))


	# start to plot
	fig = plt.figure()
	ax = Axes3D(fig)

	# dictionary to store joint positions calculated using forward kenematics
	# key is the timestamp, and value is a list of joint positions
	left_hand_joint_position_dict = {}
	right_hand_joint_position_dict = {}
	# object_position_dict = {}
 
	for frame_idx, aligned_tss in frame_tss_dict.items():
		ax.set_xlim3d(-0.5, 2.5)
		ax.set_ylim3d(0, 2.5)
		ax.set_zlim3d(-0.5, 2.5)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.view_init(elev=30, azim=180, vertical_axis='y')
		# ax.view_init(elev=10, azim=270, vertical_axis='y')
    
		# plot the left hand
		if aligned_tss['left_hand_pose'] != None and aligned_tss['sub1_left_hand_motion'] != None:
			curr_left_hand_data = left_records.loc[left_records['timestamp'] == aligned_tss['left_hand_pose']]
			curr_left_hand_t_matrix = np.array(left_hand_ts_t_matrix[aligned_tss['sub1_left_hand_motion']])
			curr_left_hand_joint_positions = plot_left_hand(ax, curr_left_hand_t_matrix, curr_left_hand_data, left_color)
			# print("timestamp type: ", type(left_records['timestamp']))
			left_hand_joint_position_dict[aligned_tss['left_hand_pose']] = curr_left_hand_joint_positions

		# # plot the right hand
		if aligned_tss['right_hand_pose'] != None and aligned_tss['sub1_right_hand_motion'] != None:
			curr_right_hand_data = right_records.loc[right_records['timestamp'] == aligned_tss['right_hand_pose']]
			curr_right_hand_t_matrix = np.array(right_hand_ts_t_matrix[aligned_tss['sub1_right_hand_motion']])
			curr_right_hand_joint_positions = plot_right_hand(ax, curr_right_hand_t_matrix, curr_right_hand_data, right_color)
			right_hand_joint_position_dict[aligned_tss['right_hand_pose']] = curr_right_hand_joint_positions

		# plot the object if it was tracked by the optitrack system
		if object_stream_name != None and aligned_tss[object_stream_name] != None:
			curr_object_data = object_records.loc[object_records['timestamp'] == aligned_tss[object_stream_name]]
			curr_object_data = curr_object_data.values.tolist()[0]
			plot_dots(ax, curr_object_data[1], curr_object_data[2], curr_object_data[3])

		save_joint_position_to_json(os.path.join(data_root, 'processed', 'left_hand_joint_positions.json'), left_hand_joint_position_dict)	
		save_joint_position_to_json(os.path.join(data_root, 'processed', 'right_hand_joint_positions.json'), right_hand_joint_position_dict)

		# set the viewing angle of the plot. A 30 degree top-down view with Y axis up and Z axis forward.
		# save hand pose and motion (plus object if tracked) in images of jpg format
		plt.savefig(os.path.join(save_dir, '{}.jpg'.format(frame_idx)))
		# plt.show() can be commented out to interactively view the hand pose and orientation.
		# plt.show()
		# plt.clf()
		ax.cla()
