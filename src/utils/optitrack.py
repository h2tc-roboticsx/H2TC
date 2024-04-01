import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

DEFAULT_FILENAME = 'optitrack.csv'
DEFAULT_FPS = 240

'''
Model id in Motive and its corresponding name
'''
OBJECTS = {
    115 : 'sub1_head',
    116 : 'sub1_right_hand',
    117 : 'sub1_left_hand',
    118 : 'sub2_head',
    101 : 'airplane',
    102 : 'round_plate',
    103 : 'apple',
    104 : 'banana',
    105 : 'hammer',
    106 : 'long_neck_bottle',
    107 : 'wristwatch',
    108 : 'bowl',
    109 : 'block',
    110 : 'cylinder',
    111 : 'cube',
    112 : 'torus',
    113 : 'wrench',
    114 : 'leopard',
    119 : 'toothbrush'
}

'''
In takes captured early, right and left hand needs to rotate an extra degree
takes 520-1559: right hand needs to rotate 90 degrees
takes 1560-1699: right hand needs to rotate 180 degrees
takes 1040-1559: left hand needs to rotate 45 degrees

For takes from 0-1699, the orientation of the helmet and headband needs to be corrected if using their orientation 
(rotate along Y axis with extra 45 degrees for the headband, and -180 degrees for the helmet).
'''
rotY_90 = [[0.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [-1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]

rotY_180 = [[-1.0, 0.0, -0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]

rotY_45 = [[0.707, 0.0, 0.707, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [-0.707, 0.0, 0.707, 0.0],
           [0.0, 0.0, 0.0, 1.0]]

'''
Global 4x4 transformation matrix of the throw-catch zone's defined origin 
takes 0-2888: use origin #0
takes 2889-9788: use origin #1
takes 9789-15000: use origin #2
'''
LOCAL_SYS_T_MATRIX = {'0': np.array([[-0.99886939, -0.04535922, -0.01408667,  0.42632084],
                               [-0.04514784,  0.99886579, -0.0149642,   0.0984003 ],
                               [ 0.01474855, -0.01431195, -0.99978858,  7.67951849],
                               [ 0.,          0.,          0.,          1.        ]]),
                      '1': np.array([[-9.99963351e-01,  8.30436476e-03,  2.13574045e-03,  1.92400245e-01],
                               [ 8.31340413e-03,  9.99956270e-01,  4.25134508e-03,  6.55417571e-02],
                               [-2.10037766e-03,  4.26893351e-03, -9.99988700e-01,  2.17126483e+00],
                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
                      '2': np.array([[-0.99997146,  0.00456379,  0.00601402,  0.19729361],
                                [ 0.00454136,  0.99998255, -0.00373799,  0.06776005],
                                [-0.00603099, -0.00371067, -0.99997492,  2.48060394],
                                [ 0.,          0.,          0.,          1.        ]])
                      }

'''
Inverse matrix of the LOCAL_SYS_T_MATRIX
'''
LOCAL_SYS_T_MATRIX_INV = {'0': np.array([[-0.99887346, -0.04514823,  0.01474953,  0.31701391],
                                         [-0.04535921,  0.99887065, -0.01431137,  0.0309528 ],
                                         [-0.01408573, -0.01496482, -0.99978902,  7.68537583],
                                         [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
                          '1': np.array([[-9.99963124e-01,  8.31338829e-03, -2.10034234e-03, 1.96408675e-01],
                                         [ 8.30438079e-03,  9.99956542e-01,  4.26894456e-03, -7.64056829e-02],
                                         [ 2.13577519e-03,  4.25133477e-03, -9.99988665e-01, 2.17055065e+00],
                                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]), 
                          '2': np.array([[-0.99997154,  0.00454136, -0.00603098,  0.21194073],
                                         [ 0.00456379,  0.99998285, -0.00371057, -0.05945483],
                                         [ 0.00601403, -0.00373809, -0.99997494,  2.47960853],
                                         [ 0.        ,  0.        ,  0.        ,  1.        ]])
                          }


def read_csv(filepath):
    '''
    Read a csv file
    INPUT:
        filepath: file path of the csv file to be read
    OUTPUT:
        df: pandas dataframe of the csv data
    '''

    df = pd.read_csv(filepath, delimiter=' ')
    return df

def get_all_obj_ids(df):
    '''
    Get all objects' ids from optitrack's raw data
    INPUT:
        df: pandas dataframe of the optitrack's raw data
    OUTPUT:
        return the unique object ids in the optitrack data
    '''
    return set(df.iloc[:, 0].to_list())


def get_timestamp(df, object_id):
    '''
    Get timestamp according to object's id from optitrack's raw data
    INPUT:
        df: pandas dataframe of the optitrack's raw data
        object_id: object id 
    OUTPUT:
        return the timestamps of the object based on its id
    '''
    df = df.loc[df.iloc[:, 0] == object_id]
    return df.iloc[:, 3].to_list()


def get_t_matrix(df, object_id, local_sys_id, t_matrix_type='global', rotate_right_hand=-1, rotate_left_hand=-1, rotate_helmet_headband=False):
    '''
    Get transformation matrix from optitrack's raw data.
    This function will be called in process.py
    INPUT:
        df: pandas dataframe of the optitrack's raw data
        object_id: object id
        local_sys_id: the id of the origin (ids should be either 0, 1, or 2)
        t_matrix_type: the type of the optitrack transformation matrix. Either 'global' or 'local'. We use 'global' in the project.
                       Check our '/doc/processing_techdetails.md' -> *Difference between local and global transformation matrices* for explanation. 
        rotate_right_hand: whether to rotate the right hand or not. -1 means not to rotate. Other options 
                           are 90 and 180, which is to rotate 90 or 180 degrees
        rotate_left_hand: whether to rotate the left hand or not. -1 means not to rotate. Other option is 45,
                           which is to rotate 45 degrees
    OUTPUT:
        return the transformation matrix expressed in the our throw-catch zone system
    '''

    df = df.loc[df.iloc[:, 0] == object_id]

    # select columns that correspond to the local or global transformation matrix
    if t_matrix_type == 'local':
        df = df.iloc[:, 5:21]

    elif t_matrix_type == 'global':
        df = df.iloc[:, 21:37]
    t_matrix = []
    for index, row in df.iterrows():
        curr_matrix = [[row[0], row[1], row[2], row[3]],
                       [row[4], row[5], row[6], row[7]],
                       [row[8], row[9], row[10], row[11]],
                       [row[12], row[13], row[14], row[15]]]

        # convert the transformation matrix from optitrack system to the our throw-catch zone system
        curr_matrix = np.matmul(LOCAL_SYS_T_MATRIX_INV[local_sys_id], curr_matrix) # coordinate transformation 

        # rotate extra degrees if the right/left hand needs to
        # the object id needs to be specified as this function also processes other object's optitrack transformation matrix.
        # 116 is the right hand id in the optitrack system
        # 117 is the left hand id in the optitrack system
        if object_id == 116 and rotate_right_hand == 90:
            curr_matrix = np.matmul(curr_matrix, rotY_90)
        if object_id == 116 and rotate_right_hand == 180:
            curr_matrix = np.matmul(curr_matrix, rotY_180)

        if object_id == 117 and rotate_left_hand == 45:
            curr_matrix = np.matmul(curr_matrix, rotY_45)

        # if the object is helmet or headband, and rotate_helmet_headband is true, 
        # then rotating them accordingly
        if object_id == 115 and rotate_helmet_headband:
            curr_matrix = np.matmul(curr_matrix, rotY_180)
        if object_id == 118 and rotate_helmet_headband:
            curr_matrix = np.matmul(curr_matrix, rotY_45)


        t_matrix.append(curr_matrix)

    return np.array(t_matrix)



def get_ts_t_matrix(df_all, object_id, local_sys_id, t_matrix_type='global', rotate_right_hand=-1, rotate_left_hand=-1, rotate_helmet_headband=False):
    '''
    Get transformation matrix from optitrack's raw data, and its corresponding timestamp. 
    This function will be called in plot_motion.py
    INPUT:
        df_all: pandas dataframe of the optitrack's raw data
        object_id: object id
        local_sys_id: the id of the origin (ids should be either 0, 1, or 2)
        t_matrix_type: the type of the optitrack transformation matrix. Either 'global' or 'local'. We use 'global' in the project.
                       NOTE THAT this 'local' does not refer to the our throw-catch coordinate system.
        rotate_right_hand: whether to rotate the right hand or not. -1 means not to rotate. Other options 
                           are 90 and 180, which is to rotate 90 or 180 degrees
        rotate_left_hand: whether to rotate the left hand or not. -1 means not to rotate. Other option is 45,
                           which is to rotate 45 degrees
    OUTPUT:
        return the transformation matrix expressed in the our throw-catch zone system, and its corresponding timestamp
    '''
   
    df_target = df_all.loc[df_all.iloc[:, 0] == object_id]
    df_ts = df_target.iloc[:, 3]

    # select the columns that correspond to the transformation matrix
    if t_matrix_type == 'local':
        df_t_matrix = df_target.iloc[:, 5:21]
    elif t_matrix_type == 'global':
        df_t_matrix = df_target.iloc[:, 21:37]


    df_t_matrix_ts = pd.concat([df_t_matrix, df_ts], axis=1)

    ts_t_matrix_dict = {}
    for index, row in df_t_matrix_ts.iterrows():

        curr_matrix = [[row[0], row[1], row[2], row[3]],
                       [row[4], row[5], row[6], row[7]],
                       [row[8], row[9], row[10], row[11]],
                       [row[12], row[13], row[14], row[15]]]

        # convert the transformation matrix from the optitrack system to the our throw-catch zone system
        curr_matrix = np.matmul(LOCAL_SYS_T_MATRIX_INV[local_sys_id], curr_matrix)

        # rotate the right/left hand with extra degrees if needed
        # the object id needs to be specified as this function also processes other object's optitrack transformation matrix.
        # 116 is the right hand id in the optitrack system
        # 117 is the left hand id in the optitrack system
        if object_id == 116 and rotate_right_hand == 90:
            curr_matrix = np.matmul(curr_matrix, rotY_90)
        if object_id == 116 and rotate_right_hand == 180:
            curr_matrix = np.matmul(curr_matrix, rotY_180)

        if object_id == 117 and rotate_left_hand == 45:
            curr_matrix = np.matmul(curr_matrix, rotY_45)

        # if the object is helmet or headband, and rotate_helmet_headband is true, 
        # then rotating them accordingly
        if object_id == 115 and rotate_helmet_headband:
            curr_matrix = np.matmul(curr_matrix, rotY_180)
        if object_id == 118 and rotate_helmet_headband:
            curr_matrix = np.matmul(curr_matrix, rotY_45)

        # save the converted matrix in a dictionary with the key being its timestamp
        ts_t_matrix_dict[int(row[-1])] = curr_matrix

    return ts_t_matrix_dict




def t_matrix_to_tum_format(t_matrix, timestamp_list):
    '''
    Convert transformation matrix to tum format (x, y, z, qx, qy, qz, qw) (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
    For detains of the tum format, please refer to [URL to be added]
    INPUT:
        t_matrix: a transformation matrix
        timestamp_list: a list of timestamps associated with the input transformation matrix
    OUTPUT:
        return a data array in the tum format 
    '''
    tum_format = []
    for i in range(len(t_matrix)):
        curr_matrix = t_matrix[i]
        r_3x3 = curr_matrix[:3][:,:3]
        posi = curr_matrix[:3][:,-1].reshape(1, 3)
        # convert rotation matrix to quaternion
        orie = R.from_matrix(r_3x3).as_quat().reshape(1, 4)

        tum_format.append(np.insert(np.squeeze(np.concatenate((posi, orie), axis=1)), 0, timestamp_list[i]))

    return np.array(tum_format)


def save_tum_to_file(tum_format, filepath):
    '''
    Save data of tum format into a local file
    INPUT:
        tum_format: data array in the tum format
        filepath: path for saving data
    '''
    with open(filepath, 'w') as f:
        f.write('timestamp,x,y,z,qx,qy,qz,qw\n')
        for i in range(len(tum_format)):
            curr_tum = tum_format[i]
            # print(curr_tum)
            f.write('{},{},{},{},{},{},{},{}\n'.format(int(float(curr_tum[0])),curr_tum[1],curr_tum[2],curr_tum[3],
                                                       curr_tum[4],curr_tum[5],curr_tum[6],curr_tum[7]))


def convert(ipt_path, opt_path, local_sys_id, t_matrix_type, rotate_right_hand, rotate_left_hand, rotate_helmet_headband):
    '''
    process.py will call this function to save all tracked objects in a take into separate csv files.
    Each csv file contains the trajectory data of tum pose format (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
    INPUT:
        ipt_path: the file path of the raw optitrack data - optitrack.csv
        opt_path: path for saving the output files
        local_sys_id: the id of the origin (ids should be either 0, 1, or 2)
        t_matrix_type: the type of the transformation matrix (either 'global' or 'local'). We use 'global' in the project. 
                       Check our '/doc/processing_techdetails.md' -> *Difference between local and global transformation matrices* for explanation.
        rotate_right_hand: whether to rotate the right hand or not (either -1, 90 or 180). -1 means not to rotate. 
                           90 and 180 mean to rotate 90 and 180 degrees, respectively
        rotate_left_hand: whether to rotate the left hand or not (either -1 or 45). -1 means not to rotate. 
                          45 means to rotate 45 degrees    
    '''
    opti_df = read_csv(ipt_path)
    # get all objects' ids from optitrack's raw data
    obj_ids = get_all_obj_ids(opti_df)
    obj_paths = []
    
    for obj_id in obj_ids:
        # get transformation matrix in our coordinidate setting
        t_matrix = get_t_matrix(opti_df, obj_id, local_sys_id, t_matrix_type=t_matrix_type, rotate_right_hand=rotate_right_hand, 
                                rotate_left_hand=rotate_left_hand, rotate_helmet_headband=rotate_helmet_headband)
        timestamps = get_timestamp(opti_df, obj_id)
        tum_format = t_matrix_to_tum_format(t_matrix, timestamps)
        obj_path = os.path.join(opt_path, '{}_motion.csv'.format(OBJECTS[obj_id]))
        save_tum_to_file(tum_format, obj_path)
        obj_paths.append(obj_path)
        
    return obj_paths


def verify_integrity(take_path, total, tolerance):
    '''
    This function is called in recorder.py to verify the frame drop of optitrack is within a defined tolerance
    INPUT:
        take_path: the path of the current take
        total: the total number of frames the optirack needs to record
        tolerance: the defined tolerance for frame dropping
    '''

    # path to the optitrack raw data
    fpath = os.path.join(take_path, DEFAULT_FILENAME)
    if not os.path.isfile(fpath):
        # raise an exception if the raw data file not found
        raise Exception("Optitrack: recording is not saved.")

    # load raw data from the file
    with open(fpath, 'r') as f: lines = f.readlines()
    
    # frames counters for each optitrack object
    # key: object ID
    # value: frame count
    frames = {}
    for line in lines: # iterate through every line in the raw data file
        # split each line by ' '
        # the first text is the object ID
        obj = line.split(' ')[0]

        # increase counting of the object
        if obj in frames:
            frames[obj] += 1
        else:
            frames[obj] = 1
            
    for obj, num_frames in frames.items(): # iterate every object and its frame counting
        # calculate the frame drop rate
        drop_rate = float(total - num_frames) / total
        if abs(drop_rate) > tolerance:
            raise Exception("Optitrack: object {} drop {:.2%} frames exceeding the tolerance {:.2%}"
                            .format(obj, drop_rate, tolerance))


def if_processed_exist(path):
    '''
    Check if the processed data of every object exists under the passed directory

    Returns:
        bool: true if the processed data of all objects exists
              false if not exist for any object

    Args:
        path (string): path to the directory the processed data

    '''
    
    for k, v in OBJECTS.items(): # iterate every object
        # path to the specific processed data file
        path = os.path.join(path, '{}_motion.csv')
        if not os.path.exists(path):
            # return false if not exist
            return False

    # true if no non-existence detected
    return True
