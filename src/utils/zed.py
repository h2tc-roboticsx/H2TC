########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import os, sys
import pyzed.sl as sl
import pandas as pd
import numpy as np
from warnings import warn
from datetime import datetime
import cv2
import csv
from pathlib import Path


'''
Constants

'''

# device numbers of ZED cameras
CAM_IDS = ['24483054', '17471', '28280967']

# file extension of the raw data
RAW_EXTENSION = 'svo'
# name raw data files with camera IDs
RAW_FILE_NAMES = {cam_id : '{}.{}'.format(cam_id, RAW_EXTENSION)
                  for cam_id in CAM_IDS}
# name timestamp files with camera IDs
TS_FILE_NAMES = {cam_id : '{}.csv'.format(cam_id)
                 for cam_id in CAM_IDS}

# parse resolution string to the format specified by ZED sdk
# add more entries here to support more resolution
RESOLUTION = {
    '720p' : sl.RESOLUTION.HD720
}

'''
Recording

'''

def wait_record(rec_state, cam_id, take_q, fps, resolution):
    '''
    Block the process and wait for the signal from another process to start/stop recording
    
    For the detail of the parameters involved in recording, please check the ZED SDK and the web:
    https://www.stereolabs.com/docs/video/

    Returns:
        None

    Args:
        rec_state (multiprocessing.Value): integer shared between the processes to control the start/stop of the recording
        cam_id (string): the id of the camera to record
        take_q (multiprocessing.Queue): the queue shared between the processes to pass the path to the take directory i.e. output path
        fps (int): frame rate of recording
        resolution (string): resolution abbreviation like 720p. the value should be a key in RESOLUTION
    
    '''
    
    # initialize the camera parameters
    init = sl.InitParameters()
    # set up resolution and FPS
    init.camera_resolution = RESOLUTION[resolution]
    init.camera_fps = fps

    # specify the camera to use by camera ID
    init.set_from_serial_number(int(cam_id))
    # initialize the camera object
    cam = sl.Camera()
    # open the camera
    status = cam.open(init)

    # raise an Exception and close the camera if it fails to open
    if status != sl.ERROR_CODE.SUCCESS:
        raise Exception("Camera {} fails to open: {}".format(cam_id, repr(status)))
        cam.close()
        
    print("ZED {} opened successfully.".format(cam_id))
    
    # iterate until receiving the signal of close the camera
    #  i.e. the recording state is changed to -1 by the main process
    while rec_state.value != -1:

        # update runtime parameters only if the queue having takes to process
        if not take_q.empty():
            # get the take path
            take_path = take_q.get()
            # generate the raw file path
            rec_path = os.path.join(take_path, RAW_FILE_NAMES[cam_id])
            # initialize recording parameters
            rec_param = sl.RecordingParameters(rec_path, sl.SVO_COMPRESSION_MODE.H264)
            # enable the recording mode
            status = cam.enable_recording(rec_param)
            # raise an exception if the camera fails to be ready for recording
            if status != sl.ERROR_CODE.SUCCESS:
                raise Exception("Camera {} fails to record:{}".format(cam_id, repr(status)))

            # initialize runtime parameters
            run_param = sl.RuntimeParameters()
            # the frame numbering
            frame = 1
            
            # store timestamp for each frame in a separate csv file
            # create the timestamp filepath
            ts_path = os.path.join(take_path, TS_FILE_NAMES[cam_id])
            # timestamps list
            ts_list = []

            print("ZED {} waits for recording".format(cam_id))

        # continue to the next iteration to block the process if not receiving the start signal
        # i.e. if the recording state doesn't equal to 1
        if rec_state.value != 1: continue
        
        # recording starts now
        
        # calling ZED api to get the current timestamp as recording start timestamp
        # convert it to UNIX timestamp and stored in timestamps list
        # link: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#af18a2528093f7d4e5515b96e6be989d0
        ts_list.append(cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())

        # iteratively grab the frames until receiving the stop signal
        # i.e. if the recording state is changed by the main process to be not 1
        while rec_state.value == 1:
            # calling camera to take a frame
            status = cam.grab(run_param)
            # print warning, but not interrupt recording, if the camera fails to take a frame
            if status != sl.ERROR_CODE.SUCCESS:
                warn("fail to grab the {}-th frame".format(frame))
            # update frame numbering
            frame += 1
            # calling ZED api to get the timestamp of taking the last (current) frame
            ts_list.append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

        # exit recording mode
        cam.disable_recording()
        # export timestamps list to a Pandas-style CSV file
        ts_df = pd.DataFrame({'nanoseconds': ts_list})
        ts_df.to_csv(ts_path, index=False)

    # close the camera
    cam.close()
    # consume objects in the shared queue, if have, to safely terminate the process
    while not take_q.empty(): take_q.get()

    
'''
Export

'''
    
def progress_bar(percent_done, bar_length=50):
    '''
    display a bar animation of the progress in console

    Args:
        percent_done (float): percentage has been done
        bar_length (int): the length of the printed bar

    '''
    
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


#!/usr/bin/python3
def depth_image_to_point_cloud(rgb, depth, scale, K, pose,mask=None):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) * scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    mask = np.ravel(mask)
    
    valid = (Z > 0) & (Z < 4.4) & (X<0) # sub1 area 
    if mask is not None:
        valid = valid & (mask==0) # human area

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    # flag = (position[0]>0)&(position[1]>0)&(position[2]>0)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B)))

    return points

#!/usr/bin/python3
def write_point_cloud(ply_filename, points, samples = 4096):
    if samples is not None:
        random_flag = np.random.RandomState(seed=42).permutation(points.shape[0])[: (samples - 1)]
        points = points[random_flag].tolist()
    else:
        points.tolist()
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

def export_to_images(ipt_path, opt_path, img_format='jpg', depth_acc='float64', save_img=True, save_depth=True, save_pc=True):
    '''
    decode RGB, depth images and depth map from raw data file

    Returns:
        None

    Args:
        ipt_path (string): path to the raw data file
        opt_path (string): path to the output directory
        img_format (string): format of the exported images like png, jpg
        depth_acc (string): float precision of the exported depth map

    '''

    # get init parameters
    init_params = sl.InitParameters()
    # set the input path to initialize the camera virtually
    init_params.set_from_svo_file(ipt_path)
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # create ZED objects
    zed = sl.Camera()

    # open the raw file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # report error and close the camera if the raw file failed to open
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # get image size
    image_size = zed.get_camera_information().camera_resolution    
    width = image_size.width
    height = image_size.height
    
    # lx: cam info
    cam_info = zed.get_camera_information().calibration_parameters.left_cam
    K = np.array([[cam_info.fx, 0.0, cam_info.cx],
                                [0.0, cam_info.fy, cam_info.cy],
                                [0.0, 0.0, 1.0]])
    cam2world = np.eye(4)

    # prepare single image containers
    left_image = sl.Mat()
    depth_image = sl.Mat()
    depth_map = sl.Mat()

    # initialize runtime parameters
    rt_param = sl.RuntimeParameters()
    # set the depth mode to FILL
    # check https://www.stereolabs.com/docs/depth-sensing/depth-settings/
    # for different depth mode and other settings
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    nb_frames = zed.get_svo_number_of_frames() # total number of frames
    nb_failed = 0 # count of failed attempts to grab a frame
    depth_frames = [] # accurate depth maps
    mask_folder = str(Path(opt_path).parent) + '/masks'
    while True:
        # grab a frame
        if zed.grab(rt_param) != sl.ERROR_CODE.SUCCESS:
            # increase the counting of failed grab if returned code is not SUCCESS
            nb_failed += 1
            print("failed to grab x{}".format(nb_failed))
            # break the export if failed to grab a frame more than 10 times
            if nb_failed == 10: break
            continue
        
        svo_position = zed.get_svo_position() # the current frame number
        # retrieve LEFT images
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        # retrieve DEPTH images
        zed.retrieve_image(depth_image, sl.VIEW.DEPTH)

        if save_pc:
            # retrieve real depth values if depth accuracy is specified
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # store depth values in numpy ndarray format
            data = depth_map.get_data()
            depth_frames.append(np.copy(data)) # depth_frames[-1].shape -> (720, 1280)
          
       
        if save_img:
            # generate file names with the frame numbering
            left_path = os.path.join(opt_path, "left_{:04d}.{}".format(svo_position, img_format))
            # save images
            cv2.imwrite(left_path, left_image.get_data())
        
        # depth images
        if save_depth:
            depth_path = os.path.join(opt_path, "depth_{:04d}.{}".format(svo_position, 'png'))
            cv2.imwrite(depth_path, depth_image.get_data())

        # display progress
        progress_bar((svo_position + 1) / nb_frames * 100, 30)
        
        # check if we have reached the end of the video
        if svo_position >= (nb_frames-1):  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break
        
    if len(depth_frames) > 0:
        # save depth map in the output path if have
        np.save(os.path.join(opt_path, 'depth.npy'), np.asarray(depth_frames, dtype=depth_acc))

    zed.close()

    
'''
Timestamp

'''

def calibrate_timestamps(in_path, out_path, fps):
    '''
    Calibrate the timestamps with the timestamp of recording started

    Returns:
        None

    Args:
        in_path (string): path to the raw timestamp file
        out_path (string): path to save the processed timestamp
        fps (int): the FPs used for recording

    '''
    
    # a detailed explanation is in https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md/#zed-rgbd
    # load timestamps
    with open(in_path, 'r') as f:
        tss = f.readlines()

    # init the list of calibrated timestamps
    # the timestamp of 1st frame = the timestamp of recording started + 1/FPS
    calibrated = [int(tss[1]) + int(1/fps * 1e9)]
    # iterate every timestamp in the raw file from index 3
    # corresponding to the original timestamp of the 2nd frame
    # the last timestamp is ignored, since the number of frames decoded from ZED svo
    # is 1 frame less than the number of timestamps.
    for i in range(3, len(tss)-1):
        # calibrated timestamp [i] = calibrated[i-1] + the offset calculated by the original timestamps
        ts = int(tss[i]) - int(tss[i-1]) + calibrated[i-3]
        calibrated.append(ts)

    # export to the save path
    ts_df = pd.DataFrame({'nanoseconds': calibrated})
    ts_df.to_csv(out_path, index=False)

    
'''
Verification

'''
    
def verify_file_integrity(take_path, cam_ids=CAM_IDS):
    '''
    raise an exception if the raw data file not found

    Returns:
        None

    Args:
        take_path (string): the path to the take directory
        cam_ids (list): camera IDs

    '''
    
    for cam_id in cam_ids:
        # raw file paths
        raw_fpath = os.path.join(take_path, RAW_FILE_NAMES[cam_id])
        # timestamp file paths
        ts_fpath = os.path.join(take_path, TS_FILE_NAMES[cam_id])
        if not (os.path.isfile(raw_fpath) and os.path.isfile(ts_fpath)):
            raise Exception("ZED {}: Recording is not saved.".format(cam_id))
            
            
def verify_frame_integrity(frame_path, cam_id, total, tolerance=0.1):
    '''
    raise an exception if the frames drop more than the threshold

    Returns:
        None

    Args:
        frame_path (string): path to the frames directory
        cam_id (string): the id of the camera being evaluated
        total (int): the expected total number of frames
        tolerance (float): the threshold percentage of frames dropped

    '''

    # get all frames (files)
    frames = os.listdir(frame_path)
    num_frames = len(frames) // 2 # divided by 2 since both RGB and Depth in the same directory
    drop_rate = float(total - num_frames) / total
    if abs(drop_rate) > tolerance:
        raise Exception("ZED {}: {:.2%} frames drop exceeding the tolerance {:.2%}"
                        .format(cam_id, drop_rate, tolerance))
