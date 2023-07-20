'''
process.py

Lin Li (linli.tree@outlook.com) / 21 Sep. 2022 
Xi Luo (sunshine.just@outlook.com) / 14 Jul. 2023 

Formating and processing the raw data of ZED cameras, Event camera, OptiTrack and [Hand Engine](https://stretchsense.com/solution/hand-engine/).

Contributions:

The main processing logic and interface was written by Lin.
The interfaces of OptiTrack and alignment was written by Jianing Qiu
The codes refinement and documents were written by Xi


'''

import os, sys, csv
from argparse import ArgumentParser

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")

sys.path.append(root_path)

from src.log import take_id_to_str, parse_convert_take_ids
from src.utils import zed, he
from src.utils.align import align
from src.utils.plot_motion import plot_hand_pose_and_motion
import src.utils.event as evt
import src.utils.optitrack as opi
import shutil


'''
Arguments and Constants

'''

argparser = ArgumentParser()

argparser.add_argument('--takes', nargs='+', default=None,
                       help="ID of takes to be processed. Set None to process all takes in the 'data' directory. This can be given with a single integer number for one take or a range linked by '-', e.g., '10-12' for takes [000010, 000011, 000012]")
argparser.add_argument("--fps_event", type=int, default=60,
                       help="FPS for decoding event data into frames")
argparser.add_argument("--fps_zed", type=int, default=60,
                       help="FPS for decoding ZED RGBD frames, this should equal to the value used for recording")
argparser.add_argument('-d', "--duration", type=float, default=5,
                       help="the duration of recording in seconds")
argparser.add_argument('-t', '--tolerance', type=float, default=0.1,
                       help="the tolerance of frame dropping in the percentage for all devices")
argparser.add_argument('-da', '--depth_accuracy', choices=['float32', 'float64'], default='float32',
                       help="By default (None), the unnormalized depth map is not exported. Set this to any float precision to enable the export of depth data in the format of unnormalized depth maps.")
argparser.add_argument('--depth_img_format', choices=['png', 'jpg'], default='png',
                       help="image format of the exported RGB-D frames for ZED")
argparser.add_argument('--xypt', action='store_true', default=False,
                       help='true to export event stream in xypt format')
argparser.add_argument('--npy', action='store_true', default=False,
                       help='true to export depth stream in npy format')
argparser.add_argument('--datapath', type=str, default="",
                       help='raw data path of all takes')

# mapping from RGBD stream ID to ZED camera ID
ZED_CAMS = {
    'rgbd0' : '17471',
    'rgbd1' : '24483054',
    'rgbd2' : '28280967'
}

DIVISION_LINE_LENG = 60 # length of the division line printed in the console

'''
Data processing

'''

def process(take, args):
    '''
    verify the raw data and convert them to the proper format for the given take

    Returns: 
        None

    Args:
        take (string); the id of the take
        args (dict): arguments parsed from the console

    '''

    # generate the path to the take
    take_dir = os.path.join(args.datapath, take)
    # the path to the processed directory
    proc_dir = os.path.join(take_dir, 'processed')
    # the path to the raw directory
    raw_dir = os.path.join(take_dir, 'raw')

    # move all raw data into the raw directory if it not exists
    if not os.path.exists(raw_dir):
        # temporary directory to transfer data
        _raw_dir = os.path.join(take_dir, '../raw')

        # move data to the temporary directory
        shutil.move(take_dir, _raw_dir)
        # recreate the take directory
        os.makedirs(take_dir)
        # move the temporary data to the raw directory
        shutil.move(_raw_dir, take_dir)


    ts_paths = [] # list with timestamp filepath of all streams ZEDx3, Event, optitrack, HEx2

    print("processing RGBD streams")
    # iterate through three ZED streams
    for stream_id in ['rgbd0', 'rgbd1', 'rgbd2']:
        # get the camera ID by the stream ID
        device_id = ZED_CAMS[stream_id]
        print("processing: {} {}".format(stream_id, device_id))
        # generate the path to the raw file of ZED recording
        raw_path = os.path.join(raw_dir, '{}.svo'.format(device_id))
        
        # raise exception if the raw file not found
        if not os.path.isfile(raw_path):
            raise Exception("ZED {}: missing raw data at {}".format(stream_id, raw_path))

        # the path under the processed directory to the output data of processing
        opt_path = os.path.join(proc_dir, stream_id)
        if os.path.exists(opt_path):
            # skip the processing if the processed data (directory) already existed
            print("frames exists")
        else:
            # process raw data if not
            # create the output directory
            os.makedirs(opt_path)
            # call API from ZED module to export frame images
            zed.export_to_images(raw_path, opt_path, args.depth_img_format, args.depth_accuracy, save_pc=args.npy)

        # output path under the processed directory to the timestamp file
        proc_ts_path = os.path.join(proc_dir, '{}_ts.csv'.format(stream_id))
        if os.path.isfile(proc_ts_path):
            print("timestamps exist: {}".format(proc_ts_path))
        else:
            # path to the raw timestamp file
            raw_ts_path = os.path.join(raw_dir, '{}.csv'.format(device_id))
            # calibrate the ZED timestamps
            zed.calibrate_timestamps(raw_ts_path, proc_ts_path, args.fps_zed)
            
        # adding timestamp file path to the list for the alignment processing
        ts_paths.append(proc_ts_path)
        # evaluate the frame drop rate
        zed.verify_frame_integrity(opt_path, stream_id, int(args.fps_zed*args.duration), args.tolerance)

    
    print("processing event data")
    # raw data filename of event under the particular raw directory
    evt_fname = evt.get_raw_file(raw_dir)
    # raise an exception if the raw file not found
    if evt_fname is None: raise Exception("missing event data")
    
    # parsing init timestamp from the filename of raw file
    # a typical event raw file name is: 'event_{TS}.raw'
    # the timestamp {TS} is retrieved by:
    # 1. last split of '_': {TS}.raw
    # 2. segment from 0 to the last 4: {TS}
    # 3. convert string to integer
    ts_init = int(evt_fname.split('_')[-1][:-4])
    # the path to the raw file
    raw_path = os.path.join(raw_dir, evt_fname)
    
    # export xypt event streams if enabled in the args
    if args.xypt:
        xypt_path = evt.export_to_xypt(raw_path, proc_dir, ts_init)

    # export frame images
    frames_path = evt.export_to_images(raw_path, proc_dir, args.fps_event)
    # evaluate the frame drop rate
    evt.verify_frame_integrity(frames_path, args.fps_event*args.duration, args.tolerance)

    # generate the UNIX timestamp for each frame
    ts_step = 1.0e9 / args.fps_event # nanosecond per frame
    # list of frame numbers retrieved from the frame image filename
    # the frame number in filenames starts from 0
    # so +1 to get the time offset at the end of the interval when multiplied with ts_step 
    frames = sorted([int(fname[:-4])+1 for fname in os.listdir(frames_path)])
    # frame timestamp = init_timestamp + frame_num * time_step
    ts = ['{}\n'.format(int(ts_init + ts_step * frame)) for frame in frames]
    # add header, align with depth timestamp format
    ts.insert(0, 'nanoseconds\n')
    # path to the timestamp file under the processed directory
    ts_path = os.path.join(proc_dir, evt.FRAME_TS_FILENAME)
    # write timestamps to the file
    with open(ts_path, 'w') as f: f.writelines(ts)
    # add timestamp file path to the list for alignment
    ts_paths.append(ts_path)
    
    
    print("processing optitrack data")
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

    # the helmet and headband in takes from 0-1699 need to apply an additional rotation to correct their initial orientation
    # this requires rotating along Y axis with additional 45 degrees for the headband, and -180 degrees for the helmet.
    if 0 <= int(take) and int(take) <= 1699:
        rotate_helmet_headband = True
    else:
        rotate_helmet_headband = False
        
    # path to the optitrack raw file
    raw_path = os.path.join(raw_dir, opi.DEFAULT_FILENAME)
    
    # raise an exception if raw file not found
    if not os.path.isfile(raw_path): raise Exception("missing optitrack data")

    # if the processed optitrack data not exists
    if not opi.if_processed_exist(proc_dir):
        # convert the data from optitrack global coordinate to our coordinate
        # we use the optitrack's global transformation matrix
        opi_paths = opi.convert(raw_path,
                                proc_dir,
                                local_sys_id,
                                t_matrix_type='global',
                                rotate_right_hand=rotate_right_hand,
                                rotate_left_hand=rotate_left_hand,
                                rotate_helmet_headband=rotate_helmet_headband)
        # add timestamp file to the list for alignment
        ts_paths.append(opi_paths)

    
    print("processing gloves hands pose data")
    # path to hands raw data generated by hand engine (he for short)
    take_he_path = os.path.join(raw_dir, 'hand')
    # # if the raw data path not exists
    # if not os.path.exists(take_he_path):
    #     # alternative path to HE raw data must be specified in the console args
    #     assert args.he_dir is not None
    #     # get all raw data directories of the take under the alternative HE directory
    #     he_takes = [he_take for he_take in os.listdir(args.he_dir) if take in he_take]
    #     # the last one is correct one when multiple takes with the same take id
    #     he_path = os.path.join(args.he_dir, sorted(he_takes)[-1])
    #     print('he_path: ', he_path)
    #     # move the raw data directory to the path under the raw directory
    #     shutil.copytree(he_path, take_he_path)
    
    # evaluate frame drop rate for LEFT hand 
    he.verify_frame_integrity(take_he_path, he.Hand.LEFT, args.duration, args.tolerance) 
    # evaluate frame drop rate for RIGHT hand 
    he.verify_frame_integrity(take_he_path, he.Hand.RIGHT, args.duration, args.tolerance) 

    # list of processed HE data paths
    hand_paths = []
    # iterate through LEFT and RIGHT hands
    for hand in he.Hand.all():
        # call API from he module to format the raw data
        data = he.format(take_he_path, hand)
        # output path under the processed dirctory
        fpath = os.path.join(proc_dir, he.FORMAT_FILENAMES[hand])
        hand_paths.append(fpath)
        # write formatted data to a new file
        with open(fpath, 'w+') as f:
            for d in data:
                # each word in a line separated by ','
                for word in d[:-1]:
                    f.write('{},'.format(word))
                f.write(d[-1])
    # add processed HE data paths to the list for alignment
    ts_paths.append(hand_paths)

    # path to the alignment file
    align_path = os.path.join(proc_dir, 'alignment.json')
    # align if the alignment file not found
    if not os.path.exists(align_path):
        align(align_path, *ts_paths)
                
    # plot hand pose and motion
    if not os.path.exists(os.path.join(proc_dir, 'hand_motion')):
        print('plotting hand pose and motion, as well as object trajectory if captured')
        plot_hand_pose_and_motion(take_dir, local_sys_id, rotate_right_hand, rotate_left_hand)

        
'''
Main procedure

'''

if __name__ == '__main__':
    # parse arguments from the console
    args = argparser.parse_args()

    # parse and convert take ids, if have, from console to the 6-digit full format
    # e.g. 100 -> 000100
    if args.takes is not None:
        args.takes = parse_convert_take_ids(args.takes)

    # dict to record failed processings
    # key: failed take id
    # values: failed reasons
    failed = {}

    # iterate all take folders under the data_path in ascending order of the take ids.
    for take in sorted(os.listdir(args.datapath), key=int):
        
        if args.takes is not None and take not in args.takes:
            # only process the specified takes when given
            continue

        # print division string
        print('-' * DIVISION_LINE_LENG)
        print('Take:\t{}'.format(take))
        print('-' * DIVISION_LINE_LENG)
        try:
            # processs the take with the args
            process(take, args)
        except Exception as e:
            # convert exception to a string
            err_info = str(e)
            # put exception info in the 'failed' dict
            failed[take] = err_info
            print("Processing failed due to: {}".format(err_info))

    # sort the failed takes by ID in ascending order
    failed_takes = sorted(list(failed.keys()), key=int)
    if len(failed_takes) == 0:
        print("No issue detected in all processed takes")
    else:
        print("The takes below should be checked: ")
        print(failed_takes)
