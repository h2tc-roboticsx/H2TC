import os, shutil
import time
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from subprocess import run, Popen, PIPE, STDOUT


'''
Constant

'''

RAW_EXTENSION = 'raw' # file extension of the raw data file
DEFAULT_FILE_NAME = "event." + RAW_EXTENSION # default file name of the raw data file

VIDEO_FILENAME = 'event.avi'
FRAME_DIR = 'event' # directory name of export frames

XYPT_FILENAME = 'event_xypt.csv'
FRAME_TS_FILENAME = 'event_frames_ts.csv'


'''
Recording

'''

def wait_record(rec_state, take_q):
    '''
    Block the process and wait for the signal from another process to start/stop recording
    
    For more information about the API used below, please check the official SDK:
    https://docs.prophesee.ai/stable/api.html

    Returns:
        None

    Args:
        rec_state (multiprocessing.Value): an integer value shared between the processes to control the start/stop of the recording
        take_q (multiprocessing.Queue): the queue shared between the processes to pass the path to the take directory i.e. output path

    '''
    
    print('before init event cam: ', time.time_ns())
    # initialize the default event camera
    cam = initiate_device(path="")
    # get the device handler
    cam_ctrl = cam.get_i_device_control()
    # set the camera to run alone
    cam_ctrl.set_mode_standalone()
    # get the event stream handler
    evt_strm = cam.get_i_events_stream()
    # get the recording biases
    cam_bias = cam.get_i_ll_biases().get_all_biases()
    # enable the event stream recording
    evt_strm.start()
    # start the camera
    cam_ctrl.start()
    # reset the camera to clear up
    cam_ctrl.reset()
    print('after init event cam: ', time.time_ns())

    # iterate until the close signal from the main process
    # i.e. the recording state is changed to -1
    while rec_state.value != -1:
        # check if events availabel in the buffer
        new_evt = evt_strm.poll_buffer()
        # skip to next iteration if no events available i.e. returning code <= 0
        if new_evt <= 0: continue
        # consume the existing events to clear up the event stream buffer
        evt_buffer = evt_strm.get_latest_raw_data()
        # get the path to the take directory if have in the queue
        if not take_q.empty():
            take_path = take_q.get()
            # create the output path
            opt_path = os.path.join(take_path, DEFAULT_FILE_NAME)

        # recording starts now
        if rec_state.value == 1:
            print('prepare to record event: {}'.format(time.time_ns()))
            # consume the existing events to clear up the event stream buffer
            evt_buffer = evt_strm.get_latest_raw_data()
            # specify the output path for the event stream
            evt_strm.log_raw_data(opt_path)
            print('event starts recording at {}'.format(time.time_ns()))

            # get the current timestamp as the recording start (initial) timestamp
            # events are stored in raw file with a microsecond offset from the recording start time
            # we maintain our own start timestamp because we can't find it in the raw file.
            ts = time.time_ns()
            # iterate until recording stops i.e. recording state is chenged by the main process to be not 1
            while rec_state.value == 1:
                # check if new events available in the buffer
                new_evt = evt_strm.poll_buffer()
                # skip to next iteration if no new event available
                if new_evt == 0: continue

                # error encountered or no more events available
                elif new_evt < 0:
                    # stop output to the raw file
                    evt_strm.stop_log_raw_data()
                    # close the camera
                    close(cam, cam_ctrl, evt_strm)
                    # set the recording state to 0 to exit the recording
                    rec_state.value = 0
                    print("Event: recording stopped due to runtime error.")
                    exit(new_evt)

                # flush the events from buffer to the output raw file
                evt_strm.get_latest_raw_data()

            evt_strm.stop_log_raw_data()
            print('event ends recording at {}'.format(time.time_ns()))

            # rename file to include the timestamp of start recording
            shutil.move(opt_path, os.path.join(take_path, "event_{}.{}".format(ts, RAW_EXTENSION)))

            # export real-time bias config
            bias_path = os.path.join(take_path, "event.bias")
            with open(bias_path, 'w') as f:
                for k, v in cam_bias.items():
                    f.write('{} % {}\n'.format(v, k))
    close(cam, cam_ctrl, evt_strm)
    # release shared q if have to safely terminate the process
    while not take_q.empty(): take_q.get()

    
def close(cam, cam_ctrl, evt_strm):
    '''
    Close the camera and release the resource
    
    Returns:
        None

    Args:
        cam (device): metavision camera device
        cam_ctrl (Camera Control): metavision camera control
        evt_strm (Event stream): metavision events stream

    '''

    # stop recording
    evt_strm.stop_log_raw_data()
    # close event stream
    evt_strm.stop()
    # close camera
    cam_ctrl.stop()
    # delete camera instance to release resources
    del cam

    
'''
Decoding

'''
    
def export_to_xypt(ipt_path, opt_path, ts_init):
    '''
    Decode xypt event stream from the raw data file

    Returns:
        string: path to the output file of xypt event stream

    Args:
        ipt_path (string): path to the raw file
        opt_path (string): path to the output directory
        ts_init (int): the UNIX timestamp in nanoseconds of recording start

    '''
    
    # a detailed explanation is in https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md/#event
    opt_path = os.path.join(opt_path, XYPT_FILENAME) # path to the output xypt file

    if os.path.isfile(opt_path):
        # return output path directly if it already exists
        print("Event xypt data has been generated before.")
        return opt_path

    # create event stream iterator from the raw file
    # iterate events in every 1000 microseconds
    # for more detail about EventsIterator, check
    # https://docs.prophesee.ai/stable/metavision_sdk/modules/core/tutorials/events_iterator.html
    mv_iterator = EventsIterator(input_path=ipt_path, delta_t=1000)
    
    with open(opt_path, 'w') as f:
        # write the headers to the output file
        f.write('x,y,polarity,timestamp\n')
        # Read formatted CD events from EventsIterator & write to CSV
        for evs in mv_iterator:
            # iterate every event in the accumulated time i.e. 1000 microseconds in this case
            for (x, y, p, t) in evs:
                # one event per line, data separated by ','
                # timestamp = initial timestamp + offset 't' (in microseconds) * 1000 (convert to nanoseconds)
                # Note that there is no any connection between the values of 'delta_t' and 't'.
                # these two '1000' are just a beautiful coincidence.
                f.write("%d,%d,%d,%d\n" % (x, y, p, ts_init + t*1000))
                
    return opt_path


def export_to_images(ipt_path, opt_path, fps=30):
    '''
    Decode frame images from the raw data file

    For more detail of this visualization, check the web:
    https://docs.prophesee.ai/stable/concepts.html

    Returns:
        string: path to the output directory of the frame images

    Args:
        ipt_path (string): path to the raw data file
        opt_path (string): parent directory of the output frames directory
        fps (int): frames per second

    '''

    # a detailed explanation is in https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md/#event
    frames_path = os.path.join(opt_path, FRAME_DIR) # directory to export the frames

    if os.path.exists(frames_path):
        # return the directory if it already exists
        print('Event frames has been extracted before')
        return frames_path

    video_path = os.path.join(opt_path, VIDEO_FILENAME) # path to the intermediate video file

    # generate RGB video from raw file if haven't before
    if os.path.isfile(video_path):
        print('Event video has been generated before')
    else: 
        # Metavision API fixes the export video to be 30 FPS
        # to allow the 60 FPS video, we adjust the slow factor to be 2 (30 FPS * 2 = 60 FPS)
        # by doing so, the length of the video increases so that the total amount of frames
        # is equivalent to that of the video in 60 FPS.
        slow_factor = int(fps / 30.0)
        # events over every 1/FPS (*1e6 to convert to microseconds) are accumulated to visualize
        # this ensures that each event appears in only one frame.
        accumulation = int(1.0 / fps * 1000000)
        # run 'metavision_raw_to_video' application from Metavision SDK to export the video
        # type 'metavision_raw_to_video -h' in console for more detail
        run(['metavision_raw_to_video',
             '-i', ipt_path,
             '-o', video_path,
             '-s', str(slow_factor),
             '-a', str(accumulation)])

    # create the frames output directory
    os.makedirs(frames_path)

    # run 'ffmpeg' program to decode frame images from the video
    rtn = run(['ffmpeg',
               '-i', video_path,
               '-start_number', '0', # frame numbering starts from 0
               '-vf', 'fps=30', # FPS=30, fixed by the metavision_raw_to_video
               os.path.join(frames_path, '%04d.jpg')],
              stdout=PIPE, # redirect the system output to PIPE
              stderr=PIPE) # redirect the error info to PIP
    # remove the intermediate video file after export
    run(['rm', video_path])
    return frames_path


'''
Verification

'''

def get_raw_file(take_path, ext=RAW_EXTENSION):
    '''
    Find the raw data file name under the directory

    Returns:
        string: filename of the raw data file

    Args:
        take_path (string): path to the take directory
        ext (string): extension of the raw data file

    '''

    # iterate through every files under the directory
    # return the filename if it matches the extension
    for fname in os.listdir(take_path):
        if ext in fname:
            return fname
    return None


def verify_file_integrity(take_path):
    '''
    Raise an exception if any raw file missing

    Returns:
        None

    Args:
        take_path (string): path to the take directory

    '''
    
    raw_fname = get_raw_file(take_path)
    if raw_fname is None:
        # raise an exception if no raw file is found
        raise Exception("Event: recording is not saved.")
    elif raw_fname == DEFAULT_FILENAME:
        # raise an exception if the found raw file has the default file name
        # i.e. no initial timestamp included
        raise Exception("Event: timestamp is missing.")
    
def verify_frame_integrity(frame_path, total, tolerance=0.1):
    '''
    Raise an exception if frames drop more than the specified tolerance

    Returns:
        None

    Args:
        frame_path (string): directory of the frame images
        total (int): expected total number of frames
        tolerance (float): the tolerated percentage of frames dropped

    '''

    # all filenames under the frame directory
    frames = os.listdir(frame_path)
    num_frames = len(frames)
    drop_rate = float(total - num_frames) / total
    if abs(drop_rate) > tolerance:
        raise Exception("Event: {:.2%} frames drop exceeding the tolerance {:.2%}"
                        .format(drop_rate, tolerance))    
