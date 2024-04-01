import multiprocessing
from multiprocessing import Process, Pool
from utils import zed
from log import *

import os, csv
import pyzed.sl as sl
import pandas as pd
from warnings import warn




def zed_run(RECORDING, cam_id, take_path, length, fps, resolution):
	init = sl.InitParameters()
	init.camera_resolution = RESOLUTION[resolution]
	init.camera_fps = fps

	init.set_from_serial_number(int(cam_id))
	cam = sl.Camera()
	# open the camera
	status = cam.open(init)
	if status != sl.ERROR_CODE.SUCCESS:
		raise Exception("Camera {} fails to open: {}".format(cam_id, repr(status)))
		cam.close()

	print("ZED {} opened successfully.".format(cam_id))



	rec_path = os.path.join(take_path, '{}.svo'.format(cam_id))
	print('>>>>>>>>>>>>>>>>>>>>>rec_path: ', rec_path)
	rec_param = sl.RecordingParameters(rec_path, sl.SVO_COMPRESSION_MODE.H264)
	print('>>>>>>>>>>>>>>>>rec_param: ', rec_param)
	status = cam.enable_recording(rec_param)
	print('>>>>>>>>>>>>>status: ', status)
	print('a' * 20)
	if status != sl.ERROR_CODE.SUCCESS:
		print('Exception rasied **************')
		raise Exception("Camera {} fails to record:{}".format(cam_id, repr(status)))



	run_param = sl.RuntimeParameters()
	fps = cam.get_camera_information().camera_fps
	nframes = fps * length
	frame = 1

	# store timestamp for each frame in a separate csv file
	ts_path = os.path.join(take_path, '{}.csv'.format(cam_id))
	# f = open(ts_path, 'w+')
	# ts_writer = csv.writer(f)
	ts_list = []


	print("ZED {} waits for recording".format(cam_id))
	# block until the recording state is changed to true by the main thread
	while not RECORDING[cam_id]: pass

	# write the initial timestamp
	# print('zed-ns: ', cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
	# ts_writer.writerow(cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
	ts_list.append(cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
	while frame <= nframes:
		status = cam.grab(run_param)
		if status != sl.ERROR_CODE.SUCCESS:
			warn("fail to grab the {}-th frame".format(frame))
		frame += 1
		ts_list.append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

	# change the recording state to false to inform the main thread
	RECORDING[cam_id] = False
	cam.disable_recording()
	# f.close()
	ts_df = pd.DataFrame({'nanoseconds': ts_list})
	ts_df.to_csv(ts_path, index=False)



'''
def test_func():
	print('pig')





# prepare the camera to be ready for recording
# block until the signal from the main thread for start recording
# the function to be called in the separate thread
def wait_record(cam_id, take_path, length):
	print('8' * 20)
	global RECORDING
	print('8-1' * 20)
	rec_path = os.path.join(take_path, '{}.svo'.format(cam_id))
	print('8-2' * 20)
	rec_param = sl.RecordingParameters(rec_path, sl.SVO_COMPRESSION_MODE.H264)
	print('8-3' * 20)
	print('zed_cams_mgr[cam_id]: ', zed_cams_mgr[cam_id])
	status = zed_cams[cam_id].enable_recording(rec_param)
	print('8-4' * 20)
	if status != sl.ERROR_CODE.SUCCESS:
		raise Exception("Camera {} fails to record:{}".format(cam_id, repr(status)))

	print('9' * 20)

	run_param = sl.RuntimeParameters()
	fps = zed_cams[cam_id].get_camera_information().camera_fps
	nframes = fps * length
	frame = 1

	# store timestamp for each frame in a separate csv file
	ts_path = os.path.join(take_path, '{}.csv'.format(cam_id))
	# f = open(ts_path, 'w+')
	# ts_writer = csv.writer(f)
	ts_list = []


	print("ZED {} waits for recording".format(cam_id))
	# block until the recording state is changed to true by the main thread
	while not RECORDING[cam_id]: pass

	# write the initial timestamp
	# print('zed-ns: ', cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
	# ts_writer.writerow(cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
	ts_list.append(zed_cams[cam_id].get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
	while frame <= nframes:
		status = zed_cams[cam_id].grab(run_param)
		if status != sl.ERROR_CODE.SUCCESS:
			warn("fail to grab the {}-th frame".format(frame))
		frame += 1
		ts_list.append(zed_cams[cam_id].get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

	# change the recording state to false to inform the main thread
	RECORDING[cam_id] = False
	zed_cams[cam_id].disable_recording()
	# f.close()
	ts_df = pd.DataFrame({'nanoseconds': ts_list})
	ts_df.to_csv(ts_path, index=False)



# initialize multiple ZED depth cameras
zed_cams = zed.init_cams(3, 30, '720p')

zed_cams_mgr =multiprocessing.Manager().dict()
print('zed_cams_mgr: ', zed_cams_mgr)
'''

take_path = '/home/ur5/Catch-Throw-Temp/src/test_zed'
length = 5
fps = 30
resolution = '720p'

RECORDING = multiprocessing.Manager().dict({
	'24483054': False, 
	'17471': False, 
	'28280967': False
	})


RESOLUTION = {
	'720p' : sl.RESOLUTION.HD720
}

process_dict = {}
# zed_p = Pool(3)
for cam_id in ['24483054', '17471', '28280967']:
	# zed_cams_mgr[cam_id] = 1
	# print('%' * 20)
	# print('cam: ', cam)
	p = Process(target=zed_run, args=(RECORDING, cam_id, take_path, length, fps, resolution))
	# zed_p.apply_async(zed_run, args=(RECORDING, cam_id, take_path, length, fps, resolution))
	# zed_p.apply_async(test_func)

	# print('>' * 20)

	p.start()
	process_dict[cam_id] = p



for p_key in process_dict:
	RECORDING[p_key] = True
	print('camera {} starts recording'.format(p_key))
	process_dict[p_key].join()
	print('camera {} ends recording'.format(p_key))

# trigger the recording in each separate thread
# by setting the corresponding RECORDING entry to be true
# zed.start_recording(zed_cams)

# RECORDING = multiprocessing.Manager().dict({
# 	'24483054': True, 
# 	'17471': True, 
# 	'28280967': True
# 	})


print('1' * 20)
# block until any zed recording finished
# while not zed.any_finished(zed_cams): pass

print('2' * 20)

# # block until all zed recording finished
# # while not zed.all_finished(zed_cams): pass
# for p_key in process_dict:
#     process_dict[p_key].join()

print('3' * 20)


# zed_p.close()
# zed_p.join()


print('all ZED stop')

while True: pass

# zed.end(zed_cams)
