import os, sys, json
from math import sqrt
import shutil
import re

# get the absolute path of the current and root directory
curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")

# add root directory to the system path
sys.path.append(root_path)

# from src.postprocess import ZED_CAMS
# from src.annotate import anno_dir, anno_dir_correct, load_timestamp, save_anno

# from src.log import data_path
from src.utils.align import align

# import src.utils.event as evt
# import src.utils.optitrack as opi
# from src.utils import zed, he
from addict import Dict

# data_path="/home/ur-5/Projects/justlx/Human_Catch_Throw_Transfer/data"
outdata_path = "/media/ur-5/catch-throw/data"


def align_ts(take, data_path):
    """
    verify the raw data and convert them to the proper format for the given take

    Returns:
        None

    Args:
        take (string); the id of the take
        args (dict): arguments parsed from the console

    """

    fps_zed = 60
    fps_event = 60
    duration = 5

    # lx
    # generate the path to the take
    take_dir = os.path.join(data_path, take)
    # tmp_take_dir = take_dir + '/data/' + take
    # the path to the processed directory
    out_take_dir = os.path.join(outdata_path, take)
    proc_dir = os.path.join(out_take_dir, "processed")
    # the path to the raw directory
    raw_dir = os.path.join(outdata_path, take, "raw")

    # move all raw data into the raw directory if it not exists
    if not os.path.exists(out_take_dir):
        os.makedirs(out_take_dir)
        # temporary directory to transfer data
        # _raw_dir = os.path.join(take_dir, '../raw')

        # lx
        shutil.copytree(take_dir, raw_dir)
        # # move data to the temporary directory
        # shutil.move(tmp_take_dir, _raw_dir)
        # # recreate the take directory
        # os.rmdir(take_dir + '/data')
        # # os.makedirs(take_dir)
        # # move the temporary data to the raw directory
        # shutil.move(_raw_dir, take_dir)

    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    ts_paths = (
        []
    )  # list with timestamp filepath of all streams ZEDx3, Event, optitrack, HEx2

    # iterate through three ZED streams
    for stream_id in ["rgbd0", "rgbd1", "rgbd2"]:
        # get the camera ID by the stream ID
        device_id = ZED_CAMS[stream_id]

        # output path under the processed directory to the timestamp file
        proc_ts_path = os.path.join(proc_dir, "{}_ts.csv".format(stream_id))
        raw_ts_path = os.path.join(raw_dir, "{}.csv".format(device_id))
        zed.calibrate_timestamps(raw_ts_path, proc_ts_path, fps_zed)

        # adding timestamp file path to the list for the alignment processing
        ts_paths.append(proc_ts_path)

    # raw data filename of event under the particular raw directory
    evt_fname = evt.get_raw_file(raw_dir)
    # raise an exception if the raw file not found
    if evt_fname is None:
        raise Exception("missing event data")

    # parsing init timestamp from the filename of raw file
    # a typical event raw file name is: 'event_{TS}.raw'
    # the timestamp {TS} is retrieved by:
    # 1. last split of '_': {TS}.raw
    # 2. segment from 0 to the last 4: {TS}
    # 3. convert string to integer
    ts_init = int(evt_fname.split("_")[-1][:-4])

    # generate the UNIX timestamp for each frame
    ts_step = 1.0e9 / fps_event  # nanosecond per frame
    frames = range(1, duration * fps_event + 1)
    # frame timestamp = init_timestamp + frame_num * time_step
    ts = ["{}\n".format(int(ts_init + ts_step * frame)) for frame in frames]
    # add header, align with depth timestamp format
    ts.insert(0, "nanoseconds\n")
    # path to the timestamp file under the processed directory
    ts_path = os.path.join(proc_dir, evt.FRAME_TS_FILENAME)
    # write timestamps to the file
    with open(ts_path, "w") as f:
        f.writelines(ts)
    # add timestamp file path to the list for alignment
    ts_paths.append(ts_path)

    # determine local coordinate system ID according the take ID
    if int(take) < 2889:
        local_sys_id = "0"
    elif int(take) < 9789:
        local_sys_id = "1"
    else:
        local_sys_id = "2"

    # the right hand in takes from 520-1699 needs to apply an extra rotation
    # 90 degrees along the Y axis for takes from 520-1559
    # and 180 degrees along the Y axis for takes from 1560-1699
    if 519 < int(take) and int(take) <= 1559:
        rotate_right_hand = 90
    elif 1559 < int(take) and int(take) <= 1699:
        rotate_right_hand = 180
    else:
        rotate_right_hand = -1

    # the left hand in takes from 1040-1559 needs to apply an extra rotation
    # 45 degrees along the Y axis
    if 1040 <= int(take) and int(take) <= 1559:
        rotate_left_hand = 45
    else:
        rotate_left_hand = -1

    # the helmet and headband in takes from 0-1699 need to apply an extra rotation to correct their initial orientation
    # this requires rotating along Y axis with extra 45 degrees for the headband, and -180 degrees for the helmet.
    if 0 <= int(take) and int(take) <= 1699:
        rotate_helmet_headband = True
    else:
        rotate_helmet_headband = False

    # path to the optitrack raw file
    raw_path = os.path.join(raw_dir, opi.DEFAULT_FILENAME)

    # raise an exception if raw file not found
    if not os.path.isfile(raw_path):
        raise Exception("missing optitrack data")

    # if the processed optitrack data not exists
    if not opi.if_processed_exist(proc_dir):
        # convert the data from optitrack global coordinate to our local coordinate
        # we use the optitrack's global transformation matrix
        opi_paths = opi.convert(
            raw_path,
            proc_dir,
            local_sys_id,
            t_matrix_type="global",
            rotate_right_hand=rotate_right_hand,
            rotate_left_hand=rotate_left_hand,
            rotate_helmet_headband=rotate_helmet_headband,
        )
        # add timestamp file to the list for alignment
        ts_paths.append(opi_paths)

    # path to HE raw data
    take_he_path = os.path.join(raw_dir, "hand")
    # if the raw data path not exists
    if not os.path.exists(take_he_path):
        # alternative path to HE raw data must be specified in the console args
        assert args.he_dir is not None
        # get all raw data directories of the take under the alternative HE directory
        he_takes = [he_take for he_take in os.listdir(args.he_dir) if take in he_take]
        # the last one is correct one when multiple takes with the same take id
        he_path = os.path.join(args.he_dir, sorted(he_takes)[-1])
        print("he_path: ", he_path)
        # move the raw data directory to the path under the raw directory
        shutil.copytree(he_path, take_he_path)

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
        with open(fpath, "w+") as f:
            for d in data:
                # each word in a line separated by ','
                for word in d[:-1]:
                    f.write("{},".format(word))
                f.write(d[-1])
    # add processed HE data paths to the list for alignment
    ts_paths.append(hand_paths)

    # path to the alignment file
    align_path = os.path.join(proc_dir, "alignment.json")
    # align if the alignment file not found
    if not os.path.exists(align_path):
        align(align_path, *ts_paths)
    return align_path


def correct_moment(moment, aligned, tss):
    rgbd0_f = str(moment.rgbd0.frame)
    for stream in moment.keys():
        ts = aligned[rgbd0_f][stream]
        moment[stream].timestamp = ts
        if stream in tss:
            moment[stream].frame = tss[stream].index(ts)


def finddatapath(take_id):
    data_root = "/home/ur-5/Projects/Human_Catch_Throw"
    data_zip_folder = ""
    take_id = int(take_id)
    if take_id >= 520 and take_id <= 759:
        data_zip_folder = "data_520-759"
    elif take_id >= 940 and take_id <= 1039:
        data_zip_folder = "data_940-1039"
    elif take_id >= 1040 and take_id <= 1559:
        data_zip_folder = "data_1040-1559"
    elif take_id >= 1560 and take_id <= 1699:
        data_zip_folder = "data_1560-1699"
    elif take_id >= 1700 and take_id <= 2089:
        data_zip_folder = "data_1700-2089"
    elif take_id >= 2090 and take_id <= 2209:
        data_zip_folder = "data_2090-2209"
    elif take_id >= 2210 and take_id <= 2359:
        data_zip_folder = "data_2210-2359"
    elif take_id >= 5299 and take_id <= 5528:
        data_zip_folder = "data_5299-5528"
    elif take_id >= 6029 and take_id <= 6398:
        data_zip_folder = "data_6029-6398"
    elif take_id >= 7189 and take_id <= 7498:
        data_zip_folder = "data_7189-7498"
    elif take_id >= 8979 and take_id <= 9128:
        data_zip_folder = "data_8979-9128"
    elif take_id >= 10669 and take_id <= 11018:
        data_zip_folder = "data_10669-11018"
    elif take_id >= 11259 and take_id <= 11408:
        data_zip_folder = "data_11259-11408"
    elif take_id >= 11409 and take_id <= 11558:
        data_zip_folder = "data_11409-11558"
    elif take_id >= 11559 and take_id <= 11708:
        data_zip_folder = "data_11559-11708"
    elif take_id >= 11709 and take_id <= 11858:
        data_zip_folder = "data_11709-11858"
    elif take_id >= 11859 and take_id <= 12008:
        data_zip_folder = "data_11859-12008"
    elif take_id >= 12009 and take_id <= 12158:
        data_zip_folder = "data_12009-12158"
    elif take_id >= 12159 and take_id <= 12308:
        data_zip_folder = "data_12159-12308"
    elif take_id >= 12459 and take_id <= 12608:
        data_zip_folder = "data_12459-12608"
    elif take_id >= 12609 and take_id <= 12755:
        data_zip_folder = "data_12609-12755"
    elif take_id >= 12756 and take_id <= 12905:
        data_zip_folder = "data_12756-12905"
    else:
        return None

    data_path = data_root + "/" + data_zip_folder
    return data_path


if __name__ == "__main__":
    sub1, sub2 = "sub1_head_motion", "sub2_head_motion"

    # a function that converts position tuple to a dict
    pos = lambda x: Dict({"x": round(float(x[0]), 2), "z": round(float(x[2]), 2)})

    take_id_list = []
    for anno_take_file in os.listdir(anno_dir):
        tmp_id = re.findall(r"\d+", anno_take_file)
        take_id_list.append(tmp_id[0])
    take_id_list.sort()

    missing_takeid = []
    error_takeid = []
    done_takeids = os.listdir(outdata_path)
    for take_id in take_id_list:
        if take_id in done_takeids:
            continue
        # data_path = finddatapath(take_id)
        data_path = "/media/ur-5/My Passport/Dataset/data"
        if data_path is None:
            missing_takeid.append(take_id)
            continue
        take_path = os.path.join(data_path, take_id)
        if not os.path.isdir(take_path):
            missing_takeid.append(take_id)
            continue
        align_path = align_ts(take_id, data_path)
        proc_path = os.path.join(outdata_path, take_id, "processed")

        with open(align_path, "r") as f:
            aligned = json.loads(f.read())

        anno_path = os.path.join(anno_dir, take_id + ".json")

        # optitack data
        # key: object ID
        # value: dict:
        #     key: timestamp
        #     value: optitrack data list
        opti = {}
        # only retrieve the optitrack data for the subjects' head motion
        # to get the location of two subjects and compute the flying speed of the thrown-away object
        for oid in [sub1, sub2]:
            opti_path = os.path.join(
                proc_path, oid + ".csv"
            )  # path to optitrack data file
            with open(opti_path, "r") as f:
                opti[oid] = {}  # new sub dict for a object ID
                for line in f.readlines()[1:]:  # iterate every row from 2nd
                    line = line.split(",")  # a list of separated data
                    # timestamp as key, the rest data as value
                    opti[oid][int(line[0])] = line[1:]

        tss = {}
        tss["rgbd1"] = load_timestamp(os.path.join(proc_path, "rgbd1_ts.csv"))
        tss["rgbd2"] = load_timestamp(os.path.join(proc_path, "rgbd2_ts.csv"))
        tss["event"] = load_timestamp(os.path.join(proc_path, "event_frames_ts.csv"))

        with open(anno_path, "r") as f:
            jsons = json.loads(f.read())
            anno = Dict(jsons)

        if anno.status == -1:
            error_takeid.append(take_id)
            continue

        try:
            throw_t = anno.throw.time_point
            catch_touch_t = anno.catch.time_point_touch
            correct_moment(throw_t, aligned, tss)
            correct_moment(catch_touch_t, aligned, tss)
            correct_moment(anno.catch.time_point_stable, aligned, tss)

            if anno.sub1_cmd.action == "throw":
                thrower = sub1
                catcher = sub2
            else:
                thrower = sub2
                catcher = sub1

            ts_thrower = throw_t[thrower].timestamp
            pos_throw = pos(opti[thrower][ts_thrower])
            anno.throw.position_thrower = pos_throw

            ts_catcher_throw = throw_t[catcher].timestamp
            anno.throw.position_catcher = pos(opti[catcher][ts_catcher_throw])

            ts_catcher_catch = catch_touch_t[catcher].timestamp
            pos_catch = pos(opti[catcher][ts_catcher_catch])
            anno.catch.position = pos_catch

            # flying duration in seconds
            flying_t = (ts_catcher_catch - ts_catcher_throw) / 1e9
            # flying distance = sqrt( (x-x')^2 + (z-z')^2)
            flying_s = sqrt(
                (pos_throw.x - pos_catch.x) ** 2 + (pos_throw.z - pos_catch.z) ** 2
            )
            # flying speed = flying distance / (flying duration+1e-9)
            # 1e-9 is a small constant to ensure numerical stability of division
            anno.throw.object_flying_speed = round(flying_s / (flying_t + 1e-9), 2)
        except:
            error_takeid.append(take_id)
            continue

        if not os.path.exists(anno_dir_correct):
            os.makedirs(anno_dir_correct)
        anno_path = os.path.join(anno_dir_correct, take_id + ".json")
        save_anno(anno, anno_path)
        print(take_id)

    print(
        "*****************************************************************************************************************************************************8"
    )
    print("---------------- saving missing take_id ---------------------------")
    # print(missing_takeid)
    miss_i = "missing_take_id.txt"
    with open(miss_i, "w") as f:
        for i in missing_takeid:
            f.writelines(i + "\n")

    print("-------------saveing error take_id-----------------------------")
    # print(missing_takeid)
    error_i = "error_take_id.txt"
    with open(error_i, "w") as f:
        for i in error_takeid:
            f.writelines(i + "\n")
