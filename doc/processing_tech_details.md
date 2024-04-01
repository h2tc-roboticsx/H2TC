# Data Processing Technical Details

This document introduces details on how we process [H<sup>2</sup>TC](https://h2tc-roboticsx.github.io/). For a detailed explanation of the data hierarchy and contents in the dataset, please see  [data file explanation](https://github.com/h2tc-roboticsx/H2TC/tree/main/doc/data_file_explanation.md). 

Here is an overview of this document:

* [**Workspace and Hardwares**](#the-workspace-and-hardwares): introduces the multi-modal [sensors](#used-devices) and the throw&catch [workspace](#the-throw-catch-coordinate-frame) for building the dataset. 
* [**Timing and Data Alignment**](#data-timing-and-alignment): introduces how [rgb, depth](#zed), [event](#event), [optitrack](#optitrack) and [hand joint motion](#hand-joint-motion) streams are timestamped, [synchronized](#clock-synchronization) and [aligned](#modality-alignment) in recording and processing. 
* [**OptiTrack Data Processing**](#optitrack-data-processing): explains how the motions of involved objects are captured by OptiTrack and how they are processed.
* [**Hand Data Processing**](#hand-data-processing): explains the [hand model](#hand-modelling) used in the dataset and how to [reconstruct](#motion-reconstruction) the hand motion, both joint and 6D global, from the data streams captured with OptiTrack and MoCap gloves.

## The Workspace and Hardwares

A `throw&catch`  activity in our dataset refers to a dyadic collaborative process where two human subjects observe, move and coordinate to throw/catch an object from one to the other.
Each activity was recorded in a flat lab area, which resembles the real throw&catch scenes characterized by unstructured, cluttered and dynamic surroundings. 
We refer interested users to our technical [paper](add) for a more detailed introduction of the throw&catch workspace.

### Hardware and Sensors
We employ multiple high-precision motion capture and visual streaming systems to capture the dataset. Briefly, we use three [ZED]((https://www.stereolabs.com/zed-2/)) stereo cameras to capture RGB and depth streams, a [Prophesee](https://www.prophesee.ai/) event camera to capture event streams, a pair of [StretchSense MoCap Pro](https://stretchsense.com/) (SMP) gloves to capture the hand joint motions, and the [OptiTrack](https://optitrack.com/) to capture the global object and human body motions.

<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/hardware.png" width = "1000" alt="hardware" />


| Device | Manufacturer | Recording Content |FPS |Resolution  |
|:-----|:-----:|:-----:|:-----:|:-----:|
| ① Gloves | [StretchSense MoCap Pro](https://stretchsense.com/) | Hand Joint Pose | 120 | - |
| ②⑤ Markers | [OptiTrack](https://optitrack.com/) |  Human Motion | 240 | - |
| ③ Event Camera | [Prophesee](https://www.prophesee.ai/) | Event | - | 1280x720 |
| ④ ZED Camera | [Stereolabs](https://www.stereolabs.com/zed-2/) |  RGB-D | 60 | 1280x720 |


### The Throw&Catch Coordinate Frame

We set the throw&catch frame at the throw&catch workspace as the global coordinate frame.  As shown below, the **origin** of the throw-catch frame lies at the bottom-left corner of the workspace. The coordinate  axes are set up as follows: XZ plane is parallel to the ground plane with Z-axis along the longer side and X-axis along the shorter side. Y-axis is perpendicular up to the XZ plane. 

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/workspace_lx.png" width = "460" alt="workspace" />
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/schema.png" width = "400" alt="workspace" />
</div>

We have transformed all data streams captured with OptiTrack (i.e. the global motion streams of the headband, helmet, gloves, and all 3d-printed objects) to the common throw&catch frame via the `process` function in [src/process.py](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/process.py).  Please check [OptiTrack data processing](#optitrack-data-processing) for more details.  
<br>

## Data Timing and Alignment

Our recording system consists of  3 [ZED](#zed) stereo cameras, 1 Prophesee event camera, a pair of StretchSense MoCap Pro  gloves, and the OptiTrack motion capture system. We describe below how each data stream is timestamped and alignd during recording and processing. 

### ZED

#### **Timestamps in recording.** 
During recording, the timestamps of each ZED stream are retrieved using the [ZED API](https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#af18a2528093f7d4e5515b96e6be989d0) method `get_timestamp(sl.TIME_REFERENCE.IMAGE)`. The returned value corresponds to the timestamp, in UNIX nanosecond, at which the image is stored in `/data/{take_id}/raw/{zed_id}.svo`. For each ZED stream, we save the timestamps of all recorded frames and the record start, resulting in N+1 timestamps in total. The timestamps are initially saved in a separate file `/data/{take_id}/raw/{zed_id}.csv` with a structure follows 

* nanoseconds: the header of the unit
* the timestamp of the record start
* the timestamp of the 1st frame 
* the timestamp of the 2nd frame
* ... 
* the timestamp of the N-th frame 

#### **Timestamps in processing.** 
We observe that the timestamps retrieved by the ZED API as above ignore the communication latency, which leads to that the value of timestamps are earlier (smaller) than the real frame times.

To fix this issue, we modify the timestamp of each frame,  via the function in [src/utils/zed.py](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/zed.py),  by adding a positive constant/offset as

```
T(1stframe) = T(start_record) + 1/FPS * 1e9
```
where T(start_record) is the timestamp of  the record start, and T(1st_frame) is the compensated timestamp of the first frame.  This is equivalent to set the timestamp of the 1st frame to the timestamp of the record start, plus the theoretical frame time (1/FPS), and keep the offset between every two consecutive frames unchanged. `* 1e9` converts the time offset to nanoseconds. 

We also noticed that the last timestamp in the raw timestamp file does not correspond to any decoded image frame, in other words, the total amount of frames is one less than the total amount of timestamps. Therefore, the last timestamp is ignored and not saved in the processed timestamp file.  The calibrated timestamps are saved into a separate file `/data/{take_id}/processed/{stream_id}.csv`.  

### Event
#### **Timestamps in recording.** 
The raw event date can be decoded into two formats, including **Contrast Detector (CD) event streams (xypt)** and **event frames (RGB images)**. Each has a slightly different timestamping result, but they are essentially based on the same raw timestamps. 

The raw data times each [Contrast Detector (CD)](https://docs.prophesee.ai/stable/concepts.html#event-generation) event with a releative offset, in microseconds, to the timepoint of the record start. Unfortunately, the raw file  includes only the timestamp of the record start in seconds, so we manually convert it into the UNIX nanosecond by [`/src/utils/event.py`](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/event.py), and attach it to the name of the raw data file, e.g.`event_1662023682456716448.raw`, where the 19-digit number is the converted timestamp of the record start.
<br>

For the xypt stream, the timestamp of each event is calculated by adding its time offset to the timestamp of recording start (initial timestamp). Note that the time offset in xypt is in microseconds, so it has to be converted to nanoseconds first before adding. 

#### **Timestamps in processing.** 
The calculated timestamps are stored directly with the decoded CD event streams in the file `/data/{take_id}/processed/event_xypt.csv`.

```python
timestamp = the initial timestamp + t * 1000
```

`t` is the recorded time offset to the initial timestamp.

For the event frames, timestamps are inferred, instead of being taken in real time, based on the aforementioned initial timestamp and the target decoding FPS (60 by default to align with the FPS of ZED RGBD streams). It is calculated as

```
timestamp = the initial timestamp + 1/FPS * 1e9 * frame number
```

`* 1e9` converts the time offset to nanoseconds. The processed timestamps are stored in the file `/data/{take_id}/processed/event_frames_ts.csv`.


### OptiTrack

#### **Timestamps in recording.** 
The [OptiTrack](https://optitrack.com/) motion capture system has recorded motions of the helmet (equiped by primary subject), the right hand, the left hand and the headband (equiped by the auxiliary subject)  as shown in the [sensor setting](#hardware-and-sensors). The recorded timestamps, motion data streams and the object ids are all stored in `/data/{take_id}/raw/optitrack.csv`.  You can refer to the [reference](https://github.com/h2tc-roboticsx/H2TC/tree/main/doc/data_file_explanation.md#reference) for the OptiTrack ids and their corresponding objects.

Note there exists the latency from the camera exposure to the data reception. We therefore use the timestamp of camera exposure as the timestamp of the frames by subtracting the latency from the original timestamps.
```
timestamp = timestamp of reception - latency
```

#### **Timestamps in processing.** 
In processing, we save the motion streams of the tracked objects into separate `.csv` files via the  `convert` function in [src/utils/optitrack.py](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/optitrack.py). 
Each `.csv` file contains the frame timestamps and their corresponding object trajectory data in the  format `(x, y, z, qx, qy, qz, qw)`, where `(x, y, z)` and `(qx, qy, qz, qw)` denote the object postion and orientation respectively.

### Hand Joint Motion

#### **Timestamps in recording.** 
There are two different schemes of timestamps, device and master, saved in `/data/{take_id}/raw/hand/P1L.csv(P1R.csv)`. They are generated by the glove software [Hand Engine (HE)](https://stretchsense.com/solution/hand-engine/). The device timecode is read from the internal clock of the gloves, while the master timecode is generated according to the host machine (where the Hand Engine runs) clock, as the frame data receives by the Hand Engine from the gloves. The internal clock of the gloves will be periodically calibrated to the host machine clock while being connected. Therefore, these two different timestamps inevitably differ for each frame, since they  record different timepoints using slightly different clocks. 
Nevertheless, we observed that the difference between them is negligible, i.e. unually no greater than 1 frame time (120 FPS by default). In practice, we adopt the device timecode as the timestamp of each frame, because the master timecode has the catastrophic issue of freezing in the first dozens of frames.
<br>

The raw timecode exported by Hand Engine is not a real timestamp, since it uses a base of 120 (same as FPS), instead of the conventional decimal system, to represent the time below a second. A typical HE timecode looks like, e.g. 171442052 which is equivalent to 17 (hour), 14 (minute), 42 (second) and 052 (frame time). 052 is converted to the decimal seconds by `52 * 1/120`. To retrieve the timestamp in UNIX time, we also need the date information, which is stored under the key `startDate` in `/data/{take_id}/raw/hand/P1LMeta.json`. 

#### **Timestamps in processing.** 
In processing, we concatenate the date and the time together to generate a single timestamp and then convert it to UNIX nanoseconds by the  function `format` in [src/utils/he.py](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/he.py). The processed timestamps are stored in the file `/data/{take_id}/processed/left_hand_pose.csv(right_hand_pose.csv)` with a structure like: 

* timestamp: The time derived from the timecode (device) in raw data file `/raw/hand/P1L.csv(P1R.csv)` in the UNIX format.
* hand, index 00-03, middle 00-03, pinky 00-03, ring 00-03 and thumb 01-03: Hand joint motions that are same as what they are in raw data.


### Clock Synchronization

In our recording system, the Hand Engine is hosted on a Windows machine, while other devices are connected, or streamed, to a same Ubuntu machine and hence timestamped based on the same system clock. To align HE data with others, we synchronized the clocks of these two host machines using Precision Time Protocol (PTP). We set up the PTP server on the Ubuntu machine using `ptpd`,  and the PTP client on the Windows machine following the official [guide](https://techcommunity.microsoft.com/t5/networking-blog/windows-subsystem-for-linux-for-testing-windows-10-ptp-client/ba-p/389181) (same as the [provided guide](https://github.com/h2tc-roboticsx/H2TC/blob/main/dev/time_sync/PTP_guide.docx). The timecode is distributed from the Ubuntu host (server) to the Windows host (client). The time drift between the clocks of the two hosts is, after synchronized, normally **around 0.3 milliseconds** and peaking at 3 milliseconds in some rare cases. This accuracy is less comparable to the theoretical accuracy, sub-microsecond range, of PTP. This is most likely due to the PTP client implementation issue in Windows, since we were able to achieve the few-microsecond level accuracy using only the Ubuntu PTP server and client. 

We currently align the streams directly with their timestamps without any further processing. This means that we did not calibrate the timestamp to the same event, e.g. camera exposure, of each stream, because timestamping the recording in this low-level, fine-grained, way is beyond the capacity of the API we used. Nevertheless, the empirical maximum offset among all data streams is **no more than 1 frames at 60 FPS**, as manually evaluated during annotation.

### Modality Alignment

Although all data streams have been timestamped in recording and processing as above, it is impossible for their timestamps to be exactly the same. There exists time drift in millisecond-level between data streams. Therefore, during processing, we use the timestamp of **rgbd0 camera**, the fixed third-person (side) camera, serial number of 17471, as the reference, to align the timestamps of the rest data streams. The resulting timestamp alignment is saved in  `alignment.json`. 


#### How to create an alignment.json file
We use the timestamps of  **rgbd0 camera** as the reference. Therefore, the total number of frames saved in the `alignment.json` is equal to the number of timestamps recorded by the rgbd0 camera. 
 
Given a timestamp of rgbd0 camera and its associated frame indx, for each of other data streams, we use the binary search alogrithm to find their **nearest** timestamp to the timestamp of rgbd0 camera. The nearest timestamp is then used as the timestamp of the frame of the corresponding data stream. Note that the difference between the nearest timestamp and its query rgbd0 camera's timestamp has to be within a threshold, which is currently set to 1/60 * 10e9 nanosecond.

#### The alignment.json file
The file `alignment.json` essentially saves a dictionary whose keys represent the frame indices of rgbd0. Inside each frame (key), there is another dictionary which corresponds to  the multi-modal streams and their corresponding timestamps. These streams are aligned by finding the closest timestamps to the reference timestamp of rgbd0. 
* frame index: Starting from 0
	* key: Stream id
	* value: Timestamp
	
The following is a snapshot of an example alignment.json file:
```
{
    "0": {
        "rgbd0": 1662023682418648047,
        "rgbd1": 1662023682421582524,
        "rgbd2": 1662023682427297843,
        "event": null,
        "left_hand_pose": 1662023682433333504,
        "right_hand_pose": 1662023682424999936,
        "sub1_head_motion": 1662023682430291456,
        "sub1_right_hand_motion": 1662023682430291456,
        "sub1_left_hand_motion": 1662023682434457856,
        "sub2_head_motion": 1662023682430291456
    },
    ...
}
```

#### Data Missing

If the timestamp of a data stream is missing in certain frames, its value is set to be `null` as shown in the above example. Such situation is rare, and is mainly caused by (1) the Optitrack when the tracked object is occluded, (2) by StretchSense gloves when the data transmission is congested, or (3) the event camera when its starting lags slightly behind the rgbd0 camera. 


## OptiTrack Data Processing

### The Coordinate System ID
Since the data collection spans more than three months, our OptiTrack motion capture system in the lab was reset twice. Therefore, there are three different transformations from the throw&catch coordinate frame to the global OptiTrack frame in the raw recordings. We label them as: 

|  Take   | Coordinate System ID| 
|  :----:  | :----:  | 
| 0-2888  | \#0 |            
| 2889-9788  | \#1 |         
| 9789-15000 |\#2   |         

To transfer the object motions captured from different OptiTrack frames to the common [throw&catch coordinate frame](#the-throw-catch-coordinate-frame), we apply coordinate transformation via the 4 x 4 transformation matrices [`origin_transformation_matrix`](#system-id-and-transformation-matrix)  via [optitrack.py](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/optitrack.py)

```python
object_tc_transformation_matrix = np.matmul(origin_transformation_matrix_inverse, object_optitrack_raw_transformation_matrix)
```

`origin_transformation_matrix_inverse` is the inverse matrix of the `origin_transformation_matrix`, which is the transformation matrix of the throw&catch frame w.r.t. the OptiTrack frame.
`object_optitrack_raw_transformation_matrix` is the object's 4 x 4 transformation matrix expressed in the Optitrack frame, and `object_tc_transformation_matrix` is the converted 4 x 4 transformation matrix expressed in the throw&catch frame.


### Extra Rotation
There are some recording issues that should be fixed. Particularly, since takes 1700 onwards, the orientations of gloves, helmet, and headband were checked everytime before the record start. Therefore, no extra rotation is needed for takes from 1700 onwards.<br>
However, for takes 520-1559, the motion of the right hand needs to rotate 90 degrees along the Y axis. For takes 1560-1699, the right hand needs to rotate 180 degrees along the Y axis. For takes 1040-1559, the left hand needs to rotate 45 degrees along the Y axis. For takes from 0-1699, the orientation of the helmet needs to rotate along the Y axis with extra 45 degrees and -180 degrees for the headband. Please see the function `get_t_matrix` in [optitrack.py](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/optitrack.py) for more details.


|  Take   | Devices | Rotation | 
|  :----:  | :----:  |:----:  |
| 520-1559  | right hand |  90 degrees along the Y axis |          
| 1560-1699  | right hand |  180 degrees along the Y axis|        
| 1040-1559 |  left hand |    45 degrees along the Y axis|
|  0-1699 |  helmet|   45 degrees  along the Y axis|
|  0-1699 | headband |   -180 degrees along the Y axis|

```
object_transformation_matrix = np.matmul(object_tc_transformation_matrix, rotY)
```
`object_tc_transformation_matrix` is the converted object transformation matrix introduced as above, `rotY` is the extra transformation for rotating along the Y axis, and `object_transformation_matrix` is the resulted transformation matrix in the throw&catch frame.  

### &#x2022; Note
1. Difference between local and global transformation matrices in `optitrack.csv`. 
The raw `optitrack.csv` file contains `local` (from column 5 to 20) and `global` (from column 21 to 36) transformation matrices. The `local` transformation matrix is expressed relative to the start pose of a recorded sequence. The `global` transformation matrix is with respect to the Optitrack coordinate frame (Y-Up). In the codebase, we only used the `global` transformation matrix for matrix manipulation. 




## Hand Data Processing

As shown below,  we employ the StretchSense MoCap Pro (SMP) gloves to collect the human hand joint motions during throw&catch, and also  the OptiTrack to capture their 6D global motions.

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/hand_devices.png" width = "400" alt="right_hand" style="display: flex; justify-content: center;">
</div>

### Hand Modelling

#### Coordinate Frames in Hand Joint Motion
The coordinate frames for the left (L) and right (R) hands in our hand model are shown below.  Please note that the left hand and the right hand use different coordinate frames. 
<!-- Each joint in the hands has its own frame.  -->
* For the left hand, the X-axis is along the bone link, the Y-axis is perpendicular to the palm, and the Z-axis is perpendicular to the XY plane. 
* For the right hand, the X-axis is against along the bone link, the Y-axis is perpendicular up towards the back of the hand, and the Z-axis is perpendicular to the XY plane. 

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/hand_frame_lx.png" width = "600" alt="hand_frame" style="display: flex; justify-content: center;">
</div>

#### Coordinate Frames  in 6D Global Hand Motion 

The figure below shows the coordinate frame of each hand in OptiTrack. The origin of the frame is roughly located at the center of the back of the palm. The Y-axis is perpendicular up to the back of the  hand, the Z-axis is parallel to the middle finger, and the X-axis is perpendicular to the YZ plane. 

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/hand_motion_frame.png" width = "400" alt="hand_in_catch_throw_frame" style="display: flex; justify-content: center;">
</div>

### Motion Reconstruction
The reconstruced hand motion frames are stored in `data/{take_id}/processed/hand_motion/`. They are primarily generated for visualization and therefore ignored from the data fierarchy.

#### 1. Align hand pose coordinate with the 3D global hand motion coordinate
As introduced above, the coordinate frames of the hand joint motion data differ from those of the 6D global motion. Before reconstruction, we first apply rotations to align the hand pose data to the 6D global coordinate. 

* For the left hand, we first rotate the pose data with -180 degrees along the X-axis, and then rotate it with -90 degrees along the Y-axis, via the function `plot_left_hand `  in [`src/utils/plot_motion.py`](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/plot_motion.py)


```python
rotX = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])    # x -180
rotY = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])     # y -90
```

* For the right hand, we rotate the pose data with 90 degrees along the Y-axis, via the function `plot_right_hand `  in [`plot_motion.py`](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/utils/plot_motion.py) 

```python
rotY = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])     # y 90
```

#### 2. Motion Reconstruction
To reconstruct the hands' global locations, we use the translations in the OptiTrack transformation matrices as the metacarpal locations. 
We reconstruct the entire hand joint pose starting from the metacarpal joint with the aligned hand joint  poses and the  [hand bone length](#hand-bone-length) using forward kinematics.  The details of how to reconstruct the hand motions are provided in  `plot_left_hand` and `plot_right_hand` of `plot_motion.py`.

###   Note

1. Metacarpal joint offset. 
Note that, as a common practice, we did not attach the markers directly to the hand, but fixed markers on a rigid maker board, and then attached them to the back of the hand. As the geometric center of the rigid object does not exactly align with the metacarpal joint of a hand, there is an offset between the reconstructed hand and the actual hand in terms of their spatial positions. However, this offset is minor, and does not change the motion of the hand.

### Hand bone length

Note that the finger bone length we used in `plot_motion.py` is enlarged for visualization. 
We use the following set of bone lengths to reconstruct and visualize the hands in `plot_motion.py`:
```
                     Metacarpal   Proximal   Middle    Distal
thumb  =  [0.25,                  0.11,                           0.06]
index     =  [0.34,                 0.15,          0.08,       0.06]
middle =   [0.33,                0.15,         0.10,        0.07]
ring        =  [0.31,                 0.13,         0.10,        0.06]
pinky     =  [0.3,                   0.08,         0.06,       0.06]
```

Note that this default set of bone lengths is used to save to json files containing hand joint positions (3D XYZ locations) calculated from euler angles using forward kenamatics. The two json files are named as `left_hand_joint_positions.json` and `right_hand_joint_positions.json`, and auto-saved in the `target_path/data/processed/` folder during the data processing procedure. Users can also specify their custom bone lengths, and we provide a script  `src/utils/extract_hand_joint_positions.py` to extract hand joint positions separately from the data processing procedure using the euler angles captured by the Stretchsense MoCap Pro gloves and user-specified bone lengths.

To run `src/utils/extract_hand_joint_positions.py`, users need to specify the root folder path of the take (e.g., data/001000):

```bash
python extract_hand_joint_positions.py --data_root path/to/the_take 
```
The extracted hand joint positions will be saved in json files, with joint positions of left hand in `path/to/the_take/processed/left_hand_joint_positions.json` and those of right hand in `path/to/the_take/processed/right_hand_joint_positions.json`.

20 joint positions are saved, and the json file contains key-value entries with key representing frame number and value being a list of joint positions. In each list, from index 0 to index 19, the saved joint positions are indicated as below:

<table  >
 <tr>
<td  >
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/hand_joint_position_index.png" width=200>
</td>
</tr>
</table  >
 
## Reference

### System ID and Transformation Matrix

Specifically, the takes 0-2888 use system ID \#0. The takes 2889-9788 use system ID \#1. The takes 9789-15000 use system ID \#2.

The 4 x 4 transformation matrix of system ID \#0 is:
```
[[-0.99886939, -0.04535922, -0.01408667, 0.42632084],
 [-0.04514784, 0.99886579, -0.0149642, 0.0984003 ],
 [ 0.01474855, -0.01431195, -0.99978858, 7.67951849],
 [ 0., 0., 0., 1. ]]
```

The 4 x 4 transformation matrix of system ID \#1 is:
```
[[-9.99963351e-01, 8.30436476e-03, 2.13574045e-03, 1.92400245e-01],
 [ 8.31340413e-03, 9.99956270e-01, 4.25134508e-03, 6.55417571e-02],
 [-2.10037766e-03, 4.26893351e-03, -9.99988700e-01, 2.17126483e+00],
 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
```

The 4 x 4 transformation matrix of system ID \#2 is:
```
[[-0.99997146, 0.00456379, 0.00601402, 0.19729361],
 [ 0.00454136, 0.99998255, -0.00373799, 0.06776005],
 [-0.00603099, -0.00371067, -0.99997492, 2.48060394],
 [ 0., 0., 0., 1. ]]
```

