# Data Processing Technical Details
[The dataset H<sup>2</sup>TC](https://lipengroboticsx.github.io/) contains multi-modal cross-device raw data streams. 
To make the dataset easier to use, we have developed the [processor source code](https://github.com/lipengroboticsx/H2TC_code/tree/main/src) and provided the [data processing document](https://github.com/lipengroboticsx/H2TC_code/tree/main#data-processing) to help readers get aligned and common-format data. 
[tbd: polishing]Considering readers may want to know the processing technical details, we introduce them in this document to auxiliarly explain the [source code](https://github.com/lipengroboticsx/H2TC_code/tree/main/src). For a more thorough explanation of the raw/processed files mentioned on this page, see [/doc/data_file_explanation.md](https://github.com/lipengroboticsx/H2TC_code/tree/main/doc/data_file_explanation.md). 

Here is an overview of this document:

* [**Workspace**](#our-workspace): introduces The [used multi-modal devices](#used-devices) and [the coordinate setting](#our-coordinate-setting). 
* [**Timestamping and Data Alignment**](#timestamping-and-data-alignment): introduce how [ZED RGBD](#zed-rgbd), [Event](#event), [Optitrack](#optitrack) and [Gloves Hands Pose](#gloves-hands-pose) data streams are timestamped and [aligned](#alignment) in recording and processing. 
    <!-- * [Clock Synchronization](#clock-synchronization) -->
* [**OptiTrack Data Processing**](#optitrack-data-processing): auxiliarly explains the original optitrack coordinate and how to transfer it to The coordinate. 
* [**Hand Data Processing**](#hand-data-processing): auxiliarly explains the gloves' [hands pose data coordinates](#•-hand-pose-data-coordinate-frame) and how to [reconstruct](#•-motion-reconstruction) hands' motion. 

## The Workspace
### Used Devices
We use a variety of specialized motion tracking and visual streaming devices to capture The dataset as illustrated below. 
<!-- The details about The recording framework are introduced in [Sec. 4 Recording Framework in The paper]().  -->

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/hardware.png" width = "1000" alt="hardware" />

| Device | Manufacturer | Recording Content |FPS |Resolution  |
|:-----|:-----:|:-----:|:-----:|:-----:|
| ① Gloves | [StretchSense MoCap Pro](https://stretchsense.com/) | Hand Pose | 120 | - |
| ②⑤ Tracker | [OptiTrack](https://optitrack.com/) |  Human Motion | 240 | - |
| ③ Event Camera | [Prophesee](https://www.prophesee.ai/) | Event | - | 1280x720 |
| ④ ZED Camera | [Stereolabs](https://www.stereolabs.com/zed-2/) |  RGB-D | - | 1280x720 |


### The Coordinate Setting 
[tbd: introduce more systematically]

#### The throw-catch coordinate frame

As shown below, the **origin** of The throw-catch zone refers to the bottom-left corner of the entire throw-catch zone. The coordinate **axes** are set up as follows: XZ plane is parallel to the ground with Z-axis along the *5 m* (*2m + 1m + 2m*) side and X-axis along the *2 m* side. Y-axis is perpendicular up to the XZ plane. 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/workspace.png" width = "600" alt="workspace" />

#### The Headband and helmet coordinate frame 
[tbd: check the coordinate]
The figure below shows the defined coordinates of the headband and the helmet. The coordinate of the helmet is the same as that of the throw-catch zone coordinate. While the coordinate of the headband has a rotate of along the Y-axis counterclockwise with 180 degrees. 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/D6403D046FEE97803D912C8DB100C11F.png" width = "200">

In a real scenario, when the primary subject is wearing the helmet and the auxiliary subject is wearing the headband, the coordinate systems will look like as in the following picture:

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/A4A102E4F59E1D43D95D22ABB44AD561.png" width = "200">


<br>

## Timestamping and Data Alignment

The recording system consists of  3 ZED RGBD cameras, 1 Prophesee event camera, 1 StretchSense data gloves, and 1 OptiTrack motion capture system. We describe below how each data stream is timestamped and alignd in recording and processing. 

[tbd: timestamps in xxx to subsection]

### &#x2022; ZED RGBD

#### **Timestamps in recording.** 
In The recording, each ZED RGBD camera timestamp is retrieved by calling the [ZED API method](https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#af18a2528093f7d4e5515b96e6be989d0) `get_timestamp(sl.TIME_REFERENCE.IMAGE)`. The returned value corresponds to the time, in UNIX nanosecond, at which the entire image was stored in `/data/{take_id}/raw/{zed_id}.svo`. For each RGBD stream, we record the timestamp of each frame and the beginning of the recording, resulting in N+1 timestamps in total. The timestamps are initially stored in a separate file `/data/{take_id}/raw/{zed_id}.csv` with a structure as
* nanoseconds: header of the unit
* the timestamp of the beginning of recording
* the timestamp of the 1st frame 
* the timestamp of the 2nd frame
* ... 
* the timestamp of the N-th frame 

#### **Timestamps in processing.** 
 We observed that the timestamp retrieved by previously mentioned ZED API ignores the communication time resulting in the value of timestamps earlier (smaller) than the real frame time.

To fix this issue, we compensate the timestamp of each frame by adding a positive constant of (addressed in the script [zed.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/zed.py) ): 

```
T(1stframe) = T(startrecording) + 1/FPS * 1e9
```
T(startrecording) is the timestamp of start recording, and T(1stframe) is the timestamp of the real 1st frame. 
This is equivalent to set the timestamp of 1st frame to the timestamp of start recording plus the theoretical frame time (1/FPS), and keep the offset between every two consecutive frames unchanged. 
<br>
We also notice that the last timestamp in the raw timestamp file doesn't correspond to any decoded frame image, in other words, the total amount of frames is one less than the total amount of timestamps. Therefore, the last timestamp is ignored and not saved in the processed timestamp file. 
<br>
After processing, the calibrated timestamps will be saved into a separate file `/data/{take_id}/processed/{stream_id}.csv`.  

### &#x2022; Event
#### **Timestamps in recording.** 
Event raw data can be exported into two alternative formats: **event streams (xypt) and event frames (RGB images)**. Each has a slightly different timestamp result, but they are essentially based on the same timestamps. 
<br>
The Event raw data file timestamps each [Contrast Detector (CD) event](https://docs.prophesee.ai/stable/concepts.html#event-generation) with an offset, in microseconds, to the timepoint of recording started. Unfortunately, the raw file only includes the timestamp of recording started in seconds, so we manually took one in UNIX nanoseconds in The recording script [`/src/utils/event.py`](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/event.py) and attached it in the name of the raw data file, e.g., `event_1662023682456716448.raw`, where the 19-digit number is the timestamp of recording started.
<br>
For xypt format, the timestamp of each event is calculated by adding its time offset (t) to the timestamp of recording started (initial timestamp). Note that the time offset in xypt is in microseconds, so it has to be converted to nanoseconds first before adding. 

#### **Timestamps in processing.** 
The calculated timestamps are stored directly with the decoded event streams in the file `/data/{take_id}/processed/event_xypt.csv`.

```python
timestamp = initial timestamp + t * 1000
```

For event frames, timestamps are inferred, instead of taken in real time, based on the aforementioned initial timestamp and the target decoding FPS (60 by default to align with the FPS of ZED RGBD streams). It is calculated as

```
timestamp = initial timestamp + 1/FPS * 1e9 * frame number
```

`* 1e9` converts the time offset to nanoseconds. The processed timestamps are stored in the file `/data/{take_id}/processed/event_frames_ts.csv`.


### &#x2022; OptiTrack

#### **Timestamps in recording.** 
The [OptiTrack](https://optitrack.com/) motion capture system has recorded helmet (equiped by primary subject), the right hand, the left hand and headband (equiped by auxiliary subject) motions as shown in [used devices](#used-devices). The recorded timestamps, motion data and object IDs are stored in `/data/{take_id}/raw/optitrack.csv`.  You can check the OptiTrack IDs and their corresponding objects [here](https://github.com/lipengroboticsx/H2TC_code/tree/main/doc/data_file_explanation.md#reference). 

Note there exists the latency from camera exposure to the reception. We then calculate the timestamp of camera exposure as the timestamp of the frame by the current timestamp minus the above latency.
```
timestamp = timestamp of reception - latency
```

#### **Timestamps in processing.** 
In processing, we save the tracked motion data of the objects described above into separate .csv files via `convert` function in script [optitrack.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/optitrack.py). 
<!-- Each .csv file contains the frame timestamps and the corresponding trajectory data in [tum pose format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) (tx, ty, tz, qx, qy, qz, qw).  -->

### &#x2022; Gloves Hands Pose

#### **Timestamps in recording.** 
There are two different schemes of timestamps, device and master, in the output files, `/data/{take_id}/raw/hand/P1L.csv(P1R.csv)`, generated by gloves' software [Hand Engine (HE)](https://stretchsense.com/solution/hand-engine/). The device timecode is read from the internal clock of the gloves, while the master timecode is generated according to the host PC (where Hand Engine runs) clock as the frame data received by Hand Engine from the gloves. The internal clock of gloves will be periodically calibrated to the host PC clock during connected. Therefore, these two different timestamps inevitably differ for the same frame, since they are recording different timepoints using slightly different clocks. Nevertheless, we observed that the difference between them is negligible, i.e., normally no greater than 1 frame time (120 FPS by default). In practice, we adopt the device timecode as the timestamp of each frame, because the master timecode has the catastrophic issue of freezing in the first dozens of frames.
<br>
Raw timecode exported by Hand Engine is not a real timestamp, since it uses a base of 120 (same as FPS), instead of the conventional decimal system, to represent the time below a second. A typical HE timecode looks like, e.g., 171442052 is equivalent to 17 (hour), 14 (minute), 42 (second) and 052 (frame time). 052 is converted to the decimal seconds by `52 * 1/120`. To retrieve the timestamp in UNIX time, we also need the information of date, which is stored under the key `startDate` in another file `/data/{take_id}/raw/hand/P1LMeta.json`. 

#### **Timestamps in processing.** 
In processing, we concatenate the date and the time together to generate a single timestamp and then convert it to UNIX nanoseconds (see function `format` in script [he.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/he.py). The processed timestamps are stored in the file `/data/{take_id}/processed/left_hand_pose.csv(right_hand_pose.csv)` with a structure like: 
* timestamp: the unix format time derived from timecode (device) in raw data file `/raw/hand/P1L.csv(P1R.csv)`.
* hand, index 00-03, middle 00-03, pinky 00-03, ring 00-03 and thumb 01-03: same as what they are in raw data.



<!-- ### Clock Synchronization

Only Hand Engine is hosted on a Windows machine, while the other devices are connected, or streamed, to the same Ubuntu machine and hence timestamped based on the same system clock. To align HE data with others, we synchronized the clocks of these two host machines using Precision Time Protocol (PTP). We set up the PTP server on Ubuntu using `ptpd` and the PTP client on Windows following the official [guide](https://techcommunity.microsoft.com/t5/networking-blog/windows-subsystem-for-linux-for-testing-windows-10-ptp-client/ba-p/389181) (same as the copy `/dev/time_sync/PTP_guide.docx`). Timecode is distributed from the Ubuntu host (server) to the Windows host (client). The time drift between the clocks of these hosts is, after synchronized, normally **around 0.3 milliseconds** and peaking at 3 milliseconds in some rare cases. This accuracy is less comparable to the theoretical accuracy, sub-microsecond range, of PTP. This is most likely due to the PTP client implementation issue of Windows, since we were able to achieve the few-microsecond level accuracy using only the Ubuntu PTP server and client. 

We currently align the streams directly with their timestamps without any further processing. This means that we didn't calibrate the timestamp to the same event, e.g., camera exposure, of each stream, because timestamping the recording in this low-level, fine-grained, way is beyond the capacity of the API we used. Nevertheless, the empirical maximum offset among all data streams is **no more than 1 frames at 60 FPS**, as manually evaluated during annotation. -->

### &#x2022; Alignment

Although all data streams have been timestamped in recording, it is impossible for their timestamps to be exactly the same. There exists time drift in millisecond-level between data streams. Therefore, during processing, we use the timestamp of **rgbd0 camera**, the fixed third-person (side) view camera, serial number: 17471, as the reference, and align the timestamps of the rest data streams to it. The resulting timestamp alignment is saved in a file called `alignment.json`. 

#### How to create an alignment.json file
We use the timestamps of **rgbd0 camera** as the reference. Therefore, the total number of frames saved in the `alignment.json` is equal to the number of timestamps recorded by rgbd0 camera. 
<br>
Given a timestamp of rgbd0 camera and its associated frame number, for each of other data streams, we use the binary search alogrithm to find their **nearest** timestamp to the timestamp of rgbd0 camera. This nearest timestamp is then used as the timestamp of that frame of the other data stream. Note that the difference between the nearest timestamp and its query rgbd0 camera's timestamp has to be within a threshold, which is currently set to 1/60 * 10e9 nanosecond.

#### The alignment.json file
[tbd: rewrite same as file_explanation]
`Alignment.json` file essentially saves a dictionary whose keys represent the frame indices. Inside each frame (key), there is another dictionary which indicates the multi-modal streams and their corresponding timestamps. These streams are aligned by finding the closest timestamp to the reference timestamp (`rgbd0`). 
* frame index: starting from 0
	* key: stream id
	* value: timestamp
	
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
If the timestamp of a data stream is missing in certain frames, its value will be `null` as shown in the above example. Such situation is rare, and it is mainly caused by 1) the Optitrack when the tracked object is occluded; or 2) by StretchSense gloves when the data transmission is congested; or 3) the open of the event camera lags slightly behind the rgbd0 camera. Therefore, the corresponding timestamp is missing.



## OptiTrack Data Processing
### &#x2022; The Coordinate System ID
As The data collection spans more than three months, during which, The throw-catch zone in the lab was moved twice. Therefore, there are **THREE different coordinates** in The recordings. We label them as: 

|  Take   | Coordinate System ID| 
|  :----:  | :----:  | 
| 0-2888  | \#0 |            
| 2889-9788  | \#1 |         
| 9789-12905 |\#2   |         

To transfer these different optitrack coordinates to [the throw-catch coordinate](#our-coordinate-setting), we apply coordinate transformation via the 4 x 4 transformation matrices captured in the original optitrack system (addressed in the script [optitrack.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/optitrack.py)). The specific transformation matrices are shown [here](#system-id-and-transformation-matrix). 
<!-- 
```
object_tc_transformation_matrix = np.matmul(origin_transformation_matrix, object_optitrack_raw_transformation_matrix)
``` -->

### &#x2022; Note
1) Additional transformation. We need additional transformation to correct some parts of tracking data in some takes (addressed in the script [optitrack.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/optitrack.py)). They are: [tbd: polishing + code ??]

|  Take   | Part | Rotation | 
|  :----:  | :----:  |:----:  |
| 520-1559  | right hand |  90 degrees |          
| 1560-1699  | right hand |  180 degrees |        
| 1040-1559 |  left hand |    45 degrees |
|  0-1699 |  helmet|   45 degrees  |
|  0-1699 | headband |   -180 degrees|

2) Difference between local and global transformation matrices in `optitrack.csv`. 
The raw `optitrack.csv` file contains `local` (from column 5 to 20) and `global` (from column 21 to 36) transformation matrices of the optitrack system. The `local` transformation matrix is expressed relative to the start pose of a recorded sequence. The `global` transformation matrix is with reference to Optitrack world coordinate (Y-Up). In The codebase, we only used the `global` transformation matrix for matrix manipulation. 


<!-- 
For takes from 520-1559, right hand needs to rotate 90 degrees (addressed in the script optitrack.py)
For takes from 1560-1699, right hand needs to rotate 180 degrees (addressed in the script optitrack.py)
For takes from 1040-1559, left hand needs to rotate 45 degrees (addressed in the script optitrack.py)
For takes from 0-1699, the orientation of the helmet and headband needs to be corrected if using their orientation (rotate along Y axis with extra 45 degrees for the headband, and -180 degrees for the helmet).

The following code adds extra rotation for the target object, where `rotY` is the extra rotation expressed in the form of a 4 x 4 transformation matrix

```
object_tc_transformation_matrix = np.matmul(object_tc_transformation_matrix, rotY)
``` -->

<!-- ### Integrate optitrack motion with hand engine pose

For each frame (e.g., if 60 fps, then 300 frames in total for a 5s long motion sequence), the translation, i.e., `x,y,z` positions in the `object_tc_transformation_matrix` are used as the metacarpal joint of the hand. Starting from the metacarpal joint, the entire hand is then recovered using forward kinematics with captured hand joint angles (euler angles) and the defined bone length. The detail of how to recover the entire hand can refer to the functions `plot_left_hand` and `plot_right_hand` in `plot_motion.py`.
Note that the left hand uses a right-handed coordinate system and the right hand uses a left-handed coordinate system. -->

## Hand Data Processing
As shown below, the pose motions of both hands are collected by StretchSense MoCap Pro gloves, and their 3D global motions are captured by OptiTrack as well. 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/hand_devices.png" width = "400" alt="right_hand" style="display: flex; justify-content: center;">
<!-- You can check [data_file_explanation.md](https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/data_file_explanation.md/#data) to get each term's meaning.  -->

### &#x2022; Hand Pose Data Coordinate 
[tbd: double-check the frame again!!]

The left (L) and right (R) hand coordinates are shown below. 
Please note that **the left hand and the right hand use different coordinate frames**. 
Each joint in the hands has its own frame. 
* For the left hand, the X-axis is along the bone, the Y-axis is perpendicular to the palm, and the Z-axis is perpendicular to the XY plane. 
* For the right hand, the X-axis is against along the bone, the Y-axis is perpendicular up towards the back of the hand, and the Z-axis is perpendicular to the XY plane. 


<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/hand_frame_lx.png" width = "800" alt="hand_frame" style="display: flex; justify-content: center;">

### &#x2022; 3D Global Hand Motion Data Coordinate 

The figure below shows the coordinate frame of the OptiTrack captured hand motion. The origin of the frame is roughly located at the center of the back of the palm. Y-axis is perpendicular up to the back of the  hand, Z-axis is parallel to the finger tip direction, and X-axis is perpendicular to the YZ plane. 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/hand_motion_frame.png" width = "700" alt="hand_in_catch_throw_frame" style="display: flex; justify-content: center;">

### &#x2022; Motion Reconstruction
The motion reconstruction results are stored in folder `{take_id}/processed/hand_motion/`. 

#### 1. Align hand pose coordinate with the 3D global hand motion coordinate
As introduced above, the coordinate frames of the hand pose data differs from that of the 3D global motion. Before reconstruction, we first use several rotations to align the hand pose data to the 3D global coordinate. 
* For the left hand, we first rotate the pose data -180 degrees along the X-axis, and then rotate it -90 degrees along the Y-axis. As shown in the `plot_left_hand ` function in [`plot_motion.py`](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/plot_motion.py), the rotation matrices are:
```
rotX = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])    # x -180
rotY = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])     # y -90
```
* For the right hand, we rotate the pose data 90 degrees along the Y-axis.  As shown in the `plot_right_hand ` function in [`plot_motion.py`](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/plot_motion.py), the rotation matrix is:
```
rotY = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])     # y 90
```

#### 2. Reconstruction
For hands' global locations, we use the translations from the OptiTrack transformation matrices as the metacarpal locations. 
For hands' entire poses, we reconstruct the entire hand pose starting from the metacarpal joint with the aligned hand joints poses and the defined [hand bone length](#hand-bone-length) using forward kinematics. 


### &#x2022; Note

1. Metacarpal joint offset. 
Note that, as a common practice, we did not attach the markers directly to the hand, but fixed markers on a rigid object, and then attached the rigid object to the back of the hand. As the geometric center of the rigid object does not exactly align with the metacarpal joint of a hand, there is an offset between the reconstructed hand and the actual hand in terms of their spatial positions. However, this offset is minor, and does not change the motion of the hand.

### Hand bone length

Note that the finger bone length we used for visualization purpose in `plot_motion.py` is enlarged. 
We use the following set of bone lengths to reconstruct and visualize the hands in `plot_motion.py`:
```
         Metacarpal Proximal Middle Distal
thumb =  [0.25,     0.11,           0.06]
index =  [0.34,     0.15,    0.08,  0.06]
middle = [0.33,     0.15,    0.10,  0.07]
ring =   [0.31,     0.13,    0.10,  0.06]
pinky =  [0.3,      0.08,    0.06,  0.06]
```

<!-- #### Paper
The bone lengths in the paper are measured from an actual hand (TODO, better to provide an average bone length model)
```
         Metacarpal Proximal Middle Distal
thumb =  [6.0,      4.0,            3.5]
index =  [8.0,      5.5,     3.0,   2.5]
middle = [8.0,      6.0,     3.5,   2.7]
ring =   [7.5,      5.5,     3.3,   2.5]
pinky =  [6.5,      4.5,     2.5,   2.5]
``` -->




## Reference
### System ID and Transformation Matrix

Specifically, takes 0-2888 use system ID \#0; takes 2889-9788 use system ID \#1; and takes 9789-12905 use system ID \#2.

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

