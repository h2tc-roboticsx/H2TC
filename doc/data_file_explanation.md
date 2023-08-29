<!-- # Files Explanation  -->
# Data Formats
This document explains the data  hierarchy and each data file stored in our dataset [H<sup>2</sup>TC](https://lipengroboticsx.github.io/). <br>

Here is a quick overview of the files involved in the dataset.  To distinguish, hereafter, the **file** presents **bold**, while the ***folder*** presents **bold** and *italic*. For those specific digits, such as subject IDs or device IDs, the [reference](#reference) explains their meanings.  <br>
<!-- ## Overview -->
* [***data/***](#data)
  * ***{take ID}/***: the take folder named by the take id, e.g. 000000
    * ***raw/***: the raw data directly exported from multiple recording sensors, e.g. ZED cameras
    * ***processed/***: the formatted data derived from the raw data
* [***annotations/***](#annotations)
    * **{take ID}.json**: the annotation result for the take id, e.g. 000000
* [***objects/***](#objects): the scanned object models
    * **object_name.stl**: the scanned object model, e.g. `apple.stl`
* [**log.xlsx**](#supporting-files): the logbook with the recording parameters of all takes
* [**subjects.csv**](#supporting-files): the list of the subjects participating in the dataset
* [**objects.csv**](#supporting-files): the list of used objects


## Data
<!-- <details><summary>Explanation about all files in data folder</summary> -->
Before diving into the data details below, we suggest users check the [data processing tutorial](https://github.com/lipengroboticsx/H2TC_code/tree/main/#data-processing) first to capture how we process the dataset.
Note that the raw data files and their contents that are **not** used in the data processing are <u>underlined</u>. 

* ***{take ID}/***: the take folder named by the take id, e.g. 000000.
  * ***raw/***: the raw data directly exported from the multiple recording sensors, e.g. ZED cameras and OptiTrack. 
    * ***hand/***: the hand pose data recorded by [Hand Engine](https://stretchsense.com/solution/hand-engine/) (i.e. the software driver of the Prophesee event camera). Note the details on the coordinate frames are introduced [here](https://github.com/lipengroboticsx/H2TC_code/tree/main/docProphesee/processing_techdetails.md#•-hand-pose-data-coordinate).
      * **P1L.csv** / **P1R.csv**: The sensor readings and hand pose data of the left (L) and right (R) hand respectively.   Specifically, each file contains by column
        * <u>Capacitance 0-15</u>: The capacitance reading in picofarad (pF) of the wrist sensor and 15 splay sensors.
        * <u>IMU 1-10</u>: N/A (the glove version we used doesn't contain IMU sensors).
        * Timecode (device): The timecode from the clock inside the gloves ticks as data starts generating.
        * <u>Timer (device)</u>: The glove's internal timer that ticks every 1/120 s.
        * <u>Timecode (master)</u>: The timecode from the clock of the host machine where [Hand Engine](https://stretchsense.com/solution/hand-engine/) runs.
        * hand: The wrist joint pose in Euler-angle format (XYZ degrees). 
        * index 00-03, middle 00-03, pinky 00-03, ring 00-03, thumb 01-03: The hand joint pose in Euler-angle format (XYZ degrees). Each joint rotates w.r.t. its parent in the kinematic tree. Please refer to our [paper](toadd) for the details of our hand model.  
      * **P1LMeta.json** / **P1RMeta.json**: The metadata about the hardware and recording setting. 
      * <u>**P1L.cal** / **P1R.cal**</u>: The calibration data. 
      * <u>**P1L.fbx** / **P1R.fbx**</u>: The hand model saved in the .fbx file format.  
    * **17471.svo / 24483054.svo / 28280967.svo**: The recording files of ZED RGB-D cameras with videos of the left and right views and metadata. Check [ZED document](https://www.stereolabs.com/docs/video/recording/) for more information. The digits denote the serial number (SN) of the recording devices. Specifically, 17471 is the fixed camera of the third-person side view, 24483054 is the dynamic camera of the egocentric view, and 28280967 is the fixed camera of the egocentric view  (or equivalently the third-person back view).
    * **17471.csv / 24483054.csv / 28280967.csv**: The timestamps of the frames in captured RGB-D streams. The digits, similarly as above, refer to the SN of the recording devices. Note that the timestamps start from the beginning of a recording, so the second timestamp is the actual timestamp of the first frame. Therefore, the first timestamp in this file is always discarded when processing or aligning the streams. Specifically, each file contains by row
      * nanoseconds: The header of the unit
      * the timestamp of the beginning of the recording
      * the timestamp of the 1st frame 
      * the timestamp of the 2nd frame
      * ... 
      * the timestamp of the N-th frame 
    *  **event_{timestamp}.raw**: The raw output of the event camera. It includes general metadata and event information encoded in [EVT3.0](https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html?highlight=data%20format) format. Refer to [Prophesee documentation](https://docs.prophesee.ai/stable/index.html) for more details. The digits `{timestamp}` in the file name log the timestamp of the starting recording, which is used later in the processing pipeline to align the event stream with other streams. 
    * <u>**event.bias**</u>: The event sensor settings. Check [event bias](https://docs.prophesee.ai/stable/hw/manuals/biases.html) for more detailed information.
    * **optitrack.csv**: The tracking information recorded by [OptiTrack](https://optitrack.com/). Specifically, the file contains by column
      * object id: The id of the tracked object. Check [OptiTrack object ID](#reference) for the id of each object in OptiTrack. 
      * <u>frame</u>: The frame index maintained by the OptiTrack server.
      * <u>mean error</u>: The mean error of localization.
      * timestamp: The frame timestamp that is calculated by subtracting the latency. 
      * <u>latency</u>: The time from the exposure of the cameras to the reception of data. 
      <!-- on the client.  -->
      * <u>local transformation matrix</u>: The local pose transformation matrix w.r.t. the start pose frame. The matrix has a size of 4x4 and is unfolded by row.
      * global transformation matrix: The global pose w.r.t. the OptiTrack world frame. The matrix has a size of 4x4 and is unfolded by row. 
  * ***processed/***: The formatted data derived from raw data. 
    * ***rgbd0/***: The left-view RGB images and normalized depth images from the fixed third-person (side) view ZED  camera (SN: 17471).
      * **left_{frame_ID}.png**: The RGB images produced by the left view camera of ZED. Left-view RGB images align with the corresponding depth maps, which is set internally by [ZED SDK](https://www.stereolabs.com/docs/video/recording/). The digits `{frame_ID}` represent the frame index. 
      * **depth_{frame_ID}.png**: The normalized depth images. They align with the corresponding left-view RGB images. The digits `{frame_ID}` represent the frame index. They should be only used for visualization since they do not contain real depth values. 
      * **depth.npy**: The 3-dimensional numpy array holding the unnormalized depth estimation of each frame. It aligns with the left-view RGB images by [ZED SDK](https://www.stereolabs.com/docs/video/recording/). The unit of depth is millimeter. 
        * dimension 0: The frame index
        * dimension 1: The height in range [0, 719] where 0 corresponds to the top of the image
        * dimension 2: The width in range [0, 1279] where 0 corresponds to the left of the image
    * ***rgbd1/***: similar data as *rgbd0/*  but from the dynamic egocentric ZED  camera (SN: 24483054).
    * ***rgbd2/***: similar data as *rgbd0/*  but from the fixed third-person (back) ZED  camera (SN: 28280967). 
    * **rgbd0_ts.csv**: The timestamps of rgbd0 frames. This is the same as the timestamp file `/raw/17471.csv` but  is renamed to be more interpretable.
    * **rgbd1_ts.csv**: similar data as `rgbd0_ts.csv` but for rgbd1 frames.
    * **rgbd2_ts.csv**: similar data as `rgbd0_ts.csv` but for rgbd2 frames.
    * **event_xypt.csv**: The decoded [Contrast Detector (CD) events](https://docs.prophesee.ai/stable/concepts.html#event-generation).
      * x: The width in range [0, 1279], where 0 corresponds to the left of the image.
      * y: The height in range [0, 719], where 0 corresponds to the top of the image.
      * p: polarity
        * 0: The corresponding CD event is off, i.e. detecting a negative contrast that light changes from lighter to darker
        * 1: The corresponding CD event is on, i.e. detecting a positive contrast that light changes from darker to lighter
      * t: The timestamp of the light change.
    * ***event/***: The frame-based visualization of events.
      * **{frame_ID}.jpg**: A frame visualization of events accumulated in a fixed period (1/60 second for 60 FPS). For each pixel, it is initialized with the background color (dark blue). If the polarity of the event located at the pixel occurs positive (negative) over the period, the pixel changes to white (light blue). Check [Event Generation](https://docs.prophesee.ai/stable/concepts.html#event-generation) for more details. The digits `{frame_ID}` represent the frame index. 
    * **event_frames_ts.csv**: The timestamps of event frames in the above directory *event/*.
    * **left_hand_pose.csv**: the pose data of the left hand produced based on `/raw/hand/P1L.csv`.
      * timestamp: the unix format time derived from timecode(device) in raw data file `/raw/hand/P1L.csv`.
      * hand, index 00-03, middle 00-03, pinky 00-03, ring 00-03 and thumb 01-03: same as what they are in raw data.
    * **right_hand_pose.csv**: similar pose data as `left_hand_pose.csv` but for right hand produced based on `/raw/hand/P1R.csv`.
    * **sub1_head_motion.csv**: the position and orientation of the subject1's helmet corresponding to the OptiTrack object 115 in `/raw/optitrack.csv`.
      * timestamp: the same timestamp as in raw data file `/raw/optitrack.csv`.
      * x, y, z: the position of the object in the throw-catch coordinate system (introduced [here](https://github.com/lipengroboticsx/H2TC_code/tree/main/doc/processing_techdetails.md#the-coordinate-setting)). Note that the coordinate is Y-up, and the origin point is at the bottom left. [tbd: frame]
      * qx, qy, qz, qw: the quaternion orientation of the object in The coordinate system. 
    * **sub1_left_hand_motion.csv**: the subject1's  left hand motion (OptiTrack object id is 117 in `/raw/optitrack.csv`). Similar data structure as `sub1_head_motion.csv`. 
    * **sub1_right_hand_motion.csv**: the subject1's  right hand motion (OptiTrack object id is 116 in `/raw/optitrack.csv`). Similar data structure as `sub1_head_motion.csv`. 
    * **sub2_head_motion.csv**: the subject2's  head motion (OptiTrack object id is 118 in `/raw/optitrack.csv`). Similar data structure as `sub1_head_motion.csv`. 
    <!-- a dictionary whose keys represent the frame indices. The corresponding value for each key is a mapping between stream id and its timestamp. In each frame (key), we align each stream timestamp to the frame reference timestamp (`rgbd0`) by finding the closest one to the reference. -->
    * [tbd: rewrite: alignment] **alignment.json**: 
    a dictionary whose keys represent the frame indices. Inside each frame (key), there is another dictionary which indicates the multi-modal streams and their corresponding timestamps. These streams are aligned by finding the closest timestamp to the reference timestamp (`rgbd0`).
      * frame index: starting from 0
        * key: stream id
        * value: timestamp
<!-- </details> -->
<br>

## Annotations
<!-- <details><summary>Exlanation about annotation labels </summary> -->
Before reading the file details below, please check the [annotation document](https://github.com/lipengroboticsx/H2TC_code/tree/main/#annotator) first to capture how we annotate data. 
* **{take ID}.json**: metadata and the annotation data of the take. 
  * status: annotation status.
    * 1: finished
    * 0: not finished
    * -1: problematic, need further inspect
  * take_id: the index of the take.
  * object: the name of the object being thrown or caught in this take.
  * catch_result: the result of catching.
    * 1: success
    * 0: failed
  * sub1_cmd: the command given to the subject 1 to instruct the behavior of throwing or catching. See [Sec. Protocol in the paper]() for more instructions details. Please refer to [log.xlsx](#supporting-files) for the explanation of the attributes below.
    * subject_id: the id of the subject 1
    * hand: grasp mode, `single` or `both`
    * position: 2 dimensional location (x, z)
    * action: `throw` or `catch`
  * sub2_cmd: the command given to the subject 2. See [Sec. Protocol in the paper]() for more instructions details and check [Figure 3 in the paper]() for hand vertical/horizontal locations.  
    * subject_id: the id of the subject 2
    * hand: grasp mode, `single` or `both`
    * position: 2 dimensional location (x, z)
    * action: `throw` or `catch`
    * hand_vertical: relative vertical position of the hand
      * value: `overhead`, `overhand`, `chest` or `underhand`
    * hand_horizontal: relative horizontal position of the hand
      * value: `right`, `middle` or `left` 
      * this attribute is only given when the action to be performed by the subject 2 is `catch`.
    * throwing_speed: the speed of throwing away the object 
      * value: `fast`, `normal` or `slow`
      * this is only given when the action to be performed by the subject 2 is `throw`
  * throw: annotation data for the moment `throw`.
    * hand: grasp mode for throwing the object
      * value: `left`, `right` or `both`
      * this is different from "hand" in the subjects' command in two ways: 1) It replaces the `single` with `left` and `right`; 2) It represents the thrower's hand action in the experiment, which may be different from the command. 
    * hand_vertical_thrower: the relative vertical position of the hand of the thrower at the moment `throw`.
      * value: `overhead`, `overhand`, `chest` or `underhand`
    * hand_vertical_catcher: similar to above "hand_vertical_thrower" but for the catcher
    * hand_horizontal_thrower: the relative horizontal position of the hand of the thrower at the moment `throw`.
      * value: `right`, `middle` or `left` 
    * hand_horizontal_catcher: similar to above "hand_horizontal_thrower" but for the catcher
    * position_thrower: the location of the thrower in The coordinates at the moment `throw`. 
      * x, z: 2 dimensional position
      <!-- * this is different from the `position` in the previous subject command like "sub2_cmd" -->
    * position_catcher: the location of the catcher in The coordinates at the moment `throw`.
      * x, z: 2 dimensional position
    * object_flying_speed: the flying speed of the thrown-away object in an unit of m/s.
    * time_point: the timestamps of streams at the moment `throw`. 
      * stream id
        * frame: the frame index of the corresponding stream
        * timestamp: the UNIX timestamp in nanoseconds of the corresponding stream 
  * catch: annotation data for the moment `catch`
    * hand: grasp mode for catching the object
      * value: `left`, `right` or `both`
    * hand_vertical: the relative vertical position of the hand of the catcher at the moment `catch (stable)`.
    * hand_horizontal: the relative horizontal position of the hand of the catcher at the moment `catch (stable)`.
    * position: the location of the catcher in The coordinates at the moment `catch (touch)`.
    * time_point_touch: the timestamps of streams when catching happens (i.e., when the hand of the catcher first touches the object in the flight).
      * stream id
        * frame: the frame index of the corresponding stream
        * timestamp: the UNIX timestamp in nanoseconds of the corresponding stream 
    * time_point_stable: the timestamps of streams when catching stabilizes (i.e., the hands and the object keeps relatively stable).
      * same data format as above "time_point_touch"
<!-- </details> -->
<br>

## Supporting Files
<!-- <details><summary>Exlanation about subjects/objects.csv and log.xlsx </summary> -->
* **subjects.csv**: the list of the subjects participating in the experiments
  * subject ID
* **objects.csv**: the list of used objects
  * object name
  * characteristic: `rigid`, `soft` or `printed`
  * attached with optitrack markers: 
    * 1: yes
    * 0: no
* **log.xlsx**: logbook with the recording parameters of all takes. 
  * **{subject ID} sheet**: each sheet maintains all instructions received by a subject during recording and is named by the id of the subject. Each entry in the sheet describes one recording setting.
    * no: the index of the entry in the spreadsheet 
    * object: the name of the object
    * equipped: if the subject is equipped with the helmet and the gloves to record data or not
      * 1: equipped
      * 0: not equipped
    * action: the action that the subject is supposed to perform when recording
      * `throw` or `catch`
    * hand: the instruction for using either single or both hands to perform the action
      * `single`, `both` or `void` (no constraint)
    * position: the initial location where the subject shall stand at the start of each recording
      * x: in range [0, 1, 2, 3]
      * y: in range [0, 1, 2, 3]
    * height: the relative vertical location for the subject hand
      * `overhead`, `overhand`, `chest`, `underhand` or `void` (no constraint)
    * horizon: the relative horizontal location of the subject hand
      * `left`, `middle`, `right` or `void` (no constraint)
    * speed: the relative velocity at which the object is supposed to be tossed out 
      * `fast`, `normal`, `slow` or `void` (no constraint)
    * take_id: the id of the take
    * success: the result of catching. 
      * 1: success
      * 0: failed
    * verified: if the recording has been verified. 
      * 1: verified and no problem detected
      * 0: verification not finished
      * -1: problem detected
    * annotated: if the annotation has been finished. 
      * 1: finished
      * 0: not finished

<!-- </details> -->


## Reference

A quick reference to the terms used above.

<!-- ## Terms We Created -->

* subject 1 and 2
  * subject 1: the subject who is equipped with the helmet and the gloves. The left person in Figure 4 of [the paper]().
  * subject 2: the subject who is equipped with only the head band. The right person in Figure 4 of [the paper]().
* OptiTrack object ID 
  * 115: the helmet
  * 116: the right hand tracking plate
  * 117: the left hand tracking plate
  * 118: the headband
  * 101: airplane
  * 102: round_plate
  * 103: apple
  * 104: banana
  * 105: hammer
  * 106: long_neck_bottle
  * 107: wristwatch
  * 108: bowl
  * 109: block
  * 110: cylinder
  * 111: cube
  * 112: torus
  * 113: wrench
  * 114: leopard
  * 119: toothbrus
* ZED device SN
  * 17471: the fixed third-person (side) view ZED RGBD sensor
  * 24483054: the dynamic egocentric view ZED RGBD sensor
  * 28280967:the fixed third-person (back) view ZED RGBD sensor 
* Stream ID
  * rgbd0: the fixed third person (side) view ZED RGBD sensor
  * rgbd1: the dynamic egocentric view ZED RGBD sensor
  * rgbd2: the fixed third-person (back) view ZED RGBD sensor 
  * event: event camera
  * left_hand_pose: left hand pose recorded by stretchsense gloves
  * right_hand_pose: right hand pose recorded by stretchsense gloves
  * sub1_head_motion: the motion of the subject 1's head (helmet) recorded by OptiTrack
  * sub1_left_hand_motion: the motion of the subject 1's left hand (left hand glove) recorded by OptiTrack
  * sub1_right_hand_motion: the motion of the subject 1's right hand (right hand glove) recorded by OptiTrack
  * sub2_head_motion: the motion of the subject 2's head (headband) recorded by OptiTrack

<!-- 5. pose coordinates: there are three coordinates involved for positioning 

   * global OptiTrack coordinates: the coordinates specified and used by the OptiTrack system
   * local coordinates: see figure xxx in The paper, and the origin is defined at the bottom right of the throw-catch zone
   * local rough coordinates: the simplified local coordinates only used to instruct the initial positions of subjects in data collection. -->
