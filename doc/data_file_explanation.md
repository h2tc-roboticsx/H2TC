<!-- # Files Explanation  -->
# Data Formats
This document explains the data hierarchy and the content of each data file stored in our dataset [H<sup>2</sup>TC](https://h2tc-roboticsx.github.io/). <br>

Here is a quick overview of the files involved in the dataset.  To distinguish, hereafter, the **file** presents **bold**, and the ***folder*** presents **bold** and *italic*. For those specific digits, such as subject ids or device ids, the [reference](#reference) explains their meanings.  <br>
<!-- ## Overview -->
* [***data/***](#data)
  * ***{take ID}/***: The take folder named by the take id, e.g. 000000
    * ***raw/***: The raw data directly exported from multiple recording sensors, e.g. ZED cameras
    * ***processed/***: The formatted data derived from the raw data
* [***annotations/***](#annotations)
    * **{take ID}.json**: The annotation result for the take id, e.g. 000000
* [***objects/***](#objects): The scanned object models
    * **object_name.stl**: The scanned object model, e.g. `apple.stl`
* [**log.xlsx**](#supporting-files): The logbook with the recording and annotating parameters of all takes
* [**subjects.csv**](#supporting-files): The list of the subjects participating in the dataset
* [**objects.csv**](#supporting-files): The list of used objects in the dataset


## Data
<!-- <details><summary>Explanation about all files in data folder</summary> -->
Before diving into the data details below, we suggest users check the [data processing tutorial](https://github.com/h2tc-roboticsx/H2TC/tree/main/#data-processing) first to see how we process the dataset.
Note that the raw data files and their contents that are **not** used in the data processing are <u>underlined</u>. 
* ***{take ID}/***: The take folder named by the take id, e.g. 000000.
  * ***raw/***: The raw data directly exported from multiple recording sensors, e.g. ZED cameras and OptiTrack. 
    * ***hand/***: The hand joint pose data recorded by [Hand Engine](https://stretchsense.com/solution/hand-engine/) (i.e. the software driver of the Prophesee event camera). Note the details on the coordinate frames are introduced [here](https://github.com/h2tc-roboticsx/H2TC/tree/main/docProphesee/processing_techdetails.md#•-hand-pose-data-coordinate).
      * **P1L.csv** / **P1R.csv**: The sensor readings and hand joint pose of the left (L) and right (R) hand respectively.   Specifically, each file contains by column
        * <u>Capacitance 0-15</u>: The capacitance reading in picofarad (pF) of the wrist sensor and 15 splay sensors.
        * <u>IMU 1-10</u>: N/A (the glove version we used does not contain IMU sensors).
        * Timecode (device): The timecode from the clock inside the gloves ticks as data starts generating.
        * <u>Timer (device)</u>: The glove's internal timer that ticks every 1/120 s.
        * <u>Timecode (master)</u>: The timecode from the clock of the host machine where [Hand Engine](https://stretchsense.com/solution/hand-engine/) runs.
        * hand: The wrist joint pose in Euler-angle format (XYZ degrees). 
        * index 00-03, middle 00-03, pinky 00-03, ring 00-03, thumb 01-03: The hand joint pose in Euler-angle format (XYZ degrees). Each joint rotates w.r.t. its parent link in the kinematic chain. Please refer to our [paper](toadd) and the [data processing tutorial](https://github.com/h2tc-roboticsx/H2TC/tree/main/#data-processing) for details of our hand model.  
      * **P1LMeta.json** / **P1RMeta.json**: The metadata about the gloves and recording setting. 
      * <u>**P1L.cal** / **P1R.cal**</u>: The calibration data. 
      * <u>**P1L.fbx** / **P1R.fbx**</u>: The hand model saved in the` .fbx` format.  
    * **17471.svo / 24483054.svo / 28280967.svo**: Each corresponds to the raw file of a ZED  camera which packs the left- and right-eye videos and the metadata. Check [ZED document](https://www.stereolabs.com/docs/video/recording/) for more information. The digits denote the serial number (SN) of the ZED cameras. Specifically, 17471 is the fixed third-person (side) camera, 24483054 is the dynamic egocentric camera, and 28280967 is another fixed third-person (back) camera.
    * **17471.csv / 24483054.csv / 28280967.csv**: The timestamps of the captured RGB-D  frames. The digits, similarly as mentioned above, refer to the SN of the ZED cameras. Note that the timestamps start from the record start, so the second timestamp is the actual timestamp of the first frame. Therefore, the first timestamp in this file is always discarded during processing or aligning the streams. Specifically, each file contains by row 
      * nanoseconds: The header of the unit
      * the timestamp of the record start
      * the timestamp of the 1st frame 
      * the timestamp of the 2nd frame
      * ... 
      * the timestamp of the N-th frame 
    *  **event_{timestamp}.raw**: The raw output of the event camera. It includes general metadata and event information encoded in [EVT3.0](https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html?highlight=data%20format) format. Refer to [Prophesee documentation](https://docs.prophesee.ai/stable/index.html) for more details. The digits `{timestamp}` in the file name log the timestamp of the record start, which is used later in the processing pipeline to align the event stream with other streams and to decode the timestamp streams. 
    * <u>**event.bias**</u>: The event sensor settings. Check [event bias](https://docs.prophesee.ai/stable/hw/manuals/biases.html) for more detailed information.
    * **optitrack.csv**: The tracking data recorded by [OptiTrack](https://optitrack.com/). Specifically, the file contains by column
      * object id: The id of the tracked object. Check the [reference](#reference) for the id and name of each tracked object by OptiTrack. 
      * <u>frame</u>: The frame index maintained by the OptiTrack server.
      * <u>mean error</u>: The mean error of localization.
      * timestamp: The frame timestamp that is calculated by subtracting the latency. 
      * <u>latency</u>: The time from the exposure of the cameras to the reception of data. 
      <!-- on the client.  -->
      * <u>local transformation matrix</u>: The local pose transformation matrix w.r.t. the start pose frame. The matrix has a size of 4x4 and is unfolded by row.
      * global transformation matrix: The global pose w.r.t. the OptiTrack world frame. The matrix has a size of 4x4 and is unfolded by row. 
    <br>

  * ***processed/***: The formatted data derived from the raw data. 
    * ***rgbd0/***: The left-eye RGB images and normalized depth images from the fixed third-person (side)  ZED  camera (SN: 17471).
      * **left_{frame_ID}.png**: The RGB images produced by the left-eye camera of the ZED. Left-eye RGB images align with the corresponding depth maps, which are set internally by [ZED SDK](https://www.stereolabs.com/docs/video/recording/). The digits `{frame_ID}` represent the frame index. 
      * **depth_{frame_ID}.png**: The normalized depth images. They align with the corresponding left-view RGB images. The digits `{frame_ID}` represent the frame index. They should be only used for visualization since they do not contain real depth values. 
      * **depth.npy**: The 3-dimensional numpy array holding the unnormalized depth estimation of each frame. It aligns with the left-eye RGB images by [ZED SDK](https://www.stereolabs.com/docs/video/recording/). The unit of depth is millimeter. 
        * dimension 0: The frame index
        * dimension 1: The height in range [0, 719] where 0 corresponds to the topmost of the image.
        * dimension 2: The width in range [0, 1279] where 0 corresponds to the leftmost of the image.
    * ***rgbd1/***: Similar data as *rgbd0/*  but from the dynamic egocentric ZED  camera (SN: 24483054).
    * ***rgbd2/***: Similar data as *rgbd0/*  but from the fixed third-person (back) ZED  camera (SN: 28280967). 
    * **rgbd0_ts.csv**: The timestamps of `rgbd0` frames. This is the same as the timestamp file `/raw/17471.csv` but is renamed to be more interpretable.
    * **rgbd1_ts.csv**: Similar data as `rgbd0_ts.csv` but for `rgbd1` frames.
    * **rgbd2_ts.csv**: Similar data as `rgbd0_ts.csv` but for `rgbd2` frames.
    * **event_xypt.csv**: The decoded [Contrast Detector (CD)](https://docs.prophesee.ai/stable/concepts.html#event-generation) events.
      * x: The width in range [0, 1279], where 0 corresponds to the leftmost of the image.
      * y: The height in range [0, 719], where 0 corresponds to the topmost of the image.
      * p: The polarity
        * 0: The corresponding CD event is off, i.e. a negative contrast is detected that light changes from lighter to darker.
        * 1: The corresponding CD event is on, i.e. a positive contrast is detected that light changes from darker to lighter.
      * t: The timestamp of the light change in the UNIX nanosecond.
    * ***event/***: The frame-based visualization of events.
      * **{frame_ID}.jpg**: A frame visualization of events accumulated in a fixed period (1/60 second for 60 FPS). For each pixel, it is initialized with the background color (dark blue). If the polarity of the event located at the pixel occurs positive (negative) over the period, the pixel changes to white (light blue). Check [Event Generation](https://docs.prophesee.ai/stable/concepts.html#event-generation) for more details. The digits `{frame_ID}` represent the frame index. 
    * **event_frames_ts.csv**: The timestamps of event frames in *event/*.
    * **left_hand_pose.csv**: The joint motion data of the left hand produced based on `/raw/hand/P1L.csv`.
      * timestamp: The timestamp in the UNIX format derived from the timecode (device) in the raw data file `/raw/hand/P1L.csv`.
      * hand, index 00-03, middle 00-03, pinky 00-03, ring 00-03 and thumb 01-03: The hand joint motions that are the same as what they are in raw data.
    * **right_hand_pose.csv**: Similar joint pose data as `left_hand_pose.csv` but for the right hand produced based on `/raw/hand/P1R.csv`.
    * **sub1_head_motion.csv**: The position and orientation of the helmet worn by the primary subject (subject1) corresponding to the OptiTrack object 115 in `/raw/optitrack.csv`.
      * timestamp: The same timestamp as in the raw data file `/raw/optitrack.csv`.
      * x, y, z: The position of the object in the [throw&catch coordinate frame](https://github.com/h2tc-roboticsx/H2TC/tree/main/doc/processing_techdetails.md#the-coordinate-setting). Note that the coordinate frame is Y-up, and its origin is at the bottom left. 
      * qx, qy, qz, qw: The orientation of the object in quaternion. 
    * **sub1_left_hand_motion.csv**: The primary subject's left hand motion. The object id in OptiTrack is 117 in`/raw/optitrack.csv`. Similar data structure as in `sub1_head_motion.csv`. 
    * **sub1_right_hand_motion.csv**: The primary subject's right hand motion. The object id in OptiTrack is 116 in`/raw/optitrack.csv`. Similar data structure as in `sub1_head_motion.csv`. 
    * **sub2_head_motion.csv**: The auxiliary subject's head motion. The object id in OptiTrack is 118 in `/raw/optitrack.csv`. Similar data structure as in `sub1_head_motion.csv`. 
    * **object.csv**: The object motion.  Similar data structure as in `sub1_head_motion.csv`. 
    * **alignment.json**:  A dictionary whose keys represent the frame indices of rgbd0. Inside each frame (key), there is another dictionary that includes the multi-modal data streams and their corresponding timestamps. These streams are aligned by finding the closest timestamp to the reference timestamp of `rgbd0` within a threshold.
      * frame index: Starting from 0
        * key: The stream id
        * value: Timestamp
        
<br>

## Annotations
<!-- <details><summary>Exlanation about annotation labels </summary> -->
Before reading the file details below, please check our [annotation tutorial](https://github.com/h2tc-roboticsx/H2T/tree/main/#annotator) and [paper](todo) first to capture how we annotate the dataset.

* **{take ID}.json**: metadata and the annotation data of the take. 
  * status: Annotation status of the take.
    * 1: The annotation is finished
    * 0: Not finished
    * -1: The annotation is problematic and needs further check
  * take_id: The index of the take, e.g. 000000 for the first take
  * object: The name of the object thrown/caught in this take, e.g. 'basketball'
  * catch_result: The result of catching.
    * 1: Success
    * 0: Fail
  * sub1_cmd: The [instruction](https://github.com/h2tc-roboticsx/H2TC/tree/main/#annotator) given to the primary subject (subject 1) to instruct his/her behavior of throwing or catching. See our [paper](toadd) for more details. Please refer to [log.xlsx](#supporting-files) for the explanation of the attributes below. Note the actual human behaviors are also annotated below.
    * subject_id: The id of the primary subject
    * hand: The grasp mode, `single` or `both`
    * position: The 2-dimensional discrete initial standing location (x, z)
    * action: `throw` or `catch`
  * sub2_cmd: The instruction given to the auxiliary subject (subject 2). See our [paper](toadd) for more details. Note the actual human behaviors are also annotated below.
    * subject_id: The  id of the auxiliary subject
    * hand: The grasp mode, `single` or `both`
    * position:  The 2-dimensional discrete initial standing location (x, z)
    * action: `throw` or `catch`
    * hand_vertical: The relative vertical position of the hand(s) to  throw or catch an object
      * value: `overhead`, `overhand`, `chest` or `underhand`
    * hand_horizontal: The relative horizontal position of the hand(s) to  catch an object
      * value: `right`, `middle` or `left` 
      * This attribute is only given when the auxiliary subject  is supposed to catch
    * throwing_speed: The relative speed of throwing away the object 
      * value: `fast`, `normal` or `slow`
      * This attribute is only given when the  auxiliary subject  is supposed to throw
  * throw: The annotation data for the moment `throw`.
    * hand: The actual grasp mode for throwing the object
      * value: `left`, `right` or `both`
      * This is different from "hand" in the subjects' instruction above in two ways: 1) It details the `single` with `left` and `right`; 2) It represents the thrower's actual grasp mode in the recorded activity, which may be different from the instruction. 
    * hand_vertical_thrower: The relative vertical position of the hand(s) of the thrower at the moment `throw`
      * value: `overhead`, `overhand`, `chest` or `underhand`
    * hand_vertical_catcher: Similar to the above "hand_vertical_thrower" but for the catcher
    * hand_horizontal_thrower: The relative horizontal position of the hand(s) of the thrower at the moment `throw`
      * value: `right`, `middle` or `left` 
    * hand_horizontal_catcher: Similar to the above "hand_horizontal_thrower" but for the catcher
    * position_thrower: The actual exact location of the thrower at the moment `throw`
      * x, z: 2-dimensional position
      <!-- * this is different from the `position` in the previous subject command like "sub2_cmd" -->
    * position_catcher: The actual exact location of the catcher at the moment `throw`.
      * x, z: 2-dimensional position
    * object_flying_speed: The average flying speed of the thrown object (during free flying) in a unit of m/s.
    * time_point: The timestamps and frame indexes of all streams at the moment `throw`
      * stream id
        * frame: The frame index of the corresponding stream
        * timestamp: The UNIX timestamp in nanoseconds of the corresponding stream 
  * catch: The annotation data for the moment `catch`
    * hand: The grasp mode for catching the object
      * value: `left`, `right` or `both`
    * hand_vertical: The relative vertical position(s) of the hand(s) of the catcher at the moment `catch_stable`, i.e. when the hand(s) and the object keep relatively stable
    * hand_horizontal: The relative horizontal position(s) of the hand(s) of the catcher at the moment `catch_stable`
    * position: The exact location of the catcher at the moment `catch_touch`, i.e.  when the hand(s) of the catcher first touch(es) the object in  flight
    * time_point_touch: The timestamps and frame indexes of all streams at the moment `catch_touch`
      * stream id
        * frame: The frame index of the corresponding stream
        * timestamp: The UNIX timestamp in nanoseconds of the corresponding stream 
    * time_point_stable: The timestamps of streams  at the moment `catch_stable`
      * Same data format as the above "time_point_touch"
<!-- </details> -->
<br>

## Supporting Files
<!-- <details><summary>Exlanation about subjects/objects.csv and log.xlsx </summary> -->
* **subjects.csv**: The list of the subjects participating in the dataset
  * subject id
* **objects.csv**: The list of used objects in the dataset
  * object name
  * characteristic: `rigid`, `soft` or `printed`
  * attached with OptiTrack markers: 
    * 1: yes
    * 0: no
* **log.xlsx**: The logbook with the recording parameters of all takes. 
  * **{subject ID} sheet**: Each sheet maintains all instructions received by a subject during recording and is named by the id of the subject. Each entry in the sheet describes one recording setting.
    * no: The index of the entry in the spreadsheet 
    * object: The name of the throw&catch object
    * equipped: The subject is the primary subject, i.e. the subject is equipped with the helmet and the motion capture gloves to record data
      * 1: Equipped
      * 0: Not equipped
    * action: The action that the subject is supposed to perform during recording
      * `throw` or `catch`
    * hand: The instruction to use either single or both hands to perform the action
      * `single`, `both` or `void` (no constraint)
    * position: The discrete initial standing location where the subject shall stand at the start of each recording
      * x: in range [0, 1, 2, 3]
      * y: in range [0, 1, 2, 3]
    * height: The relative vertical location(s) of the subject hand(s)
      * `overhead`, `overhand`, `chest`, `underhand` or `void` (no constraint)
    * horizon: The relative horizontal location(s) of the subject hand(s)
      * `left`, `middle`, `right` or `void` (no constraint)
    * speed: The relative speed at which the object is supposed to be tossed out 
      * `fast`, `normal`, `slow` or `void` (no constraint)
    * take_id: The id of the take
    * success: The catching result of the recorded activity. 
      * 1: Success
      * 0: Fail
    * verified: The recording has been verified. 
      * 1: Verified and no problem detected
      * 0: Not verified yet
      * -1: Problematic
    * annotated: The annotation has been finished. 
      * 1: Finished
      * 0: Not finished yet

<!-- </details> -->


## Reference

A quick reference to the terms used above.

<!-- ## Terms We Created -->

* Subject 1 and 2
  * subject 1: The primary subject who is equipped with a helmet and gloves. 
  * subject 2: The auxiliary subject who is equipped with only the headband. 


* OptiTrack object id 
  * 115: helmet
  * 116: the marker set attached to the right hand
  * 117: the marker set  attached to the left hand
  * 118: headband
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
  * 119: toothbrush


* ZED device SN
  * 17471: The fixed third-person (side)  ZED 
  * 24483054: The dynamic egocentric  ZED 
  * 28280967: The fixed third-person (back)  ZED  


* Stream ID
  * rgbd0: The fixed third-person (side)  ZED 
  * rgbd1: The dynamic egocentric  ZED 
  * rgbd2: The fixed third-person (back)  ZED
  * event: The event camera
  * left_hand_pose: The left hand pose recorded by Stretchsense gloves
  * right_hand_pose: The right hand pose recorded by Stretchsense gloves
  * sub1_head_motion: The motion of the primary subject's head (helmet) recorded by OptiTrack
  * sub1_left_hand_motion: The motion of the primary subject's left hand recorded by OptiTrack
  * sub1_right_hand_motion: The motion of the primary subject's right hand recorded by OptiTrack
  * sub2_head_motion: The motion of the auxiliary subject 2's head (headband) recorded by OptiTrack

