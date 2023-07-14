# Data Processing Technical Details
[Our dataset H<sup>2</sup>TC](https://lipengroboticsx.github.io/) contains multi-model cross-device raw data streams. 
To make the dataset easier to use, we have developed the [processor source code](https://github.com/lipengroboticsx/H2TC_code/tree/main/src) and provided the [data processing document](https://github.com/lipengroboticsx/H2TC_code/tree/main#data-processing) to help readers get aligned and common-format data. 
Considering readers may want to design their customized processing, we introduce the processing technical details in this document to auxiliarly explain the [source code](https://github.com/lipengroboticsx/H2TC_code/tree/main/src). 

Here is an overview of this document:

* [**Our Workspace**](#our-workspace): introduces our used devices and our coordinate setting. 
* [**OptiTrack Data Processing**](#optitrack-data-processing): explains the original optitrack coordinate and how to tranfer it to our coordinate. 
* [**Hand Pose Data Processing**](#hand-pose-data-processing): explains how to extract hand poses and how to visualize them. 
* [**Timestamp Alignment**](#timestamp-alignment): explains how to do time alignment. 

## Our Workspace
### Used Devices
We use a variety of specialized motion tracking and visual streaming devices to capture our dataset as illustrated below. The details about our recording framework are introduced in [Sec. 4 Recording Framework in our paper](). 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/hardware.png" width = "800" alt="hardware" />

| Device | Manufacturer | Recording Content |FPS |Resolution  |
|:-----|:-----:|:-----:|:-----:|:-----:|
| ① Gloves | [StretchSense MoCap Pro](https://stretchsense.com/) | Hand Pose | 120 | - |
| ②⑤ Tracker | [OptiTrack](https://optitrack.com/) |  Human Motion | 240 | - |
| ③ Event Camera | [Prophesee](https://www.prophesee.ai/) | Event | - | 1280x720 |
| ④ ZED Camera | [Stereolabs](https://www.stereolabs.com/zed-2/) |  RGB-D | - | 1280x720 |


### Our Coordinate Setting

As shown below, the **origin** of our throw-catch zone refers to the bottom-left corner of the entire throw-catch zone. The coordinate **axes** are set up as follows: XZ plane is parallel to the ground with Z-axis along the *5 m* (*2m + 1m + 2m*) side and X-axis along the *2 m* side. Y-axis is perpendicular up to the XZ plane. 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/workspace.png" width = "400" alt="workspace" />

<!-- ### Note 
* Headband and helmet coordinate system. 

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/D6403D046FEE97803D912C8DB100C11F.png" width = "200">

The above figure shows our defined coordinates of the headband and the helmet. The coordinate of the helmet is the same as that of our throw-catch zone coordinate. While the coordinate of the headband has a rotate of along the Y-axis counterclockwise with 180 degrees.

In a real scenario, when the primary subject is wearing the helmet and the auxiliary subject is wearing the headband, the coordinate systems will look like as in the following picture:

<!-- ![helmet_handband_coordinate_real_scenario.png](https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/A4A102E4F59E1D43D95D22ABB44AD561.png ) 
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/A4A102E4F59E1D43D95D22ABB44AD561.png" width = "200"> -->


<!-- ## ZED RGBD streams -->

<!-- ## Prophesee Event -->
<br>

## OptiTrack Data Processing
### The Coordinate System ID
As our data collection spans more than three months, during which, our throw-catch zone in the lab was moved twice. Therefore, there are **THREE different coordinates** in our recordings. We label them as: 

|  Take   | Coordinate System ID| 
|  :----:  | :----:  | 
| 0-2888  | \#0 |            
| 2889-9788  | \#1 |         
| 9789-12905 |\#2   |         


To transfer these different optitrack coordinates to [our coordinate](#our-coordinate-setting), we apply coordinate transformation via the 4 x 4 transformation matrices captured in the original optitrack system (shown in the script [optitrack.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/optitrack.py)). The specific transformation matrices are shown in [here](#system-id-and-transformation-matrix). 
<!-- 
```
object_tc_transformation_matrix = np.matmul(origin_transformation_matrix, object_optitrack_raw_transformation_matrix)
``` -->

### Note
* Additional transformation. Due to the capturing limitation, we need additional transformation to correct some parts of tracking data in some takes (addressed in the script [optitrack.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/utils/optitrack.py)). They are:

|  Take   | Part | Rotation | 
|  :----:  | :----:  |:----:  |
| 520-1559  | right hand |  90 degrees |          
| 1560-1699  | right hand |  180 degrees |        
| 1040-1559 |  left hand |    45 degrees |
|  0-1699 |  helmet|   45 degrees  |
|  0-1699 | headband |   -180 degrees|

* Difference between local and global transformation matrices in `optitrack.csv`. 
The raw `optitrack.csv` file contains `local` (from column 5 to 20) and `global` (from column 21 to 36) transformation matrices of the optitrack system. The `local` transformation matrix is expressed relative to the start pose of a recorded sequence. The `global` transformation matrix is with reference to Optitrack world coordinate (Y-Up). In our codebase, we only used the `global` transformation matrix for matrix manipulation. 

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

## Hand Pose Data Processing
Our hand pose data is generated by [StretchSense MoCap Pro Gloves Hand Engine](https://stretchsense.com/solution/hand-engine/). 
You can check [data_file_explanation.md](https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/data_file_explanation.md/#data) to get each term meaning in the raw data folder `hand/`.  

### Reconstruct the left hand

**Left hand uses a right-handed coordinate system** 

### Right-handed coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/97CE3BB762B65208FED74A7D8A0D4C12.png" width = "400" alt="left_hand">


The above figure shows the right-handed coordinate system of the left hand. Each joint has its own XYZ frame. The X-axis is along the bone, the Y-axis is perpendicular to the palm, and the Z-axis is perpendicular to the XY plane. The enlarged frame at the bottom is put there for clarification and easier understanding.

### Our throw-catch zone coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/C650A1275361BA54AB728D0C141801F6.png" width = "400" alt="opti_lefthand">

The above figure shows the coordinate system of the captured hand motion in the our throw-catch zone. The frame is put there for clarification and easier understanding. In practice, the origin of the frame is around the center of the back of the hand. For this coordinate system, Y-axis is perpendicular up to the back of the  hand, Z-axis is parallel to the finger tip direction, and X-axis is perpendicular to the YZ plane.

### Align right-handed coordinate system with our throw-catch zone coordinate system

As the orientation of the right-handed coordinate system differs from that of the our throw-catch zone coordinate system, we use two rotation matrices to convert the orientation of the hand coordinate system to the same as the our throw-catch zone.

Specifically, we first rotate the hand coordinate system -180 degrees along the X-axis, and then rotate it -90 degrees along the Y-axis. 


### Reconstruction
As mentioned in Optitrack\_Readme [TODO link], for each hand pose, we use the translation, i.e., x, y, z positions in its associated 4 x 4 transformation matrix that has been converted to the our throw-catch zone coordinate system as the metacarpal joint. We then reconstruct the entire hand pose starting from the metacarpal joint with the captured hand joint angles (XYZ euler angles) and the defined hand bone length (see Bone length section in this readme) using **Forward Kinematics**:

Specifically, the XYZ spatial position `P` of a left hand finger joint **(except the metacarpal joint)** in the our throw-catch zone can be calculated using the following equations: (TODO)




## Reconstruct the right hand

**Right hand uses a left-handed coordinate system**

### Left-handed coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/B08B2DCADADCFD3D2B101CC1AFFBA015.png" width = "400" alt="right_hand">

The above figures shows the left-handed coordinate system of the right hand. Each joint has its own XYZ frame. The X-axis is along the bone, the Y-axis is perpendicular up towards the back of the hand, and the Z-axis is perpendicular to the XY plane. The enlarged frame at the bottom is put there for clarification and easier understanding.

## our throw-catch zone coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/018D81940FEC63AD318DBD8B5AF0FF98.png" width = "400" alt="opti_righthand">

Similar as mentioned above, the above figure shows the coordinate system of the captured hand motion in the our throw-catch zone. The frame is put there for clarification and easier understanding. In practice, the origin of the frame is around the center of the back of the hand. For this coordinate system, Y-axis is perpendicular up to the back of the  hand, Z-axis is parallel to the finger tip direction, and X-axis is perpendicular to the YZ plane.


### Coordinate system conversion
To reconstruct the right hand, we first convert the left-handed coordinate system to the right-handed one using a matrix `t_h` (is this description corret??? also is the comment in line 126 above t\_h correct in plot\_motion.py??? what does t\_h exactly do? convert coordinate system or convert data??)

```
t_h = [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
```
We also apply a rotation matrix rotY as shown in the `plot_right_hand` function in `plot_motion.py` to rotate the hand coordinate system -90 degrees along the Y-axis

```
rotY = [[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]
```

### Reconstruction
Similar to the reconstruction of the left hand, we use the translation of the converted 4 x 4 transformation matrix that is associated with the hand pose as the metacarpal joint, and then reconstruct the entire hand pose starting from the metacarpal joint with the captured hand joint angles (XYZ euler angles) and the defined hand bone length (see Bone length section in this readme) using **Forward Kinematics**:

Specifically, the XYZ spatial position `P` of a right hand finger joint **(except the metacarpal joint)** in the our throw-catch zone can be calculated using the following equations: (TODO)



### Metacalpal joint offset

Note that, as a common practice, we did not attach the markers directly to the hand, but fixed markers on a rigid object, and then attached the rigid object to the back of the hand (See figure below). As the geometric center of the rigid object does not exactly align with the metacarpal joint of a hand, there is an offset between the reconstructed hand and the actual hand in terms of their spatial positions in the our throw-catch zone. However, this offset is minor, and does not change the motion of the hand.

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/045A78822B66A604FE54BEC901DEC56E.png" width = "400" alt="glove_with_markers">

### Hand size (<u>BONE LENGTH HAS TO BE DETERMINED</u>)

Note that the finger bone length we used for visualization purpose in `plot_motion.py` is enlarged. Also, it is not necessary to measure every single subject's finger bone length. **Therefore, we use the average bone length to reconstruct the hand as a common practice**. 


### Bone length (<u>BONE LENGTH HAS TO BE DETERMINED</u>)

#### Visualization
We use the following set of bone lengths to reconstruct and visualize the hands in `plot_motion.py`:
```
         Metacarpal Proximal Middle Distal
thumb =  [0.25,     0.11,           0.06]
index =  [0.34,     0.15,    0.08,  0.06]
middle = [0.33,     0.15,    0.10,  0.07]
ring =   [0.31,     0.13,    0.10,  0.06]
pinky =  [0.3,      0.08,    0.06,  0.06]
```

#### Paper
The bone lengths in the paper are measured from an actual hand (TODO, better to provide an average bone length model)
```
         Metacarpal Proximal Middle Distal
thumb =  [6.0,      4.0,            3.5]
index =  [8.0,      5.5,     3.0,   2.5]
middle = [8.0,      6.0,     3.5,   2.7]
ring =   [7.5,      5.5,     3.3,   2.5]
pinky =  [6.5,      4.5,     2.5,   2.5]
```

## Timestamp Alignment

Although all data streams have been timestamped during data collection, it is impossible for their timestamps to be exactly the same. There exists time drift in millisecond-level between data streams. Therefore, during processing, we use the timestamp of **rgbd0 camera**, the fixed third-person (side) view camera, serial number: 17471, as the reference, and align the timestamps of the rest data streams to it. The resulting timestamp alignment is saved in a file called `alignment.json`. 

### The alignment.json file
`Alignment.json` file essentially saves a dictionary whose keys represent the frame indices. The corresponding value for each key is a mapping between stream id and its timestamp. In each frame (key), we align each stream timestamp to the frame reference timestamp (`rgbd0`) by finding the closest one to the reference.
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
If the timestamp of a data stream is missing in certain frames, its value will be `null` as shown in the above example. Such situation is rare, and it is mainly caused by 1) the Optitrack when the tracked object is occluded; or 2) by StretchSense gloves when the data transmission is congested; or 3) the open of the event camera lags slightly behind the rgbd0 camera. Therefore, the corresponding timestamp is missing.

### How to create an alignment.json file
We use the timestamps of **rgbd0 camera** as the reference. Therefore, the total number of frames saved in the `alignment.json` is equal to the number of timestamps recorded by **rgbd0 camera**. 

Given a timestamp of **rgbd0 camera** and its associated frame number, for each of other data streams, we use the binary search alogrithm to find their nearest timestamp to the timestamp of **rgbd0 camera**. This nearest timestamp is then used as the timestamp of that frame of the other data stream. Note that the difference between the nearest timestamp and its query **rgbd0 camera**'s timestamp has to be within a threshold, which is currently set to 1/60 * 10e9 nanosecond.


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

