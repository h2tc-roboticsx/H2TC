# OptiTrack 

## The origin of the throw-catch zone

The origin of our throw-catch zone refers to the bottom-left corner of the entire throw-catch zone as shown in the following figure:
![throw-catch-zone.png](resources/72E47F163C2F1E8393908550C69B2A3C.png =912x247)

The frame of the origin is set up as follows: XZ plane is parallel to the ground with Z-axis along the 5 m side and X-axis along the 2 m side. Y-axis is perpendicular up to the XZ plane.

As our data collection spans more than three months, during which, our throw-catch zone in the laboratory was moved twice. In total, there are 3 different transformation matrices collected for the origin after each movement of the throw-catch zone.

Specifically, takes 0-2888 use origin \#0; takes 2889-9788 use origin \#1; and takes 9789-12905 use origin \#2.

The 4 x 4 transformation matrix reference to Optitrack World frame of origin \#0 is:
```
[[-0.99886939, -0.04535922, -0.01408667, 0.42632084],
 [-0.04514784, 0.99886579, -0.0149642, 0.0984003 ],
 [ 0.01474855, -0.01431195, -0.99978858, 7.67951849],
 [ 0., 0., 0., 1. ]]
```

The 4 x 4 transformation matrix reference to Optitrack World frame of origin \#1 is:
```
[[-9.99963351e-01, 8.30436476e-03, 2.13574045e-03, 1.92400245e-01],
 [ 8.31340413e-03, 9.99956270e-01, 4.25134508e-03, 6.55417571e-02],
 [-2.10037766e-03, 4.26893351e-03, -9.99988700e-01, 2.17126483e+00],
 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
```
The 4 x 4 transformation matrix reference to Optitrack World frame of origin \#2 is:
```
[[-0.99997146, 0.00456379, 0.00601402, 0.19729361],
 [ 0.00454136, 0.99998255, -0.00373799, 0.06776005],
 [-0.00603099, -0.00371067, -0.99997492, 2.48060394],
 [ 0., 0., 0., 1. ]]
```

To convert the transformation matrix from Optitrack World frame to our local throw-catch zone, we use the inverse matrix of the above 4 x 4 transformation matrix:

The inverse matrix of origin \#0 is:
```
[[-0.99887346, -0.04514823,  0.01474953,  0.31701391],
 [-0.04535921,  0.99887065, -0.01431137,  0.0309528 ],
 [-0.01408573, -0.01496482, -0.99978902,  7.68537583],
 [ 0.        ,  0.        ,  0.        ,  1.        ]]
```

The inverse matrix of origin \#1 is:
```
[[-9.99963124e-01,  8.31338829e-03, -2.10034234e-03, 1.96408675e-01],
 [ 8.30438079e-03,  9.99956542e-01,  4.26894456e-03, -7.64056829e-02],
 [ 2.13577519e-03,  4.25133477e-03, -9.99988665e-01, 2.17055065e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
```

The inverse matrix of origin \#2 is:
```
[[-0.99997154,  0.00454136, -0.00603098,  0.21194073],
 [ 0.00456379,  0.99998285, -0.00371057, -0.05945483],
 [ 0.00601403, -0.00373809, -0.99997494,  2.47960853],
 [ 0.        ,  0.        ,  0.        ,  1.        ]]
```



The following code converts the 4 x 4 transformation matrix reference to the Optitrack World system to the local throw-catch zone system, where `origin_transformation_matrix_inverse` is one of the three inverse matrices above, `object_optitrack_raw_transformation_matrix` is the tracked object's 4 x 4 transformation matrix expressed in the Optitrack World system, and `object_tc_transformation_matrix` is the converted 4 x 4 transformation matrix expressed in the local throw-catch zone system.

```
object_tc_transformation_matrix = np.matmul(origin_transformation_matrix_inverse, object_optitrack_raw_transformation_matrix)
```

## Extra rotation

Since takes 1700 onwards, the orientations of both hands, helmet, and headband were checked everytime before the start of recording. Therefore, no extra rotation is needed for takes from 1700 onwards.

For takes from 520-1559, right hand needs to rotate 90 degrees (addressed in the script optitrack.py)
For takes from 1560-1699, right hand needs to rotate 180 degrees (addressed in the script optitrack.py)
For takes from 1040-1559, left hand needs to rotate 45 degrees (addressed in the script optitrack.py)
For takes from 0-1699, the orientation of the helmet and headband needs to be corrected if using their orientation (rotate along Y axis with extra 45 degrees for the headband, and -180 degrees for the helmet. Code has been added in optitrack.py).

The following code adds extra rotation for the target object:
```
object_tc_transformation_matrix = np.matmul(object_tc_transformation_matrix, rotY)
```
where `rotY` is the extra rotation expressed in the form of a 4 x 4 transformation matrix. For example, rotating 90 degress along the Y axis is expressed as:
```
[[0.0, 0.0, 1.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [-1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0]]
```

## Integrate optitrack motion with hand engine pose

For each frame (e.g., if 60 fps, then 300 frames in total for a 5s long motion sequence), the translation, i.e., `x,y,z` positions in the `object_tc_transformation_matrix` are used as the metacarpal joint of the hand. Starting from the metacarpal joint, the entire hand is then recovered using forward kinematics with captured hand joint angles (i.e., XYZ euler angles) and the defined bone length. The two functions `plot_left_hand` and `plot_right_hand` in `plot_motion.py` are used to reconstruct the left and right hands respectively.

Note that the left hand uses a right-handed coordinate system and the right hand uses a left-handed coordinate system.

Please refer to Hand\_Engine\_Readme [LINK] for details about the reconstruction of the hand pose.



## Difference between local and global transformation matrices in optitrack

The raw `optitrack.csv` file contains local (from column 5 to 20) and global (from column 21 to 36) transformation matrices of the optitrack system. NOTE THAT this `local` of optitrack mocap system does not refer to the local throw-catch zone coordinate system.

### local transformation matrix (columns 5-20 in optitrack.csv, NOT USED IN OUR CODEBASE)

It is a 4 x 4 transformation matrix, whose pose is expressed in relative to the start pose of a recorded sequence

### global transformation matrix (columns 21-36 in optitrack.csv)

It is a 4 x 4 transformation matrix, whose pose is with reference to **Optitrack World** frame (Y-Up). In our codebase, we only used this global transformation matrix for matrix manipulation.


## Headband and helmet coordinate system

![helmet_handband_coordinate.png](resources/D6403D046FEE97803D912C8DB100C11F.png =372x492)

The above figure shows the defined frames of the headband and the helmet. The frame of the helmet is same as that of throw-catch zone's origin, with Z-axis parallel to the 5 m side, X-axis parallel to the 2 m side, and Y-axis perpendicular up to the XZ plane. The frame of the headband can be understood as rotating the frame of helmet along the Y-axis counterclockwise with 180 degrees.

In a real scenario, when the primary subject is wearing the helmet and the auxiliary subject is wearing the headband, the coordinate systems will look like as in the following picture:

![helmet_handband_coordinate_real_scenario.png](resources/A4A102E4F59E1D43D95D22ABB44AD561.png =423x482)