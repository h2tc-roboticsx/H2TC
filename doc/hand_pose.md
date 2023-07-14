# Reconstruct the left hand

**Left hand uses a right-handed coordinate system** 

## Right-handed coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/97CE3BB762B65208FED74A7D8A0D4C12.png" width = "400" alt="left_hand">


The above figure shows the right-handed coordinate system of the left hand. Each joint has its own XYZ frame. The X-axis is along the bone, the Y-axis is perpendicular to the palm, and the Z-axis is perpendicular to the XY plane. The enlarged frame at the bottom is put there for clarification and easier understanding.

## Our throw-catch zone coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/C650A1275361BA54AB728D0C141801F6.png" width = "400" alt="opti_lefthand">

The above figure shows the coordinate system of the captured hand motion in the our throw-catch zone. The frame is put there for clarification and easier understanding. In practice, the origin of the frame is around the center of the back of the hand. For this coordinate system, Y-axis is perpendicular up to the back of the  hand, Z-axis is parallel to the finger tip direction, and X-axis is perpendicular to the YZ plane.

## Align right-handed coordinate system with our throw-catch zone coordinate system

As the orientation of the right-handed coordinate system differs from that of the our throw-catch zone coordinate system, we use two rotation matrices to convert the orientation of the hand coordinate system to the same as the our throw-catch zone.

Specifically, we first rotate the hand coordinate system -180 degrees along the X-axis, and then rotate it -90 degrees along the Y-axis. 


## Reconstruction
As mentioned in Optitrack\_Readme [TODO link], for each hand pose, we use the translation, i.e., x, y, z positions in its associated 4 x 4 transformation matrix that has been converted to the our throw-catch zone coordinate system as the metacarpal joint. We then reconstruct the entire hand pose starting from the metacarpal joint with the captured hand joint angles (XYZ euler angles) and the defined hand bone length (see Bone length section in this readme) using **Forward Kinematics**:

Specifically, the XYZ spatial position `P` of a left hand finger joint **(except the metacarpal joint)** in the our throw-catch zone can be calculated using the following equations: (TODO)




# Reconstruct the right hand

**Right hand uses a left-handed coordinate system**

## Left-handed coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/B08B2DCADADCFD3D2B101CC1AFFBA015.png" width = "400" alt="right_hand">

The above figures shows the left-handed coordinate system of the right hand. Each joint has its own XYZ frame. The X-axis is along the bone, the Y-axis is perpendicular up towards the back of the hand, and the Z-axis is perpendicular to the XY plane. The enlarged frame at the bottom is put there for clarification and easier understanding.

## our throw-catch zone coordinate system
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/018D81940FEC63AD318DBD8B5AF0FF98.png" width = "400" alt="opti_righthand">

Similar as mentioned above, the above figure shows the coordinate system of the captured hand motion in the our throw-catch zone. The frame is put there for clarification and easier understanding. In practice, the origin of the frame is around the center of the back of the hand. For this coordinate system, Y-axis is perpendicular up to the back of the  hand, Z-axis is parallel to the finger tip direction, and X-axis is perpendicular to the YZ plane.


## Coordinate system conversion
To reconstruct the right hand, we first convert the left-handed coordinate system to the right-handed one using a matrix `t_h` (is this description corret??? also is the comment in line 126 above t\_h correct in plot\_motion.py??? what does t\_h exactly do? convert coordinate system or convert data??)

```
t_h = [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
```
We also apply a rotation matrix rotY as shown in the `plot_right_hand` function in `plot_motion.py` to rotate the hand coordinate system -90 degrees along the Y-axis

```
rotY = [[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]
```

## Reconstruction
Similar to the reconstruction of the left hand, we use the translation of the converted 4 x 4 transformation matrix that is associated with the hand pose as the metacarpal joint, and then reconstruct the entire hand pose starting from the metacarpal joint with the captured hand joint angles (XYZ euler angles) and the defined hand bone length (see Bone length section in this readme) using **Forward Kinematics**:

Specifically, the XYZ spatial position `P` of a right hand finger joint **(except the metacarpal joint)** in the our throw-catch zone can be calculated using the following equations: (TODO)



# Metacalpal joint offset

Note that, as a common practice, we did not attach the markers directly to the hand, but fixed markers on a rigid object, and then attached the rigid object to the back of the hand (See figure below). As the geometric center of the rigid object does not exactly align with the metacarpal joint of a hand, there is an offset between the reconstructed hand and the actual hand in terms of their spatial positions in the our throw-catch zone. However, this offset is minor, and does not change the motion of the hand.

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/045A78822B66A604FE54BEC901DEC56E.png" width = "400" alt="glove_with_markers">

# Hand size (<u>BONE LENGTH HAS TO BE DETERMINED</u>)

Note that the finger bone length we used for visualization purpose in `plot_motion.py` is enlarged. Also, it is not necessary to measure every single subject's finger bone length. **Therefore, we use the average bone length to reconstruct the hand as a common practice**. 


# Bone length (<u>BONE LENGTH HAS TO BE DETERMINED</u>)

## Visualization
We use the following set of bone lengths to reconstruct and visualize the hands in `plot_motion.py`:
```
         Metacarpal Proximal Middle Distal
thumb =  [0.25,     0.11,           0.06]
index =  [0.34,     0.15,    0.08,  0.06]
middle = [0.33,     0.15,    0.10,  0.07]
ring =   [0.31,     0.13,    0.10,  0.06]
pinky =  [0.3,      0.08,    0.06,  0.06]
```

## Paper
The bone lengths in the paper are measured from an acutual hand (TODO, better to provide an average bone length model)
```
         Metacarpal Proximal Middle Distal
thumb =  [6.0,      4.0,            3.5]
index =  [8.0,      5.5,     3.0,   2.5]
middle = [8.0,      6.0,     3.5,   2.7]
ring =   [7.5,      5.5,     3.3,   2.5]
pinky =  [6.5,      4.5,     2.5,   2.5]
```


# t\_h

Explanation of t_h