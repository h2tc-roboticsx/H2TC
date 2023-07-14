# OptiTrack 

## The origin of the throw-catch zone

The origin of our throw-catch zone refers to the bottom-left corner of the entire throw-catch zone as shown in below. As our data collection spans more than three months, during which, our throw-catch zone in the lab was moved twice. In total, there are **THREE different transformation** matrices collected for the origin after each movement of the throw-catch zone.

<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/workspace.png" width = "400" alt="workspace" />

<!-- 
<table style="border: 1px #000000 solid">
	<thead>
		<tr>
			<th style="border: 1px #000000 solid">Takes</th>
			<th style="border: 1px #000000 solid">Matrix No.</th>
			<th style="border: 1px #000000 solid">Matrix Value</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td style="border: 1px #000000 solid">0-2888</td>
			<td style="border: 1px #000000 solid">#0</td>
			<td style="border: 1px #000000 solid">
            [[-0.99886939, -0.04535922, -0.01408667, 0.42632084], <br>
 [-0.04514784, 0.99886579, -0.0149642, 0.0984003 ],<br>
 [ 0.01474855, -0.01431195, -0.99978858, 7.67951849], <br>
 [ 0.,  0., 0., 1. ]]</td>
		</tr>
		<tr>
			<td style="border: 1px #000000 solid">2889-9788</td>
			<td style="border: 1px #000000 solid">#1 </td>
			<td style="border: 1px #000000 solid">
            [[-9.99963351e-01, 8.30436476e-03, 2.13574045e-03, 1.92400245e-01],<br>
 [ 8.31340413e-03, 9.99956270e-01, 4.25134508e-03, 6.55417571e-02], <br>
 [-2.10037766e-03, 4.26893351e-03, -9.99988700e-01, 2.17126483e+00], <br>
 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]</td>
		</tr>
		<tr>
			<td style="border: 1px #000000 solid">Cell 1x3</td>
			<td style="border: 1px #000000 solid">Cell 2x3</td>
			<td style="border: 1px #000000 solid">Cell 3x3</td>
		</tr>
	</tbody>
</table> -->

|  Takes   | Matrix No. | 
|  :----:  | :----:  | 
| 0-2888  | \#0 |            
| 2889-9788  | \#1 |         
| 9789-12905 |\#2   |         



The following code converts the 4 x 4 transformation matrix captured in the original optitrack system to the local throw-catch zone system, where `origin_transformation_matrix` is one of the three matrices above.

```
object_tc_transformation_matrix = np.matmul(origin_transformation_matrix, object_optitrack_raw_transformation_matrix)
```

## Extra rotation

Since takes 1700 onwards, the orientations of both hands, helmet, and headband were checked everytime before the start of recording. Therefore, no extra rotation is needed for takes from 1700 onwards.

For takes from 520-1559, right hand needs to rotate 90 degrees (addressed in the script optitrack.py)
For takes from 1560-1699, right hand needs to rotate 180 degrees (addressed in the script optitrack.py)
For takes from 1040-1559, left hand needs to rotate 45 degrees (addressed in the script optitrack.py)
For takes from 0-1699, the orientation of the helmet and headband needs to be corrected if using their orientation (rotate along Y axis with extra 45 degrees for the headband, and -180 degrees for the helmet).

The following code adds extra rotation for the target object, where `rotY` is the extra rotation expressed in the form of a 4 x 4 transformation matrix

```
object_tc_transformation_matrix = np.matmul(object_tc_transformation_matrix, rotY)
```

## Integrate optitrack motion with hand engine pose

For each frame (e.g., if 60 fps, then 300 frames in total for a 5s long motion sequence), the translation, i.e., `x,y,z` positions in the `object_tc_transformation_matrix` are used as the metacarpal joint of the hand. Starting from the metacarpal joint, the entire hand is then recovered using forward kinematics with captured hand joint angles (euler angles) and the defined bone length. The detail of how to recover the entire hand can refer to the functions `plot_left_hand` and `plot_right_hand` in `plot_motion.py`.

Note that the left hand uses a right-handed coordinate system and the right hand uses a left-handed coordinate system.

Specifically, takes 0-2888 use origin \#0; takes 2889-9788 use origin \#1; and takes 9789-12905 use origin \#2.

The 4 x 4 transformation matrix of origin \#0 is:
```
[[-0.99886939, -0.04535922, -0.01408667, 0.42632084],
 [-0.04514784, 0.99886579, -0.0149642, 0.0984003 ],
 [ 0.01474855, -0.01431195, -0.99978858, 7.67951849],
 [ 0., 0., 0., 1. ]]
```

The 4 x 4 transformation matrix of origin \#1 is:
```
[[-9.99963351e-01, 8.30436476e-03, 2.13574045e-03, 1.92400245e-01],
 [ 8.31340413e-03, 9.99956270e-01, 4.25134508e-03, 6.55417571e-02],
 [-2.10037766e-03, 4.26893351e-03, -9.99988700e-01, 2.17126483e+00],
 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
```
The 4 x 4 transformation matrix of origin \#2 is:
```
[[-0.99997146, 0.00456379, 0.00601402, 0.19729361],
 [ 0.00454136, 0.99998255, -0.00373799, 0.06776005],
 [-0.00603099, -0.00371067, -0.99997492, 2.48060394],
 [ 0., 0., 0., 1. ]]
```