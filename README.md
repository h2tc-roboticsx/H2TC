# **H<sup>2</sup>TC**: A Large-Scale Multi-View and Multi-Modal Dataset of Human-Human Throw&Catch of Arbitrary Objects

[[Project Page]](https://h2tc-roboticsx.github.io/) [[Technical Paper]](https://h2tc-roboticsx.github.io/underreview/) [[Dataset]](https://h2tc-roboticsx.github.io/notpubyet/) [[Sample Cases]](https://www.dropbox.com/scl/fo/exb0vj76ei789w58bzhqv/h?rlkey=bpc5qr22gr3qgdd3ierf32fpd&dl=0) 
***
<div style="display: flex; justify-content: center;">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/intro.png" width = "1000" >
</div>


This repository provides tools, tutorials, source codes, and supporting documents for the dataset **H<sup>2</sup>TC**. For a quick overview of the dataset, we direct users to the [project website](https://h2tc-roboticsx.github.io/) and our [technical paper](https://h2tc-roboticsx.github.io/underreview/). Briefly, it introduces tools to [record](#recorder), [process](#data-processing), [annotate](#annotation) the dataset, and [retarget](https://github.com/h2tc-roboticsx/H2TC/tree/main/src/utils/pose_reconstruction_and_retargeting) human throw\&catch to robots.

All [source codes](https://github.com/h2tc-roboticsx/H2TC/tree/main/src) can be found in `./src`. The repository also includes [documents and tutorials](https://github.com/h2tc-roboticsx/H2TC/tree/main/doc) in `./doc` that explain in detail the data processing, data hierarchy, annotation, and the content of each data file in the dataset.

<!-- ## Bibtex -->

## Run from scratch
Simply follow the steps below to run the provided tools from scratch:
1. Install the [dependencies](#dependencies). 
2. Fetch the raw data. You have two options to fetch the raw data:
    - Download our captured dataset from [Dropbox](https://h2tc-roboticsx.github.io/notpubyet/). 
    - Capture your own data with our provided [recorder](#recorder) and suggested [sensors](https://h2tc-roboticsx.github.io/recorder/). It helps build your data recording system and then collect more in-distribution data.
3. [Process](#data-processing) the raw data. 
4. (Optional) Annotate the processed data with our provided [annotator](#annotator).


## Dependencies
<!-- <details> -->
<!-- <summary>Details</summary> -->
To run the tools, some dependencies have to be installed first. 

 ### System environment

First, the default and well-tested system environment is

* Ubuntu: 20.04
* Python: 3.8.13
* CUDA: 11.6
* Nvidia driver: 510

We have not tested our codes on other systems yet, so it is recommended to configure the same, or at least a similar environment for the best usage and functionality.

### Softwares

To run the data [processor](#data-processing), install two more applications
* spd-say: text-to-voice converter.
* ffmpeg: video decoder

They can be installed using `apt` if not installed previously on the system:

```bash
sudo apt update
// install spd-say
sudo apt install speech-dispatcher 
// install ffmpeg
sudo apt install ffmpeg
```

### Python Dependencies

The Python dependencies  can be  installed automatically via `pip`:

```bash
pip install -r requirements.txt
```

### Sensors

Our recording framework employs three [ZED](https://www.stereolabs.com/zed-2/) stereo cameras, one [Prophesee](https://www.prophesee.ai/) event camera, a pair of [StretchSense](https://stretchsense.com/) MoCap Pro (SMP) Gloves, and [Optitrack](https://optitrack.com/). Therefore, their SDK tools need to be installed to record and process the dataset.

#### ZED and Metavision SDK

You need to install [ZED SDK](https://www.stereolabs.com/docs/installation/) (3.7.6) and [Metavision SDK](https://docs.prophesee.ai/2.3.0/installation/linux.html) (2.3.0) following the official guidance, so as to use the ZED stereo camera and the Prophesee event camera to record and process the data respectively. 

For user's convenience of installing the specific version (3.7.6) of ZED SDK, we fetch the installer from the official repository and save it in [`./dev/ZED_SDK_Installer`](https://github.com/h2tc-roboticsx/H2TC). All you need is to download the SDK installer, run it and select the modules you prefer following the official [guides](https://www.stereolabs.com/docs/installation/).

Metavision SDK is not packaged in an installer way, so you will have to follow the official [guides](https://docs.prophesee.ai/2.3.0/installation/linux.html) to install it. Particularly, Metavision SDK provides multiple optional modules. Our tool uses only the functionality from the `metavision-essentials`, but you are free to install other modules to explore more functionalities.


#### Test Event and ZED Cameras

Now you should be able to launch the event recorder with the following command if your Prophesee event camera is well connected to a computer:

```bash
metavision_viewer
```

You should also be able to test the connection of  ZED  cameras by running the official [samples](https://github.com/stereolabs/zed-examples/tree/master/svo%20recording/recording/python). 

If you do not have a camera or do not intend to record your own dataset, you can simply check if the modules `pyzed` and `metavision_core` can be successfully imported by your python program.  They will be used only for [post-processing](#data-processing) of the raw data by our dataset.

<!-- </details> -->


## Recorder
<!-- <details>
<summary>Details</summary> -->

<!-- Our recorder integrates the functionality of recording with multiple devices and organizing the recorded contents. -->

<!-- Our recording system consists of 3 [Stereolabs ZED RGBD cameras](https://www.stereolabs.com/zed-2/), 1 [Prophesee event camera](https://www.prophesee.ai/), 1 [StretchSense MoCap Pro gloves](https://stretchsense.com/), and 1 [OptiTrack motion capture system](https://optitrack.com/).  -->

Our recorder integrates the functionality of recording with multiple devices and organizes the recorded contents in a hierarchical manner. 
To collect data with our provided recorder, simply follow the steps below: 

**Step 1**: Enable all recording devices.
* The ZED and Prophesee event cameras should be wired to a host Ubuntu (20.04) machine, where the recorder program is supposed to run. 
* StretchSense MoCap Pro gloves should be connected to a separate Windows machine with its official client software [Hand Engine (HE)](https://stretchsense.com/solution/hand-engine/) running on the same machine. 
* The OptiTrack server can be launched  either on a separate host or on any of the two aforementioned clients. You may need to configure the firewall on each machine to enable the User Datagram Protocol (UDP) communication.

**Step 2**: Update the configuration of OptiTrack NatNet client in [`./src/natnet_client/src/example_main.cpp`](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/natnet_client/src/example_main.cpp), 
and then rebuild it by following  our [tutorial](https://github.com/h2tc-roboticsx/H2TC/tree/main/src/natnet_client). Briefly, you need to configure the OptiTrack server IP address (`char* ip_address`), the recorder IP address (`servaddr.sin_addr`), and the recorder port (`PORT`) according to your own network setting. 

```bash
cd ./src/natnet_client
mkdir build
cd build
cmake ..
make
```

**Step 3**: Initialize your lists of human subjects and objects in `/register/subjects.csv` and `/register/objects.csv` respectively. Each subject and object should lie in a sperate line. Please refer to the sample lists in our repository for a detailed format.

**Step 4**: Launch the main [recorder](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/recorder.py) with the IP and Port of the local machine and of the HE application
```bash
python src/recorder.py --addr IP:PORT --he_addr IP:PORT     
```
There are also some other arguments to configure the recorder optionally:

|  Arguments   | Meanings  | Defaults |
|  :----     | :----  | :----  |
| addr  | IP address and port of the current machine for UDP | 10.41.206.138:3003 |
| he_addr  | IP address and port of the Hand Engine machine for UDP | 10.41.206.141:30039 |
| length  | Time duration (s) of each recording | 5 |
| nposition  | Number of the initial standing locations/cells for subjects | 16 |
| clients  | Clients allowed to communicate | ['optitrack'] |
| zed_num  | Number of ZED cameras for recording | 3 |
| fps  | FPS of ZED recording | 60 |
| resolution  | Resolution of ZED | 720p |
| tolerance  | Frame drop tolerance | 0.1 |

And then run the NatNet client in another terminal:
```bash
./src/natnet_client/build/natnet_client                   
```
Now you should be able to see a prompt indicating that two machines can successfully communicate with each other, if everything goes well.

**Step 5**:  Follow the interactive instructions prompted in the terminal  by the main recorder to perform a recording. The main recorder  will automatically communicate with and command Hand Engine and NatNet client to record multiple data modalities in a synchronous manner. Nevertheless, we do recommend you to regularly check Hand Engine and the NatNet client to see if they break.
<!-- </details> -->

## Data Processing

The data [processor](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/process.py) synchronizes and converts the raw captured data into the processed data of commonly used formats, and organizes them in a hierarchical manner.
Using the processing tool, users can easily

* Process the raw data into the formats as detailed in the table below, or
* Process the raw data into other preferred formats by modifying the provided tool. 
<br>

<table <table border="1" cellspacing="0">
    <caption style="text-align:centering">Table 1. Data modalities and their saving formats in H<sup>2</sup>TC </caption>
    <tr >
        <td rowspan="2" ><b>Device</b></td>
        <td colspan="2" ><b>Raw</b></td>
        <td colspan="2"><b>Processed</b></td>
    </tr>
    <tr >
        <td style="text-align: left;"><b>Data</b></td>
        <td style="text-align: left;"><b>File</b></td>
        <td style="text-align: left;"><b>Data</b></td>
        <td style="text-align: left;"><b>File</b></td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;" rowspan="3">ZED</td>
        <td style="text-align: left;" rowspan="3" >Left- and right-eye RGB videos</td>
        <td style="text-align: left;" rowspan="3">.SVO</td>
        <td style="text-align: left;">RGB images</td>
        <td style="text-align: left;">.PNG</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;">Depth maps (unnormalized)</td>
        <td style="text-align: left;">.NPY</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;">Depth images (normalized)</td>
        <td style="text-align: left;">.PNG</td>
    </tr>
    <tr >
        <td style="text-align: left;" rowspan="2">Event</td>
        <td style="text-align: left;">Binary events in EVT3.0 format</td>
        <td style="text-align: left;">.RAW</td>
        <td style="text-align: left;">Events (<span class="math display">x, y, p, t</span>)</td>
        <td style="text-align: left;">.CSV</td>
    </tr>
    <tr >
        <td style="text-align: left;">Sensor setting for recording</td>
        <td style="text-align: left;">.BIAS</td>
        <td style="text-align: left;">Event images</td>
        <td style="text-align: left;">.JPG</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;" rowspan="4">MoCap Pro</td>
        <td style="text-align: left;">Sensor&#39; reading and hand joint angles</td>
        <td style="text-align: left;">.CSV</td>
        <td style="text-align: left;" rowspan="1">Hand joint motion</td>
        <td style="text-align: left;" rowspan="1">.CSV</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;" >Hand calibration parameters</td>
        <td style="text-align: left;" >.CAL</td>
        <td style="text-align: left;" rowspan="1">Hand joint positions</td>
        <td style="text-align: left;" rowspan="1">.JSON</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;">3D animation visualization</td>
        <td style="text-align: left;">.FBX</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;">Metadata of the recording</td>
        <td style="text-align: left;">.JSON</td>
    </tr>
    <tr >
        <td style="text-align: left;">OptiTrack</td>
        <td style="text-align: left;">Local and global transformatiom matrices</td>
        <td style="text-align: left;">.CSV</td>
        <td style="text-align: left;">6D global motion in the throw&catch frame</td>
        <td style="text-align: left;">.CSV</td>
    </tr>
</table>
<br>

We refer users to the [data processing](https://github.com/h2tc-roboticsx/H2TC/blob/main/doc/processing_techdetails.md) for full technical details on how we process the multi-modal and cross-device raw data, and to the [data file explanation](https://github.com/h2tc-roboticsx/H2TC/blob/main/doc/data_file_explanation.md) and our [technical paper](https://h2tc-roboticsx.github.io/underreview/) for a detailed introduction of the data hierarchy and the content of each involved data file.
<br>

### How to Process

#### Step 1: Fetch the raw data

You can access all raw data from <a href="https://h2tc-roboticsx.github.io/notpubyet/">Dropbox</a> . The raw data of each recorded throw&catch activity is packed in a `.zip` file. First, download the raw data to your own folder `raw_data_path`
```
raw_data_path
└──011998.zip           // 011998 is the take number
```

#### Step 2: Extract the raw data 

We provide a scripted [extractor](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/extract.py) to unzip the packed raw data and also to organize all raw files in a suitable data hierarchy, as mentioned before. Run the following command below to extract  the raw data to your target path `target_path`: 
```
python src/extract.py --srcpath raw_data_path --tarpath target_path
```
<small>`--srcpath` is where you download and save the zipped raw files. `--tarpath` is the target path where you want to save the extracted data files.</small>

Each extracted zip file will be organized in a hierarchical structure under the folder `target_path/data`. For example, the raw data files of the recording "011998" will be organized as below

```bash
target_path
└──data
    └──011998
        ├──hand
        │   ├──P1L.csv / P1R.csv
        │   ├──P1LMeta.json / P1RMeta.json
        │   ├──P1L.cal/P1R.cal
        │   └──P1L.fbx/P1R.fbx
        ├──{zed-id}.svo  
        ├──{zed-id}.csv 
        ├──event_{timestamp}.raw
        ├──event.bias
        └──optitrack.csv
```

`{zed-id}` indicates and includes three involved ZED devices, including `17471`, `24483054` and `28280967`, which are the fixed third-person (side), the dynamic egocentric and the fixed third-person (back) respectively. `{timestamp}` is the initial timestamp of the event camera in the UNIX format.  A detailed explanation of the data hierarchy and the content of each raw data file is provided in [data file explanation](https://github.com/h2tc-roboticsx/H2TC/blob/main/doc/data_file_explanation.md/#data). 

#### Step 3: Process the extracted data
Once the raw data is extracted and organized appropriately, run the following command

```bash
python src/process.py --datapath target_path/data
```
<small>`--datapath` is where the extracted raw data files are saved. There are also some other optional arguments that can be used to configure the processor</small>

|  Arguments   | Meanings  | Defaults |
|  :----     | :----  | :----  |
| takes     | The id(s) of recording(s) to be processed. Set 'None' to process all recordings in the `data` directory. This can be given with a single integer for one take, or with a range linked by '-' for a  sequence of consecutive recordings,  e.g. '10-12' for the recordings {000010, 000011, 000012}. | None |
| fps_event | FPS for decoding event stream into image frames. | 60 |
| fps_zed   | FPS for decoding ZED RGB-D frames. This should be equal to the value used in recording. | 60 |
| duration   | The duration of recording in seconds. | 5 |
| tolerance   | The tolerance of frame drop in percentage for all devices. | 0.1 |
| depth_img_format   | The image format of the exported RGB-D frames for ZED cameras. Either 'png' or 'jpg'.  | png |
| xypt   | Set `True` to export event stream in the `.xypt` format, which is the raw format of  events. | False |
| npy   | Set `True` to export depth stream in the `.npy` format, which is 3-dimensional numpy arrary holding the unnormalized depth estimation of each frame.  | False |
| depth_accuracy   | The float precision for the unnormalized depth maps. The  depth maps are not exported by default until the flag 'npy' is set to `True`. Either 'float32' or 'float64'. | float32 |
| datapath   | The directory of raw data files that needs to specify.   | None |

You can change the default settings by adding more arguments into your command. For example, we do not export the depth `depth.npy` and the event `xypt.csv`files by default (i.e. `xypt` and `npy` are set `False` by default), as they are time/space-consuming. If you need them, simply attach `--npy` and `--xypt` to the command: 

```bash
python src/process.py --datapath target_path/data --npy --xypt
```

Once the data processing is done, as shown below, the raw data files will be moved into a new directory `target_path/data/raw/`,  and the processed data files will be stored in a directory with the following default path`target_path/data/processed/`. 

```bash
target_path
└──data
    └──011998
        ├──raw              // all raw data as above
        │ └──......
        └──processed        // all processed data
            ├──rgbd0/rgbd1/rgbd2
            │    ├──left_frame_id.png
            │    ├──depth_frame_id.png
            │    └──depth.npy
            ├──event
            │    └──frame_id.png
            ├──rgbd0_ts/rgbd1_ts/rgbd2_ts.csv
            ├──event_xypt.csv
            ├──event_frame_ts.csv
            ├──left_hand_pose.csv
            ├──right_hand_pose.csv
            ├──left_hand_joint_positions.json
            ├──right_hand_joint_positions.json
            ├──sub1_head_motion.csv
            ├──sub1_left_hand_motion.csv
            ├──sub2_head_motion.csv
            ├──object.csv
            └──alignment.json
```
The data hierarchy and the content of each processed data file are explained in detail in [data file explanation](https://github.com/h2tc-roboticsx/H2TC/blob/main/doc/data_file_explanation.md/#data) and our technical [paper](toadd).



### Customized Processing 
If you want to customize the processing process, please: 
  * First follow the Steps 1 and 2 in [How to Process](#how-to-process) to fetch the organized raw data. 
  * Then in Step 3, customize your own processing process by diving into the full technical details in the [data processing](https://github.com/h2tc-roboticsx/H2TC/blob/main/doc/processing_techdetails.md). This document explains how we process the multi-modal, cross-device data streams.

### ❗Trouble Shooting 

#### 1. Reprocess the processed takes

If you want to reprocess the processed takes, you will have to manually remove their entire folders first. If you only want to reprocess part(s) of a take, e.g. RGB and depth streams, you just need to remove their corresponding files (folders).

#### 2. Fail in decoding ZED frames 

The current mechanism allows for maximally 10 failed attempts to decode (or grab in ZED term) a ZED RGB-D frame. Once the decoding process fails for more 10 times, it will abort and the data processing will jump to the next part, e.g. the next ZED device or the next stream modality. Those frames that have already been decoded will be stored, while the rest frames will be ignored. This issue usually happens when decoding the last frame.

To fix this issue, one can simply reprocess the problematic takes via [reprocess the processed take](#1-reprocess-the-processed-take) as described above. 
<!-- </details> -->

##  Annotation
<!-- <details> -->
<!-- <summary>Details</summary> -->

### Segmentation and Annotation
The dataset is provided with a hierarchy of segmentation and annotations, both semantic and dense. Briefly, each recorded  throw&catch activity in H<sup>2</sup>TC is segmented into four phases, including *pre-throwing*, *object flying*, *catching* and *post-catching*, with three manually annotated moments including *throw*, *catch_touch* and *catch_stable*. 

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/h2tc-roboticsx.github.io/main/assets/images/seg_00.png" width=800>
</div>


The subjects' behaviors are manually checked and annotated with symbolic labels in terms of *grasp mode* and *hand locations* (as shown schematically below. Please click the picture if annotations are invisible on your browser or refer to the figure in our paper). The subjects' exact *initial standing locations* and the average flying *speed* of the object are also automatically annotated as quantitative labels. 

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/h2tc-roboticsx.github.io/main/assets/images/instruction.png" width=200>
</div>



The complete annotation hierarchy is detailed below:
<table width=1000 style="text-align: left;">
    <tr>
        <td>Name</td>
        <td>Description</td>
        <td>Value</td>
        <td>Labeling Type</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td>Object</td>
        <td>The thrown object</td>
        <td><i>&#39;object_id&#39;</i></td>
        <td>automatic</td>
    </tr>
    <tr>
        <td><b>Throw</b></td>
        <td>The moment when the subject&#39;s hand(s) breaks with the thrown object during throwing</td>
        <td>UNIX timestamp</td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Grasp mode</td>
        <td>The subject&#39;s grasp mode to throw the object at the "throw" moment</td>
        <td>{<i>&#39;left&#39;</i>, <i>&#39;right&#39;</i>, <i>&#39;both&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Throw vertical</td>
        <td>The vertical location(s) of the subject&#39;s  hand(s) to throw the object at the "throw" moment</td>
        <td>{<i>&#39;overhead&#39;</i>, <i>&#39;overhand&#39;</i>, <i>&#39;chest&#39;</i>, <i>&#39;underhand&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Throw horizontal</td>
        <td>The horizontal location(s) of the subject&#39;s  hand(s) to throw the object at the "throw" moment</td>
        <td>{<i>&#39;left&#39;</i>, <i>&#39;middle&#39;</i>, <i>&#39;right&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Catch vertical</td>
        <td>The vertical location(s) of the subject&#39;s hand(s) to catch at the "throw" moment</td>
        <td>{<i>&#39;overhead&#39;</i>, <i>&#39;overhand&#39;</i>, <i>&#39;chest&#39;</i>, <i>&#39;underhand&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Catch horizontal</td>
        <td>The horizontal location(s) of the subject&#39;s hand(s) to catch at the "throw" moment</td>
        <td>{<i>&#39;left&#39;</i>, <i>&#39;middle&#39;</i>, <i>&#39;right&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Throw location</td>
        <td>The subject&#39;s exact body location to throw at the "throw" moment</td>
        <td>(<i>x</i>, <i>z</i>)</td>
        <td>automatic</td>
    </tr>
    <tr>
        <td>- Catch location</td>
        <td>The subject&#39;s exact  body location to catch at the "throw" moment</td>
        <td>(<i>x</i>, <i>z</i>)</td>
        <td>automatic</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td><b>Catch_touch</b></td>
        <td>The moment when the subject&#39;s hand(s) first touches the flying  object during catching</td>
        <td>UNIX timestamp</td>
        <td>manual</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td>- Catch location</td>
        <td>The subject&#39;s exact location to catch the object at the "catch_touch" moment</td>
        <td>(<i>x<i>, <i>z<i>)</td>
        <td>automatic</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td>- Object speed</td>
        <td>The  object&#39;s average speed during free flying</td>
        <td>m/s</td>
        <td>automatic</td>
    </tr>
    <tr>
        <td><b>Catch_stable</b></td>
        <td>The moment when the subject catches the flying object stably during catching</td>
        <td>UNIX timestamp</td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Grasp mode</td>
        <td>The subject&#39;s grasp mode to catch the object at the "catch_stable" moment</td>
        <td>{<i>&#39;left&#39;<i>, <i>&#39;right&#39;<i>, <i>&#39;both&#39;<i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Vertical location</td>
        <td>The vertical location(s) of the subject&#39;s hand(s) to catch the object at the "catch_stable" moment</td>
        <td>{<i>&#39;overhead&#39;<i>, <i>&#39;overhand&#39;<i>, <i>&#39;chest&#39;<i>, <i>&#39;underhand&#39;<i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Horizontal location</td>
        <td>The horizontal location(s) of the subject&#39;s hand(s) to catch at the "catch_stable" moment</td>
        <td>{<i>&#39;left&#39;<i>, <i>&#39;middle&#39;<i>, <i>&#39;right&#39;<i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Catch result</td>
        <td>The result on whether the object is stably caught by the subject</td>
        <td>{&#39;<i>success<i>&#39;, &#39;<i>fail<i>&#39;} </td>
        <td>manual</td>
    </tr>
</table>

### Annotator

In case you want to annotate our dataset and your custom-captured data, 
we provide an [annotator](https://github.com/h2tc-roboticsx/H2TC/blob/main/src/annotate.py), with which one can easily annotate catch&throw activities via an interactive interface. 

To use the annotator, please follow the steps below:  
- Step 1: Process and store all raw data in the directory `YOUR_PATH/data/take_id/processed`. The raw data processing can be achieved as suggested previously in [data processing](#data-processing). 
- Step 2: Run the following command to launch the annotation tool.  

```
python src/annotate.py --datapath YOUR_PATH/data
```

There are some other arguments allowed to configure the annotation optionally:

|  Arguments   | Meanings  | Defaults |
|  :----     | :----  | :----  |
| takes     | The take ids to be annotated. Set None to annotate all takes in the 'data' directory. This can be given with a single integer number for one take or with a range linked by '-' for consecutive takes, e.g. '10-12' for takes [000010, 000011, 000012]. | None |
| review | Set true to review the takes that have been already annotated before. By default (False), the annotated takes will not be displayed for annotation again. | False |
| datapath | The directory of  processed data. Users need to specify it.  | None |

By default, the takes that are "failed" or have been already annotated (either "finished" or "problematic") are ignored by the program so that they will not present in the annotator.  To review the annotated takes, you should run the program with the option `--review` like:


```
python src/annotate.py --review
```


### Interface

Once running the above command, an interactive prompt interface will appear excluding the orange bars (they are figure annotations). 

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/h2tc-roboticsx.github.io/main/assets/images/annotator_labeled.jpg" alt="interface"  width=600>
</div>


The interface displays multi-view RGB (left column), depth (middle column), egocentric event (top right sub-window), and hand motion (middle right sub-window) streams. Annotators can check synchronized streams frame by frame via the keyboard (`left arrow` and `right arrow` respectively). 

The interface also provides `an information panel` (bottom right sub-window), which allows annotators to annotate the streams with keyboard and display  the annotation result in real-time. 
<br>

<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/h2tc-roboticsx.github.io/main/assets/images/info_panel_explanation.png" width = "400" alt="info_panel">
</div>


Each entry in the information pannel corresponds to an annotation, as described [above](#segmentation-and-annotation), 
<br>

| Number |   Representation |  Annotation Name |
|  :----:     | :----  | :----  |
|   1    |    The status of the annotation: `finished`, `unfinished` or `problematic`   |    \   |
|   2    |    The grasp mode used to throw at the *throw* moment  |    Grasp mode  |
|   3    |    The exact body location of the thrower at the *throw* moment    |    Throw location  |
|   4    |    The exact body location of the catcher at the *throw*  moment    | Catch location |
|   5    |    The average flying speed of the thrown object   | Object speed|
|   6    |    The vertical hand location of the thrower at the *throw*   moment  |Throw vertical  |
|   7    |    The horizontal hand location of the thrower at the  *throw*  moment  |Throw horizontal  |
|   8    |    The vertical hand location of the catcher at the *throw* moment    |Catch vertical  |
|   9    |    The horizontal hand location of the catcher at the  *throw*   moment | Catch horizontal |
|   10    |    The frame number and the timestamp of the moment *throw*   | \ |
|   11    |    The grasp mode used to catch at the *catch (stable)* moment    | Grasp mode |
|   12    |    The exact body location of the catcher at the *catch (touch)* moment    | Catch location |
|   13    |    The vertical hand location of the catcher at the *catch (stable)* moment    | Vertical location |
|   14    |    The horizontal hand location of the catcher at the *catch (stable)* moment  | Horizontal location |
|   15    |    The frame number and the timestamp of the moment *catch (touch)*  | \ |
|   16    |    The frame number and the timestamp of the moment *catch (stable)*  | \ |
<br>

###  Interaction Operations

Annotators can manually interact with the interface to select semantic labels with the keyboard as defined below, while the dense labels, e.g. the object speed and subject locations, will be automatically annotated by the interface. Any modification to the annotation result will be immediately saved in the corresponding annotation file `YOUR_PATH/annotations/take_id.json`.


| Key value  |     Operation |
|:----------|:-------------|
|"right arrow"| Next frame of the current take being annotated|
|"left arrow"| Last frame  of the current take being annotated|
|"down arrow"| Next take to annotate|
|"up arrow"| Last take that has been annotated|
|"return"| Take the current frame as a moment of *throw*, *catch_touch*, and *catch_stable* in order|
|"del"| Remove the last annotated moment|
|"Q"| Switch and select the value of panel 2 (Grasp mode) among *left*, *right*, and *both* |
|"A"| Switch and select the value of  panel 6 (Throw vertical) among *overhead*, *overhand*, *chest*, and *underhand*|
|"S"| Switch and select the value of panel 7 (Throw horizontal) among *left*, *middle*, and *right*|
|"D"| Switch and select the value of  panel 8 (Catch vertical) among *overhead*, *overhand*, *chest*, and *underhand*|
|"F"| Switch and select the value of  panel 9 (Catch horizontal) among *left*, *middle*, and *right*|
|"Z"| Switch and select the value of  panel 11 (Grasp mode) among *left*, *right*, and *both*|
|"C"| Switch and select the value of  panel 13 (Vertical location) among *overhead*, *overhand*, *chest*, and *underhand*|
|"V"| Switch and select the value of  panel 14 (Horizontal location) among *left*, *middle*, and *right*|
|"space"| Switch and select the value of panel 1 (annotation status) between *finished* and *unfinished*|
|"backspace"| Switch and select the value of  panel 1 (annotation status) between *problematic*" and *unfinished*|


### &#x2022; Note ❗

#### 1. Main annotation camera

We suggest that users rely mainly on the third-person (side)  and egocentric views  to annotate. The third-person (back) view can be used an auxiliary, when significant occlusion happens in the former two views. 

The viewing angle of the egocentric camera is higher than the normal height of human eyes,resulting in a top-down viewing angle. This may lead to a biased observation of the vertical hand location. You can use the third-person (back) view to provide additional information. 


#### 2. Handling missing frames or data

It is possible that some data is missing in your labeled frame. 
Then the annotation can not be switched to the status of "finished" due to the missing data. 

For example, when you label the *throw* moment and if there is no OptiTrack data, the information panel appears like below. 
To handle the issue, you should seek closer frames that include complete data and indicate the same event of the moment. 
If no frame is qualified, the entire take should be annotated as "problematic" and skip to the next take.


<div align="center">
<img src="https://raw.githubusercontent.com/h2tc-roboticsx/H2TC/main/doc/resources/annotation/missing_data_anno.png" width = "400" alt="missing_data_anno">
</div>



### Visualization

The first step of using our visualization tool is to prepare the processed data. This can be done by the provided [processor](#data-processing). Alternatively, for a quick browse, we offer the processed data of several sample takes that can be directly downloaded from [here](https://h2tc-roboticsx.github.io/dataset/#sample-cases). Eventually, you should have the data stored in a path similar to this: `PARENT_PATH/data/take_id/processed`.

Now you can run the following command to launch the visualization tool:

```
python src/visualize.py --datapath PARENT_PATH/data --take take_id --speed 120
```

The argument `--take` specifies the ID of the take to be visualized if set, otherwise the first take under the given path will be loaded. `--speed` specifies the FPS for playing the frames of streams.

Once the interface is launched, you can navigate the visualization through the following operations:

&ensp; 1. `space`: play/pause the videos of all streams <br>
&ensp; 2. `right arrow`: pause the video if played and forward to the next frame <br>
&ensp; 3. `left arrow`: pause the video if played and backward to the last frame <br>
