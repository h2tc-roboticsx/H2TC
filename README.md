# H<sup>2</sup>TC: A Large-Scale Multi-View and Multi-Modal Dataset of Human-Human Throw&Catch of Arbitrary Objects

[[Project Page]](https://lipengroboticsx.github.io/) [[Paper]]() [[Data]](https://www.dropbox.com/sh/ahet936ypjs1582/AACNYG0sjf1XdVxuZVLVL4fFa?dl=0) [[Sample Cases]](https://www.dropbox.com/sh/dghb9k4w4w938q0/AAAMIjWBbzy290QI_Nljocqda?dl=0) 

This repository provides details and tools for the dataset **H<sup>2</sup>TC**. 
For a quick overview of the dataset, we refer readers to the [project website](https://lipengroboticsx.github.io/). 
Briefly, we have introduced tools to [record](#recorder), [process](#data-processing), and [annotate](#annotator) a human-human throw-catch activity. All source codes are available in the folder `./src`. 

<!-- ## Bibtex -->

## Run from scratch
You can follow the steps to run from scratch:
1. Install the [dependencies](#dependencies). 
2. Fetch the raw data. You have two options to fetch raw data:
    - Download our captured data [here](https://www.dropbox.com/sh/ahet936ypjs1582/AACNYG0sjf1XdVxuZVLVL4fFa?dl=0). 
    - Capture your own data with our provided [recorder](#recorder). It helps build your own data collecting system. 
3. [Process](#data-processing) the raw data. 
4. (Optional) Annotate the processed data with our provided [annotator](#annotator).


## Dependencies
<!-- <details> -->
<!-- <summary>Details</summary> -->
To run the tools, some dependencies have to be installed. 

 ### System environment

First, the default and well-tested system environment is

* Ubuntu: 20.04
* Python: 3.8.13
* CUDA: 11.6
* Nvidia driver: 510

We have not tested our codes on other environments, so it is recommended to configure the same, or at least a similar environment for the best experience.

### Softwares

<!-- Apart from them, there are two more applications you have to check if you have in order to run the **processing** script: -->
The data [processor ](#data-processing) depends on

* spd-say: text-to-voice convertor.
* ffmpeg: video decoder

They can be installed, if not have, using the package management tool APT:

```bash
sudo apt update
// install spd-say
sudo apt install speech-dispatcher 
// install ffmpeg
sudo apt install ffmpeg
```

### ZED and Metavision (Event camera) SDK

Next, you need to install [ZED SDK](https://www.stereolabs.com/docs/installation/) (3.7.6) and [Metavision SDK](https://docs.prophesee.ai/2.3.0/installation/linux.html) (2.3.0) following the official guidance, so as to use ZED stereo camemra and Prophesee event camera to record and process the data respectively. 

For your convenience of installing the older version (3.7.6) of ZED SDK, we copied one from the official source in the ***/dev*** directory. All you need is to download the SDK installer, run it and select the modules you want following the [guide](https://www.stereolabs.com/docs/installation/).

Metavision SDK is not packaged in an installer way, so you will have to follow the official [guide](https://docs.prophesee.ai/2.3.0/installation/linux.html) to install it. Particularly, Metavision SDK provides several optional modules. Our code uses only the functionality from the `metavision-essentials`, but you are free to install other optional modules or not.

### Python Dependencies

Last, the remaining Python dependencies include

* addict
* mttkinter
* openpyxl
* scipy
* pandas
* numpy
* opencv-python

and can be automatically installed via pip:

```bash
pip install -r requirements.txt
```

<!-- ### Docker

Alternatively, we also provide a ready-to-use [Docker](https://www.docker.com/) with all dependencies installed in our git repository ***<u>TODO link</u>***. To use it, Docker should have been already successfully installed. -->

### Test ZED and Event Cameras

now you should be able to run the following command to launch the event recorder with your Prophesee event camera connected to the computer:

```p
metavision_viewer
```

<!-- <u>***TODO: picture of running successfully***</u> -->

You should also be able to record using ZED by running the official [sample](https://github.com/stereolabs/zed-examples/tree/master/svo%20recording/recording/python). If you don't have a camera or don't intend to record, you could just check if `pyzed` and `metavision_core` modules can be successfully imported in your python program. If any failure, you should inspect your installation if done manually and, unfortunately, troubleshooting this is beyond the scope of this instruction.
<!-- </details> -->



## Recorder
<!-- <details>
<summary>Details</summary> -->

Our recorder integrates the functionality of arranging the contents to be recorded and recording with multiple devices. Our recording system consists of  3 [Stereolabs ZED RGBD cameras](https://www.stereolabs.com/zed-2/), 1 [Prophesee event camera](https://www.prophesee.ai/), 1 [StretchSense two-hand pose capturing glove](https://stretchsense.com/), and 1 [OptiTrack motion capture system](https://optitrack.com/). To record these modalities synchronously, please: 

**Step 1**, enable all recording devices and ensure each of them function smoothly. 
* Three ZED cameras and one Prophesee event camera should be wired to the host where the recorder program is supposed to run. 
* StretchSense MoCap Pro gloves should be wireless connected to a Windows machine with its official client software [Hand Engine](https://stretchsense.com/solution/hand-engine/) running. 
* OptiTrack server can be either operated on a separate host, recommended by us, or on the same host as any of the two aforementioned ones as long as the computational resource allows and the performance will not be thus compromised. You may need to configure the firewall on each machine to allow the user datagram protocol (UDP) communication among them.

**Step 2**, update the configuration in our OptiTrack NatNet client code and rebuild the NatNet client by following [our NatNet document](https://github.com/lipengroboticsx/H2TC_code/tree/main/src/natnet_client). Briefly, you need to set the values of OptiTrack server IP address (`char* ip_address`), recorder IP address (`servaddr.sin_addr`), and recorder port (`PORT`) according to your network setting in the file [`/src/natnet_client/src/example_main.cpp`](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/natnet_client/src/example_main.cpp). 
```p
cd src/natnet_client
mkdir build
cd build
cmake ..
make
```

**Step 3**, initialize your lists of subjects and objects in the corresponding files `/register/subjects.csv` and `/register/objects.csv` respectively. Each subject and object should lie in a new line. Please check the sample lists in our repository for a detailed format.

**Step 4**, launch the main recorder application with the IP and Port of the local machine and of the HE application:
```
python src/recorder.py --addr IP:PORT --he_addr IP:PORT     
```
[tbd: add recorder.py arguments]<br>
Several arguments are allowed to configure the processing when running the script [src/recorder.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/recorder.py). The available arguments are:

|  Arguments   | Meanings  | Defaults |
|  :----     | :----  | :----  |
| addr  | ip address and port of the current machine for UDP | 10.41.206.138:3003 |
| he_addr  | ip address and port of the Hand Engine machine for UDP | 10.41.206.141:30039 |
| length  | the length of recording | 5 |
| nposition  | num of position squares to locate subjects | 16 |
| clients  | clients allowed to communicateP | ['optitrack'] |
| zed_num  | number of ZED cams for recording | 3 |
| fps  | FPS for ZED recording | 60 |
| resolution  | resolution for ZED recording | 720p |
| tolerance  | frame drop tolerance | 0.1 |

And then run the NatNet client in another terminal:
```
./src/natnet_client/build/natnet_client                   
```

now you should be able to see the prompt indicating that these two applications have successfully communicated with each other, if everything goes well, as shown below 
[tbd: pictures of connection established ??]
<!-- <u>***TODO pictures of connection established.***</u> -->

**Step 5**, operate the main recorder to record following the interactive instruction. The main recorder will automatically communicate with and command Hand Engine and NatNet client to record. Nevertheless, we do recommend you regularly check Hand Engine and the NatNet client to see if it bugs.
<!-- </details> -->
<!-- <u>***TODO picture of a complete take***</u> -->

## Data Processing
<!-- <details>
<summary>Details</summary> -->

Our [processor tool](https://github.com/lipengroboticsx/H2TC_code/tree/main/src) converts the raw data into commonly used formats. With the tool, one can easily
* process all raw data into the formats as detailed in the table below, or,
* process raw data into one's preferred formats for customized needs by modifying the provided tool. 


We refer readers to [`/doc/processing_techdetails.md`](https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md) for technical details on how we process the multi-modal and cross-device raw data.

<table <table border="1" cellspacing="0">
    <caption style="text-align:centering">Table 1. Data modalities and formats</caption>
    <tr>
        <td rowspan="2" ><b>Device</b></td>
        <td colspan="2" bgcolor="#eeeeee"><b>Raw</b></td>
        <td colspan="2" bgcolor="#eeeeee"><b>Processed</b></td>
    </tr>
    <tr >
        <!-- <td>\multicolumn{1}{c}{}</td> -->
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
        <td style="text-align: left;">Sensors&#39; reading and hand joint angles</td>
        <td style="text-align: left;">.CSV</td>
        <td style="text-align: left;" rowspan="4">Hand joint values</td>
        <td style="text-align: left;" rowspan="4">.CSV</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td style="text-align: left;" >Hand calibration parameters</td>
        <td style="text-align: left;" >.CAL</td>
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
        <td style="text-align: left;">6D pose in throw-catch coordinate</td>
        <td style="text-align: left;">.CSV</td>
    </tr>
</table>

### &#x2022; How to Process

#### Step 1: Get the raw data
Download the <a href="https://www.dropbox.com/sh/ahet936ypjs1582/AACNYG0sjf1XdVxuZVLVL4fFa?dl=0"> raw data</a> to your own folder `RAWDATA_PATH`. Each recording of our raw data is packed in a .zip file.
```
RAWDATA_PATH
└──011998.zip           - 011998 is the take number
```

#### Step 2: Extract the packed raw data 
Run the following command to extract (unzip) the packed raw data to your target path `YOUR_PATH`: 
```
python src/extract.py --srcpath RAWDATA_PATH --tarpath YOUR_PATH
```
<small>`--srcpath` is where you downloaded the packed raw data. `--tarpath` is the target path where you want to extract the packed data.</small>

Each extracted recording should be organized under the folder `YOUR_PATH/data`. 

<small>For example, the raw data of the recording "011998" should be organized in a way as below, </small>

<!-- * ***YOUR_PATH/***
  * ***data/***
    * ***011998/***
      * **{ZED-ID}.svo**  (<em>raw data of ZED camera with the ID</em>)
      * **{ZED-ID}.csv** (<em>timestamps of the raw data of ZED camera with the ID</em>)
      * **event_{timestamp}.raw** (<em>raw data of Event camera</em>)
      * **optitrack.csv** (<em>raw data of optitrack</em>)
      * ***hand/***
        * **P1L.csv / P1R.csv** (<em>raw data of left / right hand pose for Hand Engine</em>)
        * **P1LMeta.json / P1RMeta.json** (<em>metadata of recording for Hand Engine</em>) -->
```
YOUR_PATH
└──data
    └──011998
        ├──hand
        │   ├──P1L.csv / P1R.csv
        │   ├──P1LMeta.json / P1RMeta.json
        │   ├──P1L.cal/P1R.cal
        │   └──P1L.fbx/P1R.fbx
        ├──{ZED-ID}.svo
        ├──{ZED-ID}.csv 
        ├──event_{timestamp}.raw
        ├──event.bias
        └──optitrack.csv
```

`{ZED-ID}` indicates three ZED devices, `17471`, `24483054` and `28280967`, which are the fixed third-person (side) view, the dynamic egocentric view and the fixed third-person (back) view respectively.
`{timestamp}` is the initial timestamp of the event camera in a UNIX format. 
A detailed explanation of each raw data file is provided in [`/doc/data_file_explanation.md`](https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/data_file_explanation.md/#data). 

#### Step 3: Process the extracted data
Once the raw data is extracted and organized properly as in step 2), then run the following command:

```python
python src/process.py --datapath YOUR_PATH/data
```
`--datapath` is where your extracted data locate. Several arguments are allowed to configure the processing when running the script [src/process.py](https://github.com/lipengroboticsx/H2TC_code/blob/main/src/process.py). The available arguments are: [tbd: add arguments ]

|  Arguments   | Meanings  | Defaults |
|  :----     | :----  | :----  |
| takes     | ID of recordings to be processed. Set 'None' to process all recordings in the 'data' directory. This can be given with a single integer number for one take or a range linked by '-', e.g., '10-12' for recordings [000010, 000011, 000012]. | None |
| fps_event | FPS for decoding event data into frames. | 60 |
| fps_zed   | FPS for decoding ZED RGBD frames. This should equal to the value used for recording. | 60 |
| duration   | The duration of recording in seconds. | 5 |
| tolerance   | The tolerance of frame dropping in the percentage for all devices. | 0.1 |
| depth_img_format   | Image format of the exported RGB-D frames for ZED data. Either 'png' or 'jpg'.  | png |
| xypt   | Set true to export event stream in `.xypt` format which is the raw format of the events. | False |
| npy   | Set true to export depth stream in `.npy` format which is 3-dimensional numpy arrary holding the unnormalized depth estimation of each frame.  | False |
| depth_accuracy   | Float precision forThis document explains how we coordinate those multi-modal, cross-device data and how we do time alignment among them. the unnormalized depth maps. The unnormalized depth maps are not exported by default until the 'npy' are set to true. Either 'float32' or 'float64'. | float32 |
| datapath   | The raw data directory of recordings. Users need to specify it.   | None |

You can change the default setting by adding specific arguments into your command. For example, we don't export depth.npy and event xypt.csv by default (`xypt` and `npy` are `False` by default), as they are very time/space-consuming. If you need them, you can add '--npy' and '--xypt' to the command: 
```python
python src/process.py --datapath YOUR_PATH/data --npy --xypt
```
 <!-- `--npy` enables the output of 3-dimensional numpy arrary holding the unnormalized depth estimation of each frame, `--xypt` enables the output of event streams in the format of (x, y, p, t), which is the raw format of Contrast Detector events. For more customized usage, please check [Customized Processing](#•-customized-processing).  -->

<!-- This will produce all data specified in <u>TODO (link to file)</u> including particularly the events in the format of (x, y, p, t) and the real (unnormalized) depth maps. `--xypt` enables the output of event streams in the format of (x, y, p, t), which is the raw format of Contrast Detector events.`--depth_accuracy` specifies the float precision for the unnormalized depth maps. By specifying this parameter, the output of unnormalized depth maps is enabled, otherwise, disabled. In general, these two formats are used as the **input data for learning**. For the detailed explanation about these formats, please check the `/doc/data_file_explanation.md`. There are other parameters available to configure the processing. Please check the code or running the command `python src/process.py -h` for more detail.  -->
<!-- (<small>Note that the generation of unnormalized depth maps and the event streams in xypt format can be very time/space-consuming. Therefore, you could streamline the processing by disabling the output of the above two.</small> ) -->
 <!-- to produce only a minimum set of data required for annotation. By default, event streams are integrated over a fixed span of time into RGB frames, and depth maps are normalized over the pixels, for **visualization**. The command for this is 
```python
python src/process.py
``` -->

<!-- It will process all available raw data in `YOUR_PATH/data`.  -->
Once the process is done, the raw data files will be moved into a new directory `YOUR_PATH/data/raw/`, while processed data files will be stored in another new directory `YOUR_PATH/data/processed/`. The complete data hierarchy and a detailed explanation of each file are provided in [`/doc/data_file_explanation.md`](https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/data_file_explanation.md/#data). 

```
YOUR_PATH
└──data
    └──011998
        ├──raw              - all raw data
        └──processed        - all processed data
```




### &#x2022; Customized Processing 
If you want to customize the processing, please: 
  * first follow the step 1 and 2 in [How to Process](#•-how-to-process) to get the organized raw data. 
  * then in step 3, design your own processing by diving into the processing technical details in [`/doc/processing_techdetails.md`](https://github.com/lipengroboticsx/H2TC_code/blob/main/doc/processing_techdetails.md). This document explains how we process the multi-modal, cross-device data. 

### &#x2022; Trouble Shooting ❗

#### 1. Reprocess the processed take

If you want so, you have to manually remove the existing, processed, data. If you only want to reprocess the part of the whole take e.g. ZED, you don't need to remove the remaining data.

#### 2. ZED decoding frames failed

The current mechanism allows for maximally 10 failed attempts to decode (or grab in ZED term) a frame. After failed more 10 times, the decoding will abort and the processing will continue to the next part e.g. next ZED device or next stream. The frames have been decoded will be stored, while the rest frames will be ignored. This issue usually happens when decoding the last frame.

To fix this bug, one can simply reprocess the problematic takes [reprocess the processed take](#1-reprocess-the-processed-take). 
<!-- </details> -->

## Annotator
<!-- <details> -->
<!-- <summary>Details</summary> -->

In case you want to annotate your custom-captured data with annotations, 
we provide an annotation tool to label catch&throw activities with an interactive interface based on the processed data. 
To annotate, please:  
- **First** have the processed data under the directory `YOUR_PATH/data/take_id/processed`. The processed data can be obtained by processing the raw data as suggested in [Data Processing](#data-processing). 
- **Next** run the following command to launch the annotation application.  

```p
python src/annotate.py --datapath YOUR_PATH/data
```
[tbd: add arguments]
Several arguments are allowed to configure the annotation when running the script. The available arguments are:
|  Arguments   | Meanings  | Defaults |
|  :----     | :----  | :----  |
| takes     | ID of takes to be annotated. Set None to annotate all takes in the 'data' directory. This can be given with a single integer number for one take or a range linked by '-', e.g., '10-12' for takes [000010, 000011, 000012]. | None |
| review | Set true to review the takes have been already annotated before. By default (False), the annotated takes will not be displayed for annotation again. | False |
| datapath | The processed data directory of recordings. Users need to specify it.  | None |

<!-- By default, the takes that are "failed" or have been already annotated (either "finished" or "problematic") are ignored by the program so that they will not present in the annotator.  To review the annotated takes, you should run the program with the option `--review` like:

```
python src/annotate.py --review
``` -->

### Segmentation and Annotation
[tbd: add table 6 and fig 9] <br>
The annotator tool supports annotating a hierarchy of segmentation and
annotations, both semantic and dense, as detailed below: 
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
        <td><i>&#39;object\_id&#39;</i></td>
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
        <td>The subject&#39;s grasp mode to throw at the "throw" moment</td>
        <td>{<i>&#39;left&#39;</i>, <i>&#39;right&#39;</i>, <i>&#39;both&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Vertical location</td>
        <td>The vertical location(s) of the subject&#39;s  hand(s) to throw at the "throw" moment</td>
        <td>{<i>&#39;overhead&#39;</i>, <i>&#39;overhand&#39;</i>, <i>&#39;chest&#39;</i>, <i>&#39;underhand&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Horizontal location</td>
        <td>The horizontal location(s) of the subject&#39;s  hand(s) to throw at the "throw" moment</td>
        <td>{<i>&#39;left&#39;</i>, <i>&#39;middle&#39;</i>, <i>&#39;right&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Catch vertical location</td>
        <td>The vertical location(s) of the subject&#39;s hand(s) to catch at the "throw" moment</td>
        <td>{<i>&#39;overhead&#39;</i>, <i>&#39;overhand&#39;</i>, <i>&#39;chest&#39;</i>, <i>&#39;underhand&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Catch horizontal location</td>
        <td>The horizontal location(s) of the subject&#39;s hand(s) to catch at the "throw" moment</td>
        <td>{<i>&#39;left&#39;</i>, <i>&#39;middle&#39;</i>, <i>&#39;right&#39;</i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Throw location</td>
        <td>The subject&#39;s  body location to throw at the "throw" moment</td>
        <td>(<i>x</i>, <i>z</i>)</td>
        <td>automatic</td>
    </tr>
    <tr>
        <td>- Catch location</td>
        <td>The subject&#39;s  body location to catch at the "throw" moment</td>
        <td>(<i>x</i>, <i>z</i>)</td>
        <td>automatic</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td><b>Catch_touch</b></td>
        <td>The moment when the subject&#39;s hand(s) touches the flying  object during catching</td>
        <td>UNIX timestamp</td>
        <td>manual</td>
    </tr>
    <tr bgcolor="#eeeeee">
        <td>- Catch location</td>
        <td>The subject&#39;s exact location to catch at the <i>catch<i> (<i>touch<i>) moment</td>
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
        <td>The subject&#39;s grasp mode to catch at the <i>catch (stable)<i> moment</td>
        <td>{<i>&#39;left&#39;<i>, <i>&#39;right&#39;<i>, <i>&#39;both&#39;<i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Vertical location</td>
        <td>The vertical location(s) of the subject&#39;s hand(s) to catch at the <i>catch (stable)<i> moment</td>
        <td>{<i>&#39;overhead&#39;<i>, <i>&#39;overhand&#39;<i>, <i>&#39;chest&#39;<i>, <i>&#39;underhand&#39;<i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Horizontal location</td>
        <td>The horizontal location(s) of the subject&#39;s hand(s) to catch at the <i>catch (stable)<i> moment</td>
        <td>{<i>&#39;left&#39;<i>, <i>&#39;middle&#39;<i>, <i>&#39;right&#39;<i> } </td>
        <td>manual</td>
    </tr>
    <tr>
        <td>- Catch result</td>
        <td>The result on whether the object is stably catched by the subject</td>
        <td>{&#39;<i>success<i>&#39;, &#39;<i>fail<i>&#39;} </td>
        <td>manual</td>
    </tr>
</table>

Briefly, the annotations offer: 
<p style="margin-left: 2em; ">
<small>
<b>1) Interaction States for Segmentation</b>. A throw&catch activity is segmented into four phases, including pre-throwing, object flying, catching and post-catching, with three manually annotated timestamps including <u>throw</u>, <u>catch touch</u> and <u>catch stable</u>. 
</small>
</p>
<img src="https://raw.githubusercontent.com/lipengroboticsx/lipengroboticsx.github.io/lx_test/assets/images/seg_00.png" width=800>

<p style="margin-left: 2em;">
<small >
<b>2) Subjects Properties for Semantic and Dense Annotations</b>. The subjects' behaviors are manually checked and annotated with
symbolic labels in terms of <u>grasp mode</u> and <u>hand locations</u>. The subjects' <u>initial locations</u> and the <u>average flying speed of the object</u> are also automatically annotated as quantitative labels. 
</small>
</p>
<img src="https://raw.githubusercontent.com/lipengroboticsx/lipengroboticsx.github.io/lx_test/assets/images/subject_anno.png" width=400>


### &#x2022; Interface

After running the command, you can see the following interface excluding the orange bars (they are figure annotation). 
The interface consists of multi-view RGB streams (left column), multi-view depth streams (middle column), an egocentric event stream (top right sub-window), two-hand motion (middle right sub-window) and an information panel (bottom right sub-window).

<img src="https://raw.githubusercontent.com/lipengroboticsx/lipengroboticsx.github.io/main/assets/images/annotation_tool.png" alt="interface">

Inside the information panel, the annotation result is displayed in real-time. 
The representation of each text entry and the corresponding annotation name (as described [above](#segmentation-and-annotation))  are: 
[tbd: check paper for consistance]
<!-- ![](https://raw.githubusercontent.com/lipengroboticsx/lipengroboticsx.github.io/main/assets/images/info_panel_explanation.png) -->

<img src="https://raw.githubusercontent.com/lipengroboticsx/lipengroboticsx.github.io/main/assets/images/info_panel_explanation.png" width = "400" alt="info_panel">



| Number |   Representation |  Annotation Name |
|  :----:     | :----  | :----  |
|   1    |    the status of the annotation: `finished`, `unfinished` or `problematic`   |    \   |
|   2    |    which hand used to throw at the moment *throw*   |    Grasp mode  |
|   3    |    the location of the thrower at the moment *throw*   |    Throw location  |
|   4    |    the location of the catcher at the moment *throw*   | Catch location |
|   5    |    the average flying speed of the the thrown object   | Object speed|
|   6    |    the vertical hand location of the thrower at the moment *throw*   | Vertical location |
|   7    |    the horizontal hand location of the thrower at the moment *throw*   | Horizontal location |
|   8    |    the vertical hand location of the catcher at the moment *throw*   |Catch vertical location |
|   9    |    the horizontal hand location of the catcher at the moment *throw*   | Catch horizontal location |
|   10    |    the frame number and the timestamp of the moment *throw*   | \ |
|   11    |    which hand used to catch at the moment *catch (stable)*   | Grasp mode |
|   12    |    the location of the catcher at the moment *catch (touch)*   | Catch location |
|   13    |    the vertical hand location of the catcher at the moment *catch (stable)*   | Vertical location |
|   14    |    the horizontal hand location of the catcher at the moment *catch (stable)*   | Horizontal location |
|   15    |    the frame number and the timestamp of the moment *catch (touch)*  | \ |
|   16    |    the frame number and the timestamp of the moment *catch (stable)*  | \ |

<!-- 1. the status of the annotation: `finished`, `unfinished` or `problematic`
2. "grasp mode" to throw at the moment "throw"
3. the "throw location" at the moment "throw"
4. the " catch location" at the moment "throw"
5. the average flying speed " object speed" of the thrown object
6. the vertical hand location of the thrower at the moment "throw"
7. the horizontal hand location of the thrower at the moment "throw"
8. the vertical hand location of the catcher at the moment "throw"
9. the horizontal hand location of the catcher at the moment "throw"
10. the frame number and the timestamp of the moment "throw"
11. which hand used to catch at the moment "catch (stable)"
12. the location of the catcher at the moment "catch (touch)"
13. the vertical hand location of the catcher at the moment "catch (stable)"
14. the horizontal hand location of the catcher at the moment "catch (stable)"
15. the frame number and the timestamp of the moment "catch (touch)"
16. the frame number and the timestamp of the moment "catch (stable)" -->

### &#x2022; Interaction Operations

You can interact with the interface to annotate labels by the keyboard as defined below:

| Key value  |     Operation |
|:----------|:-------------|
|"right arrow"| next frame|
|"left arrow"| last frame|
|"down arrow"| next recording|
|"up arrow"| last recording|
|"return"| take the current frame as a moment of throw, catch (touch), and catch (stable) in order|
|"del"| remove the last annotated moment|
|"Q"| switch the values of info panel 2 among left, right, and both |
|"A"| switch the values of info panel 6 among overhead, overhand, chest, and underhand|
|"S"| switch the values of info panel 7 among left, middle, and right|
|"D"| switch the values of info panel 8 among overhead, overhand, chest, and underhand|
|"F"| switch the values of info panel 9 among left, middle, and right|
|"Z"| switch the values of info panel 11 among left, right, and both|
|"C"| switch the values of info panel 2 among overhead, overhand, chest, and underhand|
|"V"| switch the values of info panel 2 among left, middle, and right|
|"space"| switch the values of info panel 1 between "finished" and "unfinished"|
|"backspace"| switch the values of info panel 1 between "problematic" and "unfinished"|

<b>Any modification</b> to the annotation result will be immediately saved in the corresponding annotation file under the directory `YOUR_PATH/annotations/take_id.json`.

### &#x2022; Note ❗
<!-- [tbd: move to seg and anno] -->
<!-- #### 1. Criteria for horizontal and vertical hand positions
The vertical and horizontal hand positions are determined according to the below illustration.
<div style="display: flex; justify-content: center;">
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/annotation/workspace_00.png" width = "400" alt="missing_data_anno">
</div> -->
<!-- <div style="display: flex; justify-content: center;">Hand Locations</div> -->

<!-- #### 2. Hand position is referenced to the present human body

The horizontal and vertical hand position is referenced to the human body at the annotated moment. <br>
For example, a catcher may stand before catching and squat to catch so that the body center (chest) lowers. In this case, when annotating the vertical hand position of catcher (13 in information panel) at the moment *catch (stable)*, one should refer the hand vertical position to the chest position (lowered) when squat instead of stand. The same situation can happen, while annotating horizontal hand position, if the subject turned sideways at the annotation moment. Note that, the body pose at the moment of catching can be significantly different from the standard standing pose regarding both position and orientation. This will also affect the position and orientation of the coordinates used to determine the horizontal and vertical hand positions. For example, in the case below, the subject squatted and leaned the chest down, so that the region that is classified as "chest" is simultaneously lowered and turned down. 

![ref_catch_stand](https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/annotation/ref_catch_stand.png)

![ref_catch_squat](https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/annotation/ref_catch_squat.png) -->

#### 1. Main annotation camera

We suggest that users rely mainly on <b>the third-person (side)</b> and <b>egocentric views</b> to annotate, while the third-person back view is used as an auxiliary when significant occlusion is observed in the former two views. 

The viewing angle of cameras of egocentric are all higher than the normal height of human eyes resulting in a top-down viewing angle. This may lead to a biased observation of the vertical hand position. You can use the third-person back view to provide additional information. 
<!-- One should also pay attention to the viewing angle of cameras particularly the third-person (back) and egocentric. They are all higher than the normal height of human eyes resulting in a top-down viewing angle. This may lead to a biased observation of the vertical hand position. The same situation also applies to observing horizontal hand position since the cameras may not face to the object and the subject straight. -->

#### 2. Handling missing frames or data

It is possible that some data is missing in your labeled frame. 
Then the annotation can not be switched to the status of "finished" due to the missing data. 

For example, when you label the moment of "throw" but without OptiTrack data, the information panel appears like below. 
To handle the issue, you should seek close frames that include complete data and indicate the same event of the moment. 
If no frame is qualified, the entire take should be annotated as "problematic" and skipped to the next take.
<div style="display: flex; justify-content: center;">
<img src="https://raw.githubusercontent.com/lipengroboticsx/H2TC_code/main/doc/resources/annotation/missing_data_anno.png" width = "400" alt="missing_data_anno">
</div>
<!-- #### 4. Wrong result of catching
In some cases, the result of catching can be miss-typed during recording. 
For example, a take was labeled as "success" (should be "failed") but the catcher actually failed to catch the object. This should be corrected alongside the annotation. One should leave the take unannotated in the annotator program and manually correct this by editing the value of the "success" column in the logbook `/log.xlsx`.  Note that the annotator program will automatically filter out the takes labeled as "failed", so only the case of false "success" is possible to appear during annotation. -->
<!-- </details> -->