We have developed three tools to [record](#recorder), [process](#data-processing) the raw data, and [annotate](#annotator) a throw-catch activity. All source code has been published on GitHub. ***<u>TODO github link</u>***

## File Structure

For the detail of each file and the content of each directory, please refer to the `/doc/data_structure_full.md` file and the README.md under each directory.

* ***src/***: source code
* ***dev/***: dockerfile and PTP configuration
* ***register/***: subjects and objects data
* ***data/***: raw and processed data
  * ***xxxxxx/***: take folder named by the take id e.g. 000000
    * [***raw/***](#raw-data): raw data directly exported by each recording device
    * [***processed/***](#processed-data): formatted data derived from raw data
* ***annotations/***: annotation result files
  * [**xxxxxx.json**](#annotation): annotation result for the take id e.g. 000000
* ***statistics/***: the statistics data of all takes. This folder is generated after running the `src/tools/statistics.py`.
* ***website***: the source code of the website
* ***doc/***: the technical documentations.
* **log.xlsx**: logbook with the recording parameters of all takes
* **requirements.txt**: python dependencies

## Dependencies

To run our code, some dependencies have to be installed. 

 ### System environment

First, the default, and well-tested, system environment is

* Ubuntu: 20.04
* Python: 3.8.13
* CUDA: 11.6
* Nvidia driver: 510

We have not tested our code on other development environments, so you are recommended to configure the same, or at least similar, environment for the best experience.

### Softwares

Apart from them, there are two more applications you have to check if you have in order to run the **postprocessing** script:

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

Next, you need to install [ZED SDK](https://www.stereolabs.com/docs/installation/) (3.7.6) and [Metavision SDK](https://docs.prophesee.ai/2.3.0/installation/linux.html) (2.3.0) following the official guidance in the links in order to record and process the data of ZED and Prophesee Event cameras respectively. 

For your convenience of installing the older version (3.7.6) of ZED SDK, we copied one from the official source in the ***/dev*** directory. All you need to do is downloading the SDK installer, running it and selecting the modules you want following the [guide](https://www.stereolabs.com/docs/installation/).

Metavision SDK is not packaged in an installer way so we can't share the SDK like above. You will have to follow the [guide](https://docs.prophesee.ai/2.3.0/installation/linux.html) to install. Particularly, Metavision SDK provides several optional modules, in additional to the "essential" modules, like machine learning modules for installation. Our code only uses the functionality from the `metavision-essentails `, so you are free to install those optional modules or not.

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

```p
pip install -r requirements.txt
```

### Docker

Alternatively, we also provide a ready-to-use [Docker](https://www.docker.com/) with all dependencies installed in our git repository ***<u>TODO link</u>***. To use it, Docker should have been already successfully installed.

### Test ZED and Event Cameras

now you should be able to run the following command to launch the event recorder with your Prophesee event camera connected to the computer:

```p
metavision_viewer
```

<u>***TODO: picture of running successfully***</u>

You should also be able to record using ZED by running the official [sample](https://github.com/stereolabs/zed-examples/tree/master/svo%20recording/recording/python). If you don't have a camera or don't intend to record, you could just check if `pyzed` and `metavision_core` modules can be successfully imported in your python program. If any failure, you should inspect your installation if done manually and, unfortunately, troubleshooting this is beyond the scope of this instruction.



## Recorder

Our recorder integrates the functionality of arranging the content to be recorded, recording with multiple devices, and annotating the result of the recording into one user-friendly interactive program. 

**First**, enable all recording devices and ensure each of them function smoothly. Three ZED cameras and one Prophesee event camera should be wired to the host where the recorder program is supposed to run. StretchSense MoCap Pro gloves should be wireless connected to a Windows machine with its official client software Hand Engine running. OptiTrack server can be either operated on a separate host, recommended by us, or on the same host as any of the two aforementioned ones as long as the computational resource allows and the performance will not be thus compromised. You may need to configure the firewall on each machine to allow the UDP communication among them.

**Second**, update the configuration in our OptiTrack NatNet client code and rebuild the NatNet client. You need to set the values of OptiTrack server IP address (`char* ip_address`), recorder IP address (`servaddr.sin_addr`), and recorder port (`PORT`) according to your network setting in the file `/src/natnet_client/src/example_main.cpp`. 

```p
cd natnet_client
mkdir build
cd build
cmake ..
make
```

**Third**, initialize your lists of subjects and objects in the corresponding files `/register/subjects.csv` and `/register/objects.csv` respectively. Each subject and object should lie in a new line. Please check the sample lists in our repository for detailed format.

**Next**, launch the main recorder application with the IP and Port of the local machine and of the HE application:

```
python src/recorder.py --addr IP:PORT --he_addr IP:PORT
```

and the NatNet client:

```
./src/natnet_client/build/natnet_client
```

now you should be able to see the prompt indicating that these two applications have successfully communicated with each other, if everything goes well, as shown blow 

<u>***TODO pictures of connection established.***</u>

**Last**, operate the main recorder to record following the interactive instruction. The main recorder will automatically communicate with and command Hand Engine and NatNet client to record. Nevertheless, we do recommend you to regularly check Hand Engine and NatNet client to see if bug.

<u>***TODO picture of a complete take***</u>
