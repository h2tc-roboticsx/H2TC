
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