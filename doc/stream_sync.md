# Timestamping and Data Synchronization

The whole recording system consists of  3 ZED RGBD cameras, 1 Prophesee event camera, 1 StretchSense data gloves, and 1 OptiTrack motion capture system. We first describe how each data stream is timestamped and then how we synchronize and align them.

## ZED RGBD

For each RGBD stream, we record the timestamp of each frame and the beginning of the recording, resulting in N+1 timestamps in total. The timestamps are initially stored in a separate file `/data/{take_id}/raw/{zed_id}.csv` with a structure as

* nanoseconds: header of the unit
* the timestamp of the beginning of recording
* the timestamp of the frame 1 (first)
* the timestamp of the frame 2
* the timestamp of the frame N

From the perspective of implementation, timestamp is retrieved by calling the [ZED API method](https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#af18a2528093f7d4e5515b96e6be989d0) `get_timestamp(sl.TIME_REFERENCE.IMAGE)`. The returned value corresponds to the time, in UNIX nanosecond, at which the entire image was **available in the PC memory**. We observed that the timestamp retrieved by this API ignores the communication time resulting in the value of timestamps earlier (smaller) than the real frame time.

To calibrate this issue, we compensate the timestamp of each frame by adding a positive constant of

```
timestamp of start recording + 1/FPS * 1e9 - timestamp of 1st frame
```

This is equivalent to set the timestamp of 1st frame to the timestamp of start recording plus the theoretical frame time (1/FPS), and keep the offset between every two consecutive frames unchanged. 

Finally, we also observed that the last timestamp in the raw timestamp file doesn't correspond to any decoded frame image, in other words, the total amount of frames is one less than the total amount of timestamps. Therefore, the last timestamp is ignored and not saved in the processed timestamp file. 

After processing, the calibrated timestamps will be saved into a separate file `/data/{take_id}/processed/{stream_id}.csv`.  

## Event

Event raw data can be exported into two alternative formats: event streams (xypt) and event frames (RGB images). Each has a slightly different timestamp result, but they are essentially based on the same timestamps. The Event raw data file timestamps each Contrast Detector (CD) event with an offset, in microseconds, to the timepoint of recording started. Unfortunately, the raw file only includes the timestamp of recording started in seconds (see [this](https://docs.prophesee.ai/stable/data_formats/file_formats/raw.html)), so we manually took one in UNIX nanoseconds in our recording API `/src/utils/event.py` and attached it in the name of the raw data file, e.g., `event_1662023682456716448.raw`, where the 19-digit number is the timestamp of recording started.

For xypt format, the timestamp of each event is calculated by adding its time offset (t) to the timestamp of recording started (initial timestamp). Note that the time offset in xypt is in microseconds, so it has to be converted to nanoseconds first before adding. The calculated timestamps are stored directly with the decoded event streams in the file `/data/{take_id}/processed/event_xypt.csv`.

```python
timestamp = initial timestamp + t * 1000
```

For event frames, timestamps are inferred, instead of taken in real time, based on the aforementioned initial timestamp and the target decoding FPS (60 by default to align with the FPS of ZED RGBD streams). It is calculated as

```
timestamp = initial timestamp + 1/FPS * 1e9 * frame number
```

, where `* 1e9` converts the time offset to nanoseconds. The generated timestamps are stored in the file `/data/{take_id}/processed/event_frames_ts.csv`.

## Hand Engine

There are two different schemes of timestamps, device and master, in the output files, `/data/{take_id}/raw/hand/P1L.csv(P1R.csv)`, of Hand Engine (HE). The device timecode is read from the internal clock of the gloves, while the master timecode is generated according to the host PC (where Hand Engine runs) clock as the frame data received by Hand Engine from the gloves. The internal clock of gloves will be periodically calibrated to the host PC clock during connected. Therefore, these two different timestamps inevitably differ for the same frame, since they are recording different timepoints using slightly different clocks. Nevertheless, we observed that the difference between them is negligible, i.e., normally no greater than 1 frame time (120 FPS by default). In practice, we adopt the device timecode as the timestamp of each frame, because the master timecode has the catastrophic issue of freezing in the first dozens of frames.

Raw timecode exported by Hand Engine is not a real timestamp, since it uses a base of 120 (same as FPS), instead of the conventional decimal system, to represent the time below a second. A typical HE timecode looks like, e.g., 171442052 is equivalent to 17 (hour), 14 (minute), 42 (second) and 052 (frame time). 052 is converted to the decimal seconds by `52 * 1/120`. To retrieve the timestamp in UNIX time, we also need the information of date, which is stored under the key `startDate` in another file `/data/{take_id}/raw/hand/P1LMeta.json`. Finally, we concatenate the date and the time together to generate a single timestamp and then convert it to UNIX nanoseconds.

## OptiTrack

OptiTrack data is received by the client software NatNet on the same host PC as ZED and Event cameras. For each frame data, NatNet measures and provides the latency from camera exposure to the reception on the local host PC. We then calculate the timestamp of camera exposure as the timestamp of the frame by the current timestamp minus the above latency.

```
timestamp = the UNIX time of receiving the frame data - latency
```

## Clock Synchronization

Only Hand Engine is hosted on a Windows machine, while the other devices are connected, or streamed, to the same Ubuntu machine and hence timestamped based on the same system clock. To align HE data with others, we synchronized the clocks of these two host machines using Precision Time Protocol (PTP). We set up the PTP server on Ubuntu using `ptpd` and the PTP client on Windows following the official [guide](https://techcommunity.microsoft.com/t5/networking-blog/windows-subsystem-for-linux-for-testing-windows-10-ptp-client/ba-p/389181) (same as the copy `/dev/time_sync/PTP_guide.docx`). Timecode is distributed from the Ubuntu host (server) to the Windows host (client). The time drift between the clocks of these hosts is, after synchronized, normally **around 0.3 milliseconds** and peaking at 3 milliseconds in some rare cases. This accuracy is less comparable to the theoretical accuracy, sub-microsecond range, of PTP. This is most likely due to the PTP client implementation issue of Windows, since we were able to achieve the few-microsecond level accuracy using only the Ubuntu PTP server and client. 

We currently align the streams directly with their timestamps without any further processing. This means that we didn't calibrate the timestamp to the same event, e.g., camera exposure, of each stream, because timestamping the recording in this low-level, fine-grained, way is beyond the capacity of the API we used. Nevertheless, the empirical maximum offset among all data streams is **no more than 1 frames at 60 FPS**, as manually evaluated during annotation.