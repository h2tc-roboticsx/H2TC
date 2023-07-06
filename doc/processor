## Data Processing

Our data processing script converts the raw data to the format as described in [Processed Data](https://lipengroboticsx.github.io/dataset/). To run the script, you have to **first** put the raw data of each recording into an individual folder named by the recording ID under the directory `/data` and organize the data from different sensors as displayed in [Raw Data](https://lipengroboticsx.github.io/dataset/). This sorting can be much effortless if the raw data is recorded using our recorder program since they will be produced in a way ready to be processed. For example, the raw data of the recording "011998" should be organized in a way as below:

* ***data/***
  * ***011998/***
    * **{ZED-ID}.svo**: raw data of ZED camera with the ID
    * **{ZED-ID}.csv**: timestamps of the raw data of ZED camera with the ID
    * **event_{timestamp}.raw**: raw data of Event camera
    * **optitrack.csv**: raw data of optitrack
    * ***hand/***
      * **P1L.csv / P1R.csv**: raw data of left / right hand pose for Hand Engine
      * **P1LMeta.json / P1RMeta.json**: metadata of recording for Hand Engine

This is the minimum set of raw data files required for processing. It is smaller than the real set of raw files exported from each sensor, because some are not used in processing. For a full set of raw files and the detailed explanation of each file, please refer to the post `/doc/data_structure_full.md`.

**Second**, once the data organized appropriately, all you need to do is just running the following command:

```python
python src/postprocess.py --xypt --depth_accuracy float32
```

This will produce all data specified in <u>TODO (link to file)</u> including particularly the events in the format of (x, y, p, t) and the real (unnormalized) depth maps. `--xypt` enables the output of event streams in the format of (x, y, p, t), which is the raw format of Contrast Detector events.`--depth_accuracy` specifies the float precision for the unnormalized depth maps. By specifying this parameter, the output of unnormalized depth maps is enabled, otherwise, disabled. In general, these two formats are used as the **input data for learning**. For the detailed explanation about these formats, please check the `/doc/data_structure_full.md`. There are other parameters available to configure the processing. Please check the code or running the command `python src/postprocess.py -h` for more detail.

Note that the generation of unnormalized depth maps and the event streams in xypt format can be very time/space-consuming. Therefore, you could streamline the processing by disabling the output of the above two to produce only a minimum set of data required for annotation. By default, event streams are integrated over a fixed span of time into RGB frames, and depth maps are normalized over the pixels, for **visualization**. The command for this is

```python
python src/postprocess.py
```

After the processing finished, the raw and processed data files will be separately stored in their own directory like below. All raw data will be transferred (copied) into a new directory `raw/`. The processed data is stored in `processed/`.

* ***data/***
  * ***011998/***
    * ***raw/***: all raw data
    * ***processed/***: all processed data

For the technical detail of how we process the data, please refer to `/doc/postprocessing.md`.

### trouble shooting

#### 1. reprocess the processed take

If you want so, you have to manually remove the existing, processed, data. If you only want to reprocess the part of the whole take e.g. ZED, you don't need to remove the remaining data.

#### 2. ZED decoding frames failed

The current mechanism allows for maximally 10 failed attempts to decode (or grab in ZED term) a frame. After failed more 10 times, the decoding will abort and the processing will continue to the next part e.g. next ZED device or next stream. The frames have been decoded will be stored, while the rest frames will be ignored. This issue usually happens when decoding the last frame.

To fix this bug, one can simply reprocess the problematic takes following the [instruction](#reprocess-the-processed-take).