# Timestamp alignment

Although all data streams were timestamped during data collection, it is impossible for their timestamps to be exactly the same. There exists time drift in millisecond-level between data streams. Therefore, during post-processing, we use the timestamp of **RGBD0 camera** as the reference, and align the timestamps of the rest data streams to it. The resulting timestamp alignment is saved in a file called `alignment.json`.

## The alignment.json file
Alignment.json file essentially saves a dictionary whose key is the frame number and whose value is all data streams' timestamps that correspond to that frame. The following is a snapshot of an example alignment.json file:
```
{
    "0": {
        "rgbd0": 1662023682418648047,
        "rgbd1": 1662023682421582524,
        "rgbd2": 1662023682427297843,
        "event": null,
        "left_hand_pose": 1662023682433333504,
        "right_hand_pose": 1662023682424999936,
        "leopard_motion": null,
        "sub1_head_motion": 1662023682430291456,
        "sub1_right_hand_motion": 1662023682430291456,
        "sub1_left_hand_motion": 1662023682434457856,
        "sub2_head_motion": 1662023682430291456
    },
    ...
    ,
    "33": {
        "rgbd0": 1662023682968521047,
        "rgbd1": 1662023682971711524,
        "rgbd2": 1662023682960619843,
        "event": 1662023682973383168,
        "left_hand_pose": 1662023682966666752,
        "right_hand_pose": 1662023682966666752,
        "leopard_motion": null,
        "sub1_head_motion": 1662023682967801088,
        "sub1_right_hand_motion": 1662023682967801088,
        "sub1_left_hand_motion": 1662023682967801088,
        "sub2_head_motion": 1662023682967801088
    },
    "34": {
        "rgbd0": 1662023682985276047,
        "rgbd1": 1662023682988500524,
        "rgbd2": 1662023682977112843,
        "event": 1662023682990049792,
        "left_hand_pose": 1662023682983333376,
        "right_hand_pose": 1662023682983333376,
        "leopard_motion": 1662023683001162752,
        "sub1_head_motion": 1662023682984467712,
        "sub1_right_hand_motion": 1662023682984467712,
        "sub1_left_hand_motion": 1662023682984467712,
        "sub2_head_motion": 1662023682984467712
    },
    ...
}
```
If the timestamp of a data stream is missing in certain frames, its value will be `null` as shown in the above example. Such situation is rare, and it is mainly caused by 1) the Optitrack when the tracked object is occluded; or 2) by [Hand Engine](https://stretchsense.com/solution/hand-engine/) when the data transmission is congested; or 3) the open of the event camera lags slightly behind the RGBD0 camera. Therefore, the corresponding timestamp is missing.

### How to create an alignment.json file
We use the timestamps of **RGBD0 camera** as the reference. Therefore, the total number of frames saved in the `alignment.json` is equal to the number of timestamps recorded by **RGBD0 camera**. 

Given a timestamp of **RGBD0 camera** and its associated frame number, for each of other data streams, we use the binary search alogrithm to find their nearest timestamp to the timestamp of **RGBD0 camera**. This nearest timestamp is then used as the timestamp of that frame of the other data stream. Note that the difference between the nearest timestamp and its query **RGBD0 camera**'s timestamp has to be within a threshold, which is currently set to 1/60 * 10e9 nanosecond.