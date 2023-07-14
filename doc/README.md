* **data_structure_concise.md**: an overview of the entire file organization and the data structure of the raw and processed data.
* **data_structure_full.md**: full detail of the file organization and data structure.
* **processing_techdetails.md**: the technical detail of how raw data is processed.
* **stream_sync.md**: the mechanism of timestamp generation and synchronizing different streams.
* **hardware_setup.md**: the instruction of setting up hardware for recording
* **alignment.md**: the explanation of timestamp alignment, which uses RGBD0 camera's timestamp as reference.
* **hand_pose.md**: the explanation of reconstructing the left and right hand poses, including each hand's coordinate system, bone length, etc.
* **optitrack.md**: the explanation of optitrack-related processing.
* **object_coordinate_system.docx**: the document that illustrates the coordinate system of each 3D printed object when opened in SolidWorks. These are 15 objects attached with markers to be tracked by Optitrack. When creating rigid body of each object in Motive (Optitrack's software), we ensured the coordinate system of the rigid body aligns with the one in SolidWorks.
* **subject_height_arm_length.xlsx**: the measured height and arm length of the subjects (not complete yet).
* **annotation_progress.md**: the progress summary of annotation.
* **anno_to_correct.xlsx**: the list of take IDs whose annotation should be corrected with calibrated ZED timestamps. 