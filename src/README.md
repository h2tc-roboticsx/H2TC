This folder contains the source codes of our tools to record, process and annotate the dataset. 
If you want to employ the tools, please simply follow the [tutorials](https://github.com/h2tc-roboticsx/H2TC/tree/main#run-from-scratch) in detail.<br>

Briefly, below is an overview of the provided tools and their corresponding tutorials.<br>

|  Scripts   | Features | Documents |
|  :----  | :----  | :---- |
| **recorder.py**  | supports high-quality synchronized streaming, saving and visualization of human-human throw&catch activities from multi-modality sensors. |    [Recoder](https://github.com/h2tc-roboticsx/H2TC/tree/main#recorder)        |
| **process.py**  | synchronizes and processes the collected raw data into the processed data of commonly used [formats](https://github.com/h2tc-roboticsx/H2TC/blob/main/doc/data_file_explanation.md). |  [Processer](https://github.com/h2tc-roboticsx/H2TC/tree/main#data-processing)         |
| **annotate.py** | provides an interactive interface that supports users to visually validate and annotate the dataset with a variety of dense and symbolic [lables](https://github.com/h2tc-roboticsx/H2TC/tree/main#-interface).  |   [Annotator](https://github.com/h2tc-roboticsx/H2TC/tree/main#annotator)    |
| visualize.py | provides an interactive interface that allows users to browse the streams simultaneously like a video | [Visualization](https://h2tc-roboticsx.github.io/tools/#visualization) |
| **extract.py** | unzips all raw data and organizes them in a hierarchical manner that is acceptable by the processor.  |  n.a.     |
| **log.py** |  coming soon. |  n.a.     |

