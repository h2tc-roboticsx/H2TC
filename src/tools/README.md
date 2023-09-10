# Auxiliary Tools

* **statistics.py**: Calculate the statistics of all takes
  * the total number of takes, the number of success/annotated/problematic
  * the frequency of each object being used
  * the drop rates of frames for each HE and OptiTrack
* **merge_log.py**:  Merge two logbooks into a new one. This is used to concatenate the annotation results from different annotators (logbooks).
* **clean.py**: Remove the data folders of the annotated takes under the data directory.
* **unzip.py**: Automatically unzip the raw data zips into the data directory.

## Statistics

First, you have to ensure that the logbook `log.xlsx`, annotation data files and the raw data files are put in the default path. This should be guaranteed if you haven't manually changed it. To calculate the statistics, you need to run the command:

```bash
python src/tools/statistics.py
```

The statistics will be generated, printed in the console and saved under the `statistics/` directory. Note that the general counting is currently only printed in the console. That includes the number of takes, success, annotated, and problematic. In addition, the following files will be produced:

* **statistics.json**: the statistics dict in `json` format. This is the raw statistics data ideal for the further process.
* **statistics.xlsx**: The statistics data stored in the spreadsheet. This is more human-friendly for direct reading.
* **objects.json**: The frequency of all objects being used. 

Alongside the normal printing in the console, the program may report, if it has, some potential issues including:

* missing annotation data: The take is specified as annotated in the logbook but without annotation data files found in the annotation directory
* unfinished annotations: The annotation data file exists in the annotation directory, but the status shows not finished.

They should be inspected for further processing.

## Merge Logbooks

First, put the old (existing) logbook and the new (to be merged with) logbook both under the project root directory (the same level as `src/`). By default, the old logbook should be named as `old.xlsx`, the new one should be named as `log.xlsx` and the output, merged, the logbook will be named as `log.xlsx` (same as the new one, so **the new logbook will be replaced**). Alternatively, you could specify other names by the arguments `--old` and `--new`. To merge two logbooks, run the command

```bash
python src/tools/merge_log.py
```

 The program only merges the values of `success`, `verified` and `annotated` in the logbooks. It sets the `success` to be failed ('0') if the values are inconsistent between two sources. It sets the rest two entries to the non-empty values in any logbook (normally there will be only one logbook with non-empty values).

## Unzip

By default, you should put the downloaded zips of raw data under the project root directory (the same level as `src/`) and then run the command:

```bash
python src/tools/unzip.py
```

Alternatively, you could specify another directory to find the raw data zips by the command below. Note that the directory must be specified in an absolute format.

```bash
python src/tools/unzip.py --zip_dir /path/to/raw/data/zips
```

## Clean

To remove the raw data of the annotated takes, run the command with the number of takes (`--num`) to remove:

```bash
python src/tools/clean.py --num int
```