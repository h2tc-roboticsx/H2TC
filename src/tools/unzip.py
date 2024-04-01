import os, sys
from subprocess import run
from argparse import ArgumentParser

# get the absolute path of the current and root directory
curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")

# add root directory to the system path
sys.path.append(root_path)

argparser = ArgumentParser()
argparser.add_argument("--zip_dir", default=root_path,
                       help="the full directory where the raw data zips are stored. By default, it looks up the project root directory for raw data zips.")
argparser.add_argument('--datapath', type=str, default="",
                       help='raw data path of all takes')

args = argparser.parse_args()

existed_takes = []
failed_takes = []

data_path = args.datapath

for file_name in os.listdir(args.zip_dir):
    # filter out the files other than .zip
    if file_name[-4:] != '.zip': continue

    # parse the take ID from the file name
    take_id = file_name[:-4]
    # path to unzip data in
    unzip_path = os.path.join(data_path, take_id)
    # path to raw data zip
    zip_path = os.path.join(args.zip_dir, file_name)
    
    if os.path.exists(unzip_path): # the take folder already exists in the data directory
        existed_takes.append(take_id)
    else: # the take folder not exists before
        # unzip the raw data zip by running 'unzip '
        # Note that the raw data is previously zipped in a directory structure like
        # data/{take_id}/, so the unzipped take folder will be automatically put in the data directory
        proc = run(['unzip', zip_path])
        if proc.returncode == 0: # unzipping succeeds
            run(['rm', zip_path]) # remove the raw data zip file
        else: # unzipping failed
            failed_takes.append(take_id)

# report the takes that already exists in the data directory
if len(existed_takes) != 0:
    print("!Existed_Takes:", existed_takes)

# report the takes that fail to be unzipped
if len(failed_takes) != 0:
    print("!!Failed_Takes:", failed_takes)
