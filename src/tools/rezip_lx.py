import os, sys
from subprocess import run
from argparse import ArgumentParser
import shutil

# get the absolute path of the current and root directory
curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")

# add root directory to the system path
sys.path.append(root_path)

from src.log import data_path

zip_dirs = ["/media/ur-5/My Book/ALLDATA/0-119-zip",
            # "/media/ur-5/My Book/ALLDATA/12756-12905-zip",
            # "/media/ur-5/My Book/ALLDATA/12609-12755-zip",
            # "/media/ur-5/My Book/ALLDATA/12459-12608-zip",
            # "/media/ur-5/My Book/ALLDATA/12309-12458-zip",
            # "/media/ur-5/My Book/ALLDATA/12159-12308-zip",
            # "/media/ur-5/My Book/ALLDATA/12009-12158-zip",
            # "/media/ur-5/My Book/ALLDATA/11859-12008-zip",
            # "/media/ur-5/My Book/ALLDATA/11709-11858-zip",
            # "/media/ur-5/My Book/ALLDATA/11559-11708-zip",
            # "/media/ur-5/My Book/ALLDATA/11409-11558-zip",
            # "/media/ur-5/My Book/ALLDATA/11259-11408-zip",
            # "/media/ur-5/My Book/ALLDATA/11019-11258-zip",
            # "/media/ur-5/My Book/ALLDATA/10669-11018-zip",
            # "/media/ur-5/My Book/ALLDATA/10269-10668-zip",
            # "/media/ur-5/My Book/ALLDATA/10009-10268-zip",
]

tar_dir = "/media/ur-5/My Book/ALLDATA"
# argparser = ArgumentParser()
# argparser.add_argument("--zip_dir", default=root_path,
#                        help="the full directory where the raw data zips are stored. By default, it looks up the project root directory for raw data zips.")
# argparser.add_argument("--tar_dir", default="/media/ur-5/My Book/data_1w")

# args = argparser.parse_args()
######################################################################

existed_takes = []
failed_takes = []

# path to unzip data in
unzip_path = tar_dir

for zip_dir in zip_dirs:
    walk = os.walk(zip_dir)

    # for file_name in os.listdir(args.zip_dir):
    for path, dir_list, file_list in walk:
        for file_name in file_list:
            # parse the take ID from the file name
            take_id = file_name[:-4]
            # path to raw data zip
            zip_path = os.path.join(path, file_name)
            
            unzip_folder = os.path.join(path, take_id)
            if os.path.exists(unzip_folder): # the take folder already exists in the data directory
                existed_takes.append(take_id)
            else: # the take folder not exists before
                # unzip the raw data zip by running 'unzip '
                proc = run(['unzip', zip_path,'-d',path])
                if proc.returncode != 0: # unzipping succeeds
                    failed_takes.append(take_id)
                temp_folder = os.path.join(path,"data",take_id)
                shutil.move(temp_folder, path)
                shutil.rmtree(os.path.join(path,"data"))
                # zip again
                cwd = os.getcwd()
                os.chdir(path)
                shutil.make_archive(take_id, 'zip', path, base_dir=take_id)
                os.chdir(cwd)
                # run(['zip', '-r', take_id+'.zip',unzip_folder])
                
                shutil.rmtree(unzip_folder)
                
                
                

# report the takes that already exists in the data directory
if len(existed_takes) != 0:
    print("!Existed_Takes:", existed_takes)

# report the takes that fail to be unzipped
if len(failed_takes) != 0:
    print("!!Failed_Takes:", failed_takes)
