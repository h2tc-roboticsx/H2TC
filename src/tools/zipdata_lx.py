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

data_dirs = [
"/media/ur-5/My Book/ALLDATA/data_2360-2539",
"/media/ur-5/My Book/ALLDATA/data_2540-2739",
"/media/ur-5/My Book/ALLDATA/data_2740-2888",
"/media/ur-5/My Book/ALLDATA/data_2889-3120",
"/media/ur-5/My Book/ALLDATA/data_3539-3748",
"/media/ur-5/My Book/ALLDATA/data_3929-4138",
"/media/ur-5/My Book/ALLDATA/data_4139-4338",
"/media/ur-5/My Book/ALLDATA/data_4779-5178",
]

######################################################################

existed_takes = []
failed_takes = []

# path to unzip data in

for data_dir in data_dirs:
    # 1. get tar folder
    folder_name = data_dir.split('/')[-1][5:]
    tar_folder = os.path.join(os.path.dirname(data_dir), folder_name+'-zip')
    if not os.path.exists(tar_folder):
        os.makedirs(tar_folder, exist_ok=True)
    
    # 2. zip folder 
    walk = os.walk(data_dir+'/data')
    for path, dir_list, file_list in walk:
        for take_id in dir_list:

                cwd = os.getcwd()
                os.chdir(data_dir)
                run(['zip', '-r', tar_folder + '/' + take_id+'.zip', 'data/'+ take_id])
                os.chdir(cwd)

