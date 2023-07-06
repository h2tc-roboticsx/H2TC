import os
import json
import shutil
from addict import Dict
from os import walk

from align_correct_lx import outdata_path
from src.annotate import anno_dir_correct

def make_dir(dir):
    if not os.path.exists(dir): 
        os.makedirs(dir)

throw_id = []
catch_id = []
for path, dir_list, file_list in os.walk(anno_dir_correct):
    for filename in file_list:
        file = os.path.join(path, filename)
        with open(file, 'r') as f:
            jsons = json.loads(f.read())
            anno = Dict(jsons)
        take_id = filename[:-5]
        if anno.sub1_cmd.action =="throw":
            throw_id.append(take_id)
        elif anno.sub1_cmd.action =="catch":
            catch_id.append(take_id)

dataset_path =  outdata_path
throw_folder = os.path.join(dataset_path, 'throw')
catch_folder = os.path.join(dataset_path, 'catch')
make_dir(throw_folder)
make_dir(catch_folder)

# walk through folder
for id in os.listdir(dataset_path):
    if id == "throw" or id == "catch":
        continue
    old_path = os.path.join(dataset_path, id)
    if id in throw_id:
        shutil.move(old_path, throw_folder) 
    elif id in catch_id:
         shutil.move(old_path, catch_folder) 
        

# find sub1-throw, move it to sub_folder