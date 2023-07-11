import os
import json
import shutil
from addict import Dict
from os import walk
import datetime

from align_correct_lx import outdata_path
from src.annotate import anno_dir_correct

src_dataset_path = "/media/ur-5/My Passport/dataset_processed"
# tar_dataset_path =  "/media/ur-5/catch-throw/data"
throw_folder = "/media/ur-5/catch-throw/data/throw"
catch_folder = "/media/ur-5/data_catch/catch"
processed_throw_folders = os.listdir(throw_folder)
processed_catch_folders = os.listdir(catch_folder)

def make_dir(dir):
    if not os.path.exists(dir): 
        os.makedirs(dir)

make_dir(throw_folder)
make_dir(catch_folder)

throw_id = []
catch_id = []
for path, dir_list, file_list in os.walk(anno_dir_correct):
    for filename in file_list:
        file = os.path.join(path, filename)
        with open(file, 'r') as f:
            jsons = json.loads(f.read())
            anno = Dict(jsons)
        take_id = filename[:-5]
        if anno.sub1_cmd.action =="throw" and take_id not in processed_throw_folders:
            throw_id.append(take_id)
        elif anno.sub1_cmd.action =="catch" and take_id not in processed_catch_folders:
            catch_id.append(take_id)


# walk through folder
failed = {}
log_handle = open("log_failed_divide_throw_catch%s.txt" % datetime.datetime.now(), "w+") 
for id in os.listdir(src_dataset_path):
    try:
        a = int(id)
    except:
        continue
    old_path = os.path.join(src_dataset_path, id)
    json_name = "%s.json" % id
    try:
        if id in throw_id:
            shutil.move(old_path, throw_folder) 
            shutil.copyfile(os.path.join(anno_dir_correct,json_name), \
                os.path.join(throw_folder,id,json_name))
            print("DONE throw : %s" % id)
        elif id in catch_id:
            shutil.move(old_path, catch_folder) 
            shutil.copyfile(os.path.join(anno_dir_correct,json_name), \
                os.path.join(catch_folder,id,json_name))
            print("DONE catch : %s" % id)
            
    except Exception as e:
        # convert exception to a string
        err_info = str(e)
        # put exception info in the 'failed' dict
        failed[id] = err_info
        print("Processing failed due to: {}".format(err_info))
        log_handle.write("TAKE [[%s]] failed due to: %s\n" % (id,err_info))
    
            
        
        

# find sub1-throw, move it to sub_folder