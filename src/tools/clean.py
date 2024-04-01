import os, json, sys
from subprocess import run
from argparse import ArgumentParser

# get the absolute path of the current and root directory
curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")

# add root directory to the system path
sys.path.append(root_path)

from src.log import data_path
from src.annotate import anno_dir

argparser = ArgumentParser()
argparser.add_argument("--num", type=int, default=None,
                       help="the maximum number of takes to be removed. By default (None), all eligible takes will be removed.")

args = argparser.parse_args()

removed_counter = 0 # the number of takes has been removed
# sorted annotation data files under annotation directory
anno_files = sorted(os.listdir(anno_dir)) 
for anno_file_name in anno_files:
    # path to the annotation data file
    anno_file_path = os.path.join(anno_dir, anno_file_name)
    # load annotation data from the file
    with open(anno_file_path, 'r') as f:
        anno = json.loads(f.read())

    # parse take ID from the annotation data file
    take_id = anno_file_name[:-5]
    # path to the take folder under the data directory
    data_folder_path = os.path.join(data_path, take_id)

    # the annotation status is finished in annotation data file
    # and the take data folder exists under the data directory
    if anno['status'] == 1 and os.path.exists(data_folder_path):
        print("{} removed".format(take_id))
        # remove the entire take folder by 'rm -rf'
        run(['rm', '-rf', data_folder_path])
        removed_counter += 1

    # terminate the processing if reaching the specified number of removes
    if args.num is not None and removed_counter == args.num:
        break

