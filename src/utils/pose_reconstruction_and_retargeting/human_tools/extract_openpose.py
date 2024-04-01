import os, sys, shutil, argparse, subprocess, time, json, glob
import platform


def main(img_dir, out_dir,  img_out=None, video_out=None,SKELETON="BODY_25"):
# def main(img_dir, out_dir,  img_out=None, video_out=None,SKELETON="COCO"):
    os.makedirs(out_dir, exist_ok=True)
    
    # openpose_path = "D:\\OneDrive\\0_MyProjects\\RoboticsX\\DynamicManipulation\\ref-CODE\\humor\\external\\openpose"
    openpose_path = "/home/ur-5/Projects/justlx/openpose"
    # run open pose
    # must change to openpose dir path to run properly
    og_cwd = os.getcwd()
    os.chdir(openpose_path)
    
    # then run openpose
    run_cmds = [openpose_path + '/build/examples/openpose/openpose.bin', \
                '--image_dir', img_dir, '--write_json', out_dir, \
                '--display', '0', '--model_pose', SKELETON, '--number_people_max', '2']
    if platform.system().lower() == 'windows':
        run_cmds = [openpose_path +'\\bin\\OpenPoseDemo.exe', \
                '--image_dir', img_dir, '--write_json', out_dir, \
                '--display', '0', '--model_pose', SKELETON, '--number_people_max', '2']
    if video_out is not None:
        run_cmds +=  ['--write_video', video_out, '--write_video_fps', '30']
    if img_out is not None:
        run_cmds += ['--write_images', img_out]
    if not (video_out is not None or img_out is not None):
        run_cmds += ['--render_pose', '0']
    print(run_cmds)
    subprocess.run(run_cmds)
    
    os.chdir(og_cwd) # change back to resume

if __name__ == '__main__':
    
    # img_dir = "F:\\fig_2"
    # out_dir = img_dir + "/openpose_out"
    # main(img_dir, out_dir, out_dir)
    
    root_folders = [
        "/media/ur-5/golden_t/data/throw", 
        "/media/ur-5/lx_assets/data/throw",
        # "H:\\catch", # black_c 
    ]
    render_image_flag = False

    for root_folder in root_folders:
        takes = os.listdir(root_folder)
        takes.sort()
        for take in takes:
            take_folder = root_folder + '/' + take
            try:
                
                # for rgbd_idx in ['rgbd1','rgbd2']:
                for rgbd_idx in ['rgbd1']: 
                    
                    rgbd_folder = take_folder + '/processed/' + rgbd_idx 
                    if not os.path.exists(rgbd_folder):
                        continue
                    
                    img_dir = rgbd_folder
                    out_dir = img_dir + "/openpose_out"
                    
                    if os.path.exists(out_dir+"/left_0280_keypoints.json"):
                        continue
                    
                    if render_image_flag==False:
                        img_out = None
                    else:
                        img_out = out_dir
                    
                    main(img_dir, out_dir, img_out)
                
            except Exception as e:
                print("Error: ",take_folder, str(e))