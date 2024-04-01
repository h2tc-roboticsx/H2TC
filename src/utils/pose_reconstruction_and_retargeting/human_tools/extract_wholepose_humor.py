import torch
import os
import os.path as osp
import trimesh
import numpy as np
import smplx
from smplx import SMPL, SMPLH, SMPLX
import torchgeometry

from common import constants

import time

import sys
sys.path.append("./")
sys.path.append("./mano_tools")
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
#     __file__, __name__, str(__package__)))

# from util_rot import matrix_to_axis_angle
from utils.utils import extract_opticdata, loadbodypose_humor
from utils.util_rot import *
from mano_tools.extract_hands_mano import extract_hands_manopose
from human_tools.body_model import BodyModel
import platform

print(os.getcwd())

# ============ config ===========================
root_folder = "F:/data/throw"
if platform.system().lower() != "windows":
    root_folder = "/media/ur-5/golden_t/data/throw"
# img_dir = "D:/JustLX/data/011998/processed/rgbd0"
# save_results = True 
save_meshes = True
stage = "throw"


# ===============================================

def main(take_folder, device):
    
    # folders
    take_folder = take_folder.replace("\\","/")
    basename = take_folder.split('/')[-1]
    meshes_dir = osp.join(take_folder, "processed/rgbd0/humor_out_v3/body_meshes_humor")
    
    # load body pose 
    smpl_poses, betas, smpl_transl, root_orient = loadbodypose_humor(take_folder)
    
    # # load hand pose 
    r_mano_pose, l_mano_pose, s_frame, e_frame = extract_hands_manopose(osp.dirname(take_folder), basename, stage, save_results=False)
    r_mano_pose, l_mano_pose = r_mano_pose.to(device), l_mano_pose.to(device)

    # load smplh model 
    SMPLH_HUMOR_MODEL = "./human_tools/smplh_humor/male/smplh_male.npz"
    smplh_model = BodyModel(SMPLH_HUMOR_MODEL, \
         num_betas=16, \
        batch_size = smpl_poses.shape[0],\
            flat_hand_mean = False,\
            ).to("cuda")


    smpl_poses = torch.from_numpy(smpl_poses).to(torch.float32).to("cuda")
    root_orient = torch.from_numpy(root_orient).to(torch.float32).to("cuda")
    smpl_transl = torch.from_numpy(smpl_transl).to(torch.float32).to("cuda")
    with torch.no_grad():
        pred_output = smplh_model.bm(
                                body_pose=smpl_poses,
                                global_orient=root_orient,
                                transl=smpl_transl)
    verts = pred_output.vertices.cpu().numpy()
    faces = smplh_model.bm.faces
    joints = pred_output.joints.cpu().numpy()
    
    
    roi_joints_dict = {}
    roi_joints_dict['Right_Foot'] = joints[:,11:12,:]   
    roi_joints_dict['RWrist'] = joints[:,21:22,:]   
    roi_joints = np.concatenate((roi_joints_dict['Right_Foot'], roi_joints_dict['RWrist']), axis=1)
    np.savez(meshes_dir + '/roi_joints.npz', joints)
       
    if save_meshes:
        print(f"save meshes to \"{meshes_dir}\"")
        os.makedirs(meshes_dir, exist_ok=True)
        
        n = len(verts)
        id = 0
        for ii in range(n):
            verts0 = np.array(verts[ii])
            mesh0 = trimesh.Trimesh(verts0, faces)
            mesh_j = trimesh.Trimesh(vertices=roi_joints[ii])
            
                
            # save mesh0
            fram_name =  str(ii)
            filename =  "humor_body_%s.obj" % (fram_name)                                                            
            out_mesh_path = osp.join(meshes_dir, filename)
            out_joints_path = osp.join(meshes_dir, "joints_%d_%s.ply" % (0, fram_name))
            
            mesh0.export(out_mesh_path)
            mesh_j.export(out_joints_path)
            
            a = 1
            
"""
print("--------------------------- Visualization ---------------------------")
# make the output directory
os.makedirs(front_view_dir, exist_ok=True)
print("Front view directory:", front_view_dir)
if show_sideView:
    os.makedirs(side_view_dir, exist_ok=True)
    print("Side view directory:", side_view_dir)
if show_bbox:
    os.makedirs(bbox_dir, exist_ok=True)
    print("Bounding box directory:", bbox_dir)

pred_vert_arr = np.array(pred_vert_arr)
for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
    chosen_mask = detection_all[:, 0] == img_idx
    chosen_vert_arr = pred_vert_arr[chosen_mask]

    # setup renderer for visualization
    img_h, img_w, _ = orig_img_bgr.shape
    focal_length = estimate_focal_length(img_h, img_w)
    renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                        faces=smplh_model.faces,
                        same_mesh_color=True)
    front_view = renderer.render_front_view(chosen_vert_arr,
                                            bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())

    # save rendering results
    basename = osp.basename(img_path_list[img_idx]).split(".")[0]
    filename = basename + "_front_view_cliff_%s.jpg" % backbone
    front_view_path = osp.join(front_view_dir, filename)
    cv2.imwrite(front_view_path, front_view[:, :, ::-1])

    if show_sideView:
        side_view_img = renderer.render_side_view(chosen_vert_arr)
        filename = basename + "_side_view_cliff_%s.jpg" % backbone
        side_view_path = osp.join(side_view_dir, filename)
        cv2.imwrite(side_view_path, side_view_img[:, :, ::-1])

    # delete the renderer for preparing a new one
    renderer.delete()

    # draw the detection bounding boxes
    if show_bbox:
        chosen_detection = detection_all[chosen_mask]
        bbox_info = chosen_detection[:, 1:6]

        bbox_img_bgr = orig_img_bgr.copy()
        for min_x, min_y, max_x, max_y, conf in bbox_info:
            ul = (int(min_x), int(min_y))
            br = (int(max_x), int(max_y))
            cv2.rectangle(bbox_img_bgr, ul, br, color=(0, 255, 0), thickness=2)
            cv2.putText(bbox_img_bgr, "%.1f" % conf, ul,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 0, 255), thickness=1)
        filename = basename + "_bbox.jpg"
        bbox_path = osp.join(bbox_dir, filename)
        cv2.imwrite(bbox_path, bbox_img_bgr)

# make videos
if make_video:
    print("--------------------------- Making videos ---------------------------")
    from common.utils import images_to_video
    images_to_video(front_view_dir, video_path=front_view_dir + ".mp4", frame_rate=frame_rate)
    if show_sideView:
        images_to_video(side_view_dir, video_path=side_view_dir + ".mp4", frame_rate=frame_rate)
    if show_bbox:
        images_to_video(bbox_dir, video_path=bbox_dir + ".mp4", frame_rate=frame_rate)
"""    

              

if __name__ == "__main__":
    
    # # == human detection model  ==
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       
    # folders = os.listdir(root_folder)
    # folders.sort()
    # folders.reverse()
    
    # erro_info = {}
    # with open("err_info/error_extract_wholepose_%s.txt"%time.time(),"w") as f_err:
    #     for f in folders:
            
            
    #         take_folder = root_folder + '/' + f 
    #         print("processing: %s---------------------------" % take_folder)
            
            
    #         try:
    #             main(take_folder, device)
    #         except Exception as e:
    #             erro_info[f] = str(e) + '\n'
    #             print("ERROR in [%s] : %s " % (f, erro_info[f]))
    #             f_err.write("[%s] : %s " % (f, erro_info[f]) )
                
    # print(erro_info)
        
    # == human detection model  ==
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       
    folders = [
        # "/media/ur-5/black_c/catch/012307",
        "/media/ur-5/golden_t/data/throw/002992"
    ]
    
    erro_info = {}
    with open("err_info/error_extract_wholepose_%s.txt"%time.time(),"w") as f_err:
        for f in folders:
            
            
            take_folder = f
            print("processing: %s---------------------------" % take_folder)
            
            
            try:
                main(take_folder, device)
            except Exception as e:
                erro_info[f] = str(e) + '\n'
                print("ERROR in [%s] : %s " % (f, erro_info[f]))
                f_err.write("[%s] : %s " % (f, erro_info[f]) )
                
    print(erro_info)
            
    



