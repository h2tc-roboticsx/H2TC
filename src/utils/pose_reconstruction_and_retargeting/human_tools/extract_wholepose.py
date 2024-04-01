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
from utils.utils import extract_opticdata
from utils.util_rot import *
from mano_tools.extract_hands_mano import extract_hands_manopose
import platform
import pickle
import joblib

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
    result_filepath = osp.join(take_folder + "/processed/rgbd0", "tcmr_output.pkl")
    meshes_dir = osp.join(take_folder, "%s_out_body_meshes_tcmr" % basename )
    
    # load data and recover the body pose 
    smpl_para = joblib.load(open(result_filepath,'rb'))
    # smpl_para = np.load(result_filepath, allow_pickle=True)
    person_ids = smpl_para.keys()
    # get the left person
    LEFT_PERSON_ID = 0
    left_x = 1e10
    for p_id in list(person_ids):
        bboxes = smpl_para[p_id]['bboxes']
        x_mean = np.mean(bboxes[:,0])
        if x_mean<left_x:
            left_x = x_mean
            LEFT_PERSON_ID = p_id
    
    smpl_poses = torch.from_numpy(smpl_para[LEFT_PERSON_ID]['pose']).to(device) # n*72
    smpl_poses
    n = smpl_poses.shape[0] 
    betas = torch.from_numpy(smpl_para[LEFT_PERSON_ID]['betas']).view(n,-1).to(device)
  
    # # Setup the SMPL model
    # smplh_model = SMPLH(constants.SMPLH_MODEL_PATH, gender='male',batch_size = n).to(device)

    # # skinning 
    # # global_orient = torch.zeros_like(pred_rotmat[:, [0]]) # lx:set global to zero
    # # global_orient[:,:,...] = torch.eye(3)
    # pred_output = smplh_model(betas=betas,
    #                             body_pose=smpl_poses[:, 1:],
    #                             global_orient=smpl_poses[:, [0]],
    #                         )
    # pred_vertices = pred_output.vertices
    # pred_vertices = pred_vertices.cpu().numpy()
    # faces=smplh_model.faces
    
    # for i in range(n):
    #     verts = pred_vertices[i]
    #     mesh = trimesh.Trimesh(verts, faces)
    #     mesh.export('test.obj')
    
    # load hand pose 
    r_mano_pose, l_mano_pose, s_frame, e_frame = extract_hands_manopose(osp.dirname(take_folder), basename, stage, save_results=False)
    r_mano_pose, l_mano_pose = r_mano_pose.to(device), l_mano_pose.to(device)

    smpl_poses = smpl_poses[s_frame:e_frame+1,:]
    betas = betas[s_frame:e_frame+1,:]
    
    # load head pose
    # head_pose = extract_opticdata(take_folder+"/processed","sub1_head_motion",s_frame,e_frame,dim_feat=7)
    # transl = head_pose[:,:3]
    # transl = torch.from_numpy(transl).to(device).to(torch.float32)
    # head_rota_qua = torch.from_numpy(head_pose[:,3:])
    # # coordinate transformation
    # head_mat = quaternion_to_matrix(head_rota_qua).to(torch.float32).to(device)
    # mat_smpl2optic = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]]).to(torch.float32).to(device) # rotate 90 degree around y-axis
    # mat_optic2smpl = torch.inverse(mat_smpl2optic)
    # head_mat_smpl = head_mat
    # head_mat_smpl[:] = mat_smpl2optic
    # # head_mat_smpl = head_mat@mat_smpl2optic
    # head_rota_aa = matrix_to_axis_angle(head_mat_smpl).to(device).to(torch.float32)

    # load right wrist pose
    wrist_pose = extract_opticdata(take_folder+"/processed","sub1_right_hand_motion",s_frame,e_frame,dim_feat=7)
    wrist_qua = torch.from_numpy(wrist_pose[:,3:]).to(torch.float32).to(device)
    wrist_mat = quaternion_to_matrix(wrist_qua)
    rh_rot = torch.tensor([[0,0,-1],[0,0,0],[1,0,0]]).to(torch.float32).to('cuda') 
    # coordinate transformation
    wrist_mat_smpl = wrist_mat@rh_rot
    # wrist_mat_smpl = wrist_mat
    # to axis-angle
    r_wrist_aa = matrix_to_axis_angle(wrist_mat_smpl) 
    
    # load left wrist pose
    wrist_pose = extract_opticdata(take_folder+"/processed","sub1_left_hand_motion",s_frame,e_frame,dim_feat=7)
    wrist_qua = torch.from_numpy(wrist_pose[:,3:]).to(torch.float32).to(device)
    wrist_mat = quaternion_to_matrix(wrist_qua)
    lh_rot = torch.tensor([[0,0,1],[0,0,0],[-1,0,0]]).to(torch.float32).to('cuda') 
    # coordinate transformation
    wrist_mat_smpl = wrist_mat@lh_rot
    # wrist_mat_smpl = wrist_mat
    # to axis-angle
    l_wrist_aa = matrix_to_axis_angle(wrist_mat_smpl) 

    smpl_poses[:,-9:-6] = r_wrist_aa # smpl id:21
    smpl_poses[:,-12:-9] = l_wrist_aa # smpl id:20

    # Setup the smplh model
    smplh_path = "./human_tools/smplh_humor/male/smplh_male.npz"
    smplh_model = SMPLH(smplh_path,\
        # constants.SMPLH_MODEL_PATH, \
        gender='male',\
            batch_size = betas.shape[0],\
                flat_hand_mean=True,\
                 use_pca=False, 
                ).to(device)

    # run smplh deformation
    with torch.no_grad():
        pred_output = smplh_model(betas=betas,
                                    body_pose=smpl_poses[:, 3:-6],
                                    # global_orient=smpl_poses[:, 0:3], #b*3 from ctmr
                                    # global_orient=head_rota_aa, #b*3 from optic
                                    # transl = transl, # b*3
                                    right_hand_pose = r_mano_pose,
                                    left_hand_pose = l_mano_pose)
    pred_vertices = pred_output.vertices
    pred_vertices = pred_vertices.cpu().numpy()
       
    if save_meshes:
        print(f"save meshes to \"{meshes_dir}\"")
        os.makedirs(meshes_dir, exist_ok=True)
        
        faces=smplh_model.faces
        n = len(pred_vertices)
        id = s_frame
        for ii in range(n):
            verts0 = np.array(pred_vertices[ii])
            mesh0 = trimesh.Trimesh(verts0, faces)
                
            # save mesh0
            fram_name =  str(s_frame + ii)
            filename =  "mesh_tcmr_body_%s.obj" % (fram_name)                                                            
            out_mesh_path = osp.join(meshes_dir, filename)
            mesh0.export(out_mesh_path)
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
    
    # == human detection model  ==
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       
    folders = os.listdir(root_folder)
    erro_info = {}
    with open("err_info/error_extract_wholepose_%s.txt"%time.time(),"w") as f_err:
        for f in folders:
            
            if f != '002740':
                continue
            
            take_folder = root_folder + '/' + f 
            print("processing: %s---------------------------" % take_folder)
            
            
            try:
                main(take_folder, device)
            except Exception as e:
                erro_info[f] = str(e) + '\n'
                print("ERROR in [%s] : %s " % (f, erro_info[f]))
                f_err.write("[%s] : %s " % (f, erro_info[f]) )
                
    print(erro_info)
        
            
    



