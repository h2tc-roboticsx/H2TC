import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time, math, shutil, json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.logging import Logger, mkdir
from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues

from motion_modeling.h2tc_dataset import CONTACT_INDS

from human_tools.body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose

from pose_fitting.fitting_utils import OP_IGNORE_JOINTS, parse_floor_plane, compute_cam2prior, OP_EDGE_LIST, log_cur_stats
from pose_fitting.fitting_loss import FittingLoss

from pose_fitting.fitting_utils import *
import trimesh
from pathlib import Path


LINE_SEARCH = 'strong_wolfe'
J_BODY = len(SMPL_JOINTS)-1 # no root

CONTACT_THRESH = 0.5

class MotionOptimizer():
    ''' Fits SMPL shape and motion to observation sequence '''

    def __init__(self, device,
                       body_model, # SMPL model to use (its batch_size should be B*T)
                       num_betas, # beta size in SMPL model
                       batch_size, # number of sequences to optimize
                       seq_len, # length of the sequences
                       observed_modalities, # list of the kinds of observations to use
                       loss_weights, # dict of weights for each loss term
                       pose_prior, # VPoser model
                       motion_prior=None, # humor model
                       init_motion_prior=None, # dict of GMM params to use for prior on initial motion state
                       optim_floor=False, # if true, optimize the floor plane along with body motion (need 2d observations)
                       camera_matrix=None, # camera intrinsics to use for reprojection if applicable
                       robust_loss_type='none',
                       robust_tuning_const=4.6851,
                       joint2d_sigma=100,
                       stage3_tune_init_state=True,
                       stage3_tune_init_num_frames=15,
                       stage3_tune_init_freeze_start=30,
                       stage3_tune_init_freeze_end=50,
                       stage3_contact_refine_only=False,
                       use_chamfer=False,
                       im_dim=(1080,1080),# image dimensions to use for visualization
                       args=None): 
        B, T = batch_size, seq_len
        self.device = device
        self.batch_size = B
        self.seq_len = T
        self.body_model = body_model
        self.num_betas = num_betas
        self.optim_floor = optim_floor
        self.stage3_tune_init_state = stage3_tune_init_state
        self.stage3_tune_init_num_frames = stage3_tune_init_num_frames
        self.stage3_tune_init_freeze_start = stage3_tune_init_freeze_start
        self.stage3_tune_init_freeze_end = stage3_tune_init_freeze_end
        self.stage3_contact_refine_only = stage3_contact_refine_only
        self.im_dim = im_dim
        self.args = args
        self.take_id = int(self.args.data_path.replace('\\', '/').split('/')[-3])

        #
        # create the optimization variables
        #

        # number of states to explicitly optimize for
        # For first stages this will always be the full sequence
        num_state_steps = T
        # latent body pose
        self.pose_prior = pose_prior
        self.latent_pose_dim = self.pose_prior.latentD
        self.latent_pose = torch.zeros((B, num_state_steps, self.latent_pose_dim)).to(device)
        # root (global) transformation
        self.body_pose = torch.zeros((B, num_state_steps, 63)).to(device)
        self.trans = torch.zeros((B, num_state_steps, 3)).to(device)
        self.root_orient = torch.zeros((B, num_state_steps, 3)).to(device) # aa parameterization
        self.root_orient[:,:,0] = np.pi
        # body shape
        self.betas = torch.zeros((B, num_betas)).to(device) # same shape for all steps
        self.left_hand_pose, self.right_hand_pose = None, None

        self.motion_prior = motion_prior
        self.init_motion_prior = init_motion_prior
        self.latent_motion = None
        if self.motion_prior is not None:
            # need latent dynamics sequence as well
            self.latent_motion_dim = self.motion_prior.latent_size
            self.cond_prior = self.motion_prior.use_conditional_prior
            # additional optimization params to set later
            self.trans_vel = None
            self.root_orient_vel = None
            self.joints_vel = None
        # else:
        #     Logger.log('Need the motion prior to use all-implicit parameterization!')
        #     exit()

        self.init_fidx = np.zeros((B)) # the frame chosen to use for the initial state (first frame by default)

        self.cam_f = self.cam_center = None
        if self.optim_floor:
            if camera_matrix is None:
                Logger.log('Must have camera intrinsics (camera_matrix) to optimize the floor plane!')
                exit()
            # NOTE: we assume a static camera, so we optimize the params of the floor plane instead of camera extrinsics
            self.floor_plane = torch.zeros((B, 3)).to(device) # normal vector (a, b, c) scaled by offset (d)
            self.floor_plane[:,2] = 1.0 # up axis initially
            # will not be optimized, extra auxiliary variables which are determined from the floor plane and root orient pose
            #       we only have one transformation for the chosen "key" frame in the sequence
            # self.cam2prior_R = torch.eye(3).reshape((1, 3, 3)).expand((B, 3, 3)).to(device)
            # self.cam2prior_t = torch.zeros((B, 3)).to(device)
            self.cam2prior_R = torch.eye(3).to(device)
            self.cam2prior_t = torch.zeros((3,1)).to(device)
            self.cam2prior_root_height = torch.zeros((B, 1)).to(device)
            
            cam_fx = camera_matrix[:, 0, 0]
            cam_fy = camera_matrix[:, 1, 1]
            cam_cx = camera_matrix[:, 0, 2]
            cam_cy = camera_matrix[:, 1, 2]
            # focal length and center are same for all timesteps
            self.cam_f = torch.stack([cam_fx, cam_fy], dim=1)
            self.cam_center = torch.stack([cam_cx, cam_cy], dim=1)
        self.use_camera = self.cam_f is not None and self.cam_center is not None

        #
        # create the loss function
        #
        self.smpl2op_map = smpl_to_openpose(body_model.model_type, use_hands=False, use_face=False, use_face_contour=False, openpose_format='coco25')
        self.fitting_loss = FittingLoss(loss_weights, 
                                        self.init_motion_prior,
                                        self.smpl2op_map,
                                        OP_IGNORE_JOINTS,
                                        self.cam_f,
                                        self.cam_center,
                                        robust_loss_type,
                                        robust_tuning_const,
                                        joints2d_sigma=joint2d_sigma,
                                        use_chamfer=use_chamfer,
                                        args = args).to(device)
        
        
     
    
    # load previous optimizing results
    def load_optresult(self, opt_path):
        smpl_para = np.load(opt_path)
        smpl_poses = smpl_para['pose_body'] # seq_len*72
        smpl_poses = torch.from_numpy(smpl_poses).unsqueeze(0).to('cuda')
        self.body_pose = smpl_poses
        self.betas[0,:] = torch.from_numpy(smpl_para['betas']) # 16
        self.trans[0,:,:] = torch.from_numpy(smpl_para['trans']) # seq_len*3
        self.root_orient[0,:,:] = torch.from_numpy(smpl_para['root_orient'])
        self.latent_pose = self.pose2latent(smpl_poses).detach()


    def initialize(self, observed_data):
        
        if self.optim_floor:
            # initialize the floor
            # assumes observed floor is (a, b, c, d) where (a, b, c) is the normal and (d) the offset
            floor_normals = observed_data['floor_plane'][:,:3]
            floor_offsets = observed_data['floor_plane'][:,3:]
            self.floor_plane = floor_normals * floor_offsets
            self.floor_plane = self.floor_plane.to(torch.float).clone().detach()
            self.floor_plane.requires_grad = True

            # optimizing from 2D data, must initialize cam/body trans
            if 'points3d' in observed_data:
                # initialize with mean of point cloud
                point_seq = observed_data['points3d'] # B x T x N x 3
                self.trans = torch.mean(point_seq, dim=2).clone().detach()
            elif 'joints2d' in observed_data:
                # only have RGB data to use
                # use focal length and bone lengths to approximate
                # (based on PROX https://github.com/mohamedhassanmus/prox/blob/master/prox/fitting.py)

                # get 3D joints mapped to OpenPose
                body_pose = self.latent2pose(self.latent_pose)
                pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
                joints3d_full = torch.cat([pred_data['joints3d'], pred_data['joints3d_extra']], dim=2)
                joints3d_op = joints3d_full[:,:,self.smpl2op_map,:]
                # openpose observations
                joints2d_obs = observed_data['joints2d'][:,:,:,:2]
                joints2d_conf = observed_data['joints2d'][:,:,:,2]

                # find least-occluded 2d frame
                num_2d_vis = torch.sum(joints2d_conf > 0.0, dim=2)
                best_2d_idx = torch.max(num_2d_vis, dim=1)[1]

                # calculate bone lengths and confidence in each bone length
                bone3d = []
                bone2d = []
                conf2d = []
                for pair in OP_EDGE_LIST:
                    diff3d = torch.norm(joints3d_op[:,0,pair[0],:] - joints3d_op[:,0,pair[1],:], dim=1) # does not change over time
                    diff2d = torch.norm(joints2d_obs[:,:,pair[0],:] - joints2d_obs[:,:,pair[1],:], dim=2)
                    minconf2d = torch.min(joints2d_conf[:,:,pair[0]], joints2d_conf[:,:,pair[1]])
                    bone3d.append(diff3d)
                    bone2d.append(diff2d)
                    conf2d.append(minconf2d)

                bone3d = torch.stack(bone3d, dim=1)
                bone2d = torch.stack(bone2d, dim=2)
                bone2d = bone2d[np.arange(self.batch_size), best_2d_idx, :]
                conf2d = torch.stack(conf2d, dim=2)
                conf2d = conf2d[np.arange(self.batch_size), best_2d_idx, :]

                # mean over all
                mean_bone3d = torch.mean(bone3d, dim=1)
                mean_bone2d = torch.mean(bone2d*(conf2d > 0.0), dim=1)

                # approx z based on ratio
                init_z = self.cam_f[:,0] * (mean_bone3d / mean_bone2d)
                self.trans[:,:,2] = init_z.unsqueeze(1).expand((self.batch_size, self.seq_len)).detach()


    def run(self, observed_data,
                  data_fps=30,
                  lr=1.0,
                  num_iter=[30, 70, 70],
                  lbfgs_max_iter=20,
                  stages_res_out=None,
                  fit_gender='neutral'):

        # load hand pose 
        self.left_hand_pose, self.right_hand_pose = observed_data['lhand_pose'][:,:self.seq_len,:], observed_data['rhand_pose'][:,:self.seq_len,:]
        
        ## load camera extrinsic
        data_folder = str(Path(self.args.data_path).parent.parent)
        extri_file = osp.join(data_folder,'CamExtr.txt')
        if osp.exists(extri_file) and (not self.args.cal_cam):
            cam_RT = torch.from_numpy(np.loadtxt(extri_file)).float()
        else:
            assert osp.exists(extri_file)
            
        self.fitting_loss.cam_R = cam_RT[:3,:3]
        self.fitting_loss.cam_t = cam_RT[:3,3]
        cam_RT = cam_RT.repeat(observed_data['rhand'].shape[1],1,1).to('cuda') # [seq_num, 4, 4]
        
        if len(num_iter) != 3:
            print('Must have num iters for 3 stages! But %d stages were given!' % (len(num_iter)))
            exit()

        per_stage_outputs = {} # SMPL results after each stage

        #
        # Initialization using mm pose
        #
        body_pose = observed_data['mmpose']
        self.latent_pose = self.pose2latent(body_pose).detach()
        
        flag_rhand = (observed_data['rhand'][0,:,0]!=0).squeeze()
        flag_lhand = (observed_data['lhand'][0,:,0]!=0).squeeze()

        #
        # Stage I: Only global root and orientation
        #
        Logger.log('Optimizing stage 1 - global root translation and orientation for %d interations...' % (num_iter[0]))
        cur_res_out_path = os.path.join(stages_res_out[0], 'stage1_results.npz')
        if os.path.exists(cur_res_out_path):
            print('Loading stage 1 fitting results in %s' % cur_res_out_path)
            bdata = np.load(cur_res_out_path)
            self.trans = torch.from_numpy(bdata['trans']).unsqueeze(0).to(self.device)
            self.root_orient = torch.from_numpy(bdata['root_orient']).unsqueeze(0).to(self.device)
            self.betas = torch.from_numpy(bdata['betas']).unsqueeze(0).to(self.device)
            body_pose = torch.from_numpy(bdata['pose_body']).unsqueeze(0).to(self.device)
            pose_hand = torch.from_numpy(bdata['pose_hand']).unsqueeze(0).to(self.device)
            self.latent_pose = self.pose2latent(body_pose).detach()
            stage1_pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
            per_stage_outputs['stage1'] = stage1_pred_data
        else:
            self.fitting_loss.set_stage(0)
            self.trans.requires_grad = True
            self.root_orient.requires_grad = True
            self.betas.requires_grad = False
            self.latent_pose.requires_grad = False

            root_opt_params = [self.trans, self.root_orient]

            root_optim = torch.optim.LBFGS(root_opt_params,
                                            max_iter=lbfgs_max_iter,
                                            lr=lr,
                                            line_search_fn=LINE_SEARCH)
            
            for i in range(num_iter[0]): 
                # Logger.log('ITER: %d' % (i))
                self.fitting_loss.cur_optim_step = i 
                stats_dict = None 
                def closure(): 
                    root_optim.zero_grad() 
                    
                    pred_data = dict() 
                    # Use current params to go through SMPL and get joints3d, verts3d, points3d 
                    body_pose = self.latent2pose(self.latent_pose) 
                    pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas) 
                    # compute data losses only 
                    loss, stats_dict = self.fitting_loss.root_fit(observed_data, pred_data) 
                    
                    # lx: facing constrain 
                    j_0 = pred_data["joints3d"][0,:,0]
                    j_13 = pred_data["joints3d"][0,:,13]
                    j_14 = pred_data["joints3d"][0,:,14]
                    j_o_13 = j_13 - j_0
                    j_13_14 = j_14 - j_13 
                    facing = torch.cross(j_o_13, j_13_14)
                    if self.args.is_sub1 == "sub1":
                        loss_facing = torch.sum(torch.relu(-facing[:,2])) # z>0 
                    elif self.args.is_sub1 == "sub2":
                        loss_facing = torch.sum(torch.relu(facing[:,2])) # z<0 
                        
                    # lx: align to opti-track 
                    loss_opti = self.fitting_loss.optitrack_fit(observed_data, pred_data, flag_rhand, flag_lhand)
                    loss = loss + 1000*loss_facing + 20*loss_opti

                    loss.backward() 
                    return loss 

                root_optim.step(closure)

            body_pose = self.latent2pose(self.latent_pose)
            stage1_pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
            per_stage_outputs['stage1'] = stage1_pred_data

            # save results
            if stages_res_out is not None:
                res_betas = self.betas.clone().detach().cpu().numpy()
                res_trans = self.trans.clone().detach().cpu().numpy()
                res_root_orient = self.root_orient.clone().detach().cpu().numpy()
                res_body_pose = body_pose.clone().detach().cpu().numpy()
                pose_hand = torch.cat([ self.left_hand_pose.squeeze(0), self.right_hand_pose.squeeze(0)],dim=1) if self.left_hand_pose!=None else None
                
                for bidx, res_out_path in enumerate(stages_res_out):
                    np.savez(cur_res_out_path, betas=res_betas[bidx],
                                            trans=res_trans[bidx],
                                            root_orient=res_root_orient[bidx],
                                            pose_body=res_body_pose[bidx],
                                            pose_hand = pose_hand.cpu().numpy())
            
        # 
        # Stage II pose and wrist
        # 
        Logger.log('Optimizing stage 2 - pose and wrist for %d iterations..' % (num_iter[1]))
        cur_res_out_path = os.path.join(stages_res_out[0], 'stage2_results.npz')
        self.fitting_loss.set_stage(1)
        self.trans.requires_grad = True
        self.root_orient.requires_grad = True
        self.betas.requires_grad = True
        self.latent_pose.requires_grad = True

        smpl_opt_params = [self.trans, self.root_orient, self.betas, self.latent_pose]

        smpl_optim = torch.optim.LBFGS(smpl_opt_params,
                                    max_iter=lbfgs_max_iter,
                                    lr=lr,
                                    line_search_fn=LINE_SEARCH)

        MSELoss = nn.MSELoss()
        # gt hands pose
        gt_pose_lh=torch.cat((observed_data['lhand'][:,:,-1:],observed_data['lhand'][:,:,3:-1]), dim=2)
        gt_pose_rh=torch.cat((observed_data['rhand'][:,:,-1:],observed_data['rhand'][:,:,3:-1]), dim=2)
        # gt pose to matrix 
        gt_pose_lh_mat = quaternion_to_matrix(gt_pose_lh).to(torch.float32).to('cuda')
        gt_pose_rh_mat = quaternion_to_matrix(gt_pose_rh).to(torch.float32).to('cuda')
        lh_roty = torch.tensor([[0,0,-1],[0,1,0],[1,0,0]]).to(torch.float32).to('cuda')  # y -90° 
        rh_roty = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]]).to(torch.float32).to('cuda') # y 90
        gt_pose_lh_mat =  torch.matmul(gt_pose_lh_mat, lh_roty).squeeze(0)
        gt_pose_rh_mat = torch.matmul(gt_pose_rh_mat, rh_roty).squeeze(0)

        gt_pose_lh_euler = matrix_to_euler_angles(gt_pose_lh_mat,"XYZ")
        gt_pose_rh_euler = matrix_to_euler_angles(gt_pose_rh_mat,"XYZ")
        # jitter in rotation
        flag_jitter_pose_lh =torch.sum((gt_pose_lh_euler[1:,:] - gt_pose_lh_euler[:-1,:])**2, dim=1) > 10.0
        flag_jitter_pose_rh =torch.sum((gt_pose_rh_euler[1:,:] - gt_pose_rh_euler[:-1,:])**2, dim=1) > 10.0 
        indx_jitter_pose_lh = flag_jitter_pose_lh.nonzero(as_tuple=True)[0]
        indx_jitter_pose_rh = flag_jitter_pose_rh.nonzero(as_tuple=True)[0]
        flag_lhand[indx_jitter_pose_lh] = False
        flag_rhand[indx_jitter_pose_rh] = False

        loss_record = torch.zeros(1,1)      
        for i in range(num_iter[1]):
            # Logger.log('ITER: %d' % (i))
            def closure():
                smpl_optim.zero_grad()
                
                pred_data = dict()
                # Use current params to go through SMPL and get joints3d, verts3d, points3d
                body_pose = self.latent2pose(self.latent_pose)
                
                pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
                pred_data['latent_pose'] = self.latent_pose 
                pred_data['betas'] = self.betas
                # compute data losses and pose prior
                loss, stats_dict = self.fitting_loss.smpl_fit(observed_data, pred_data, self.seq_len)
                # log_cur_stats(stats_dict, loss, iter=i)
                
                ## loss_opti: align to opti-track 
                if self.args.is_sub1=="sub1":
                    loss_opti = self.fitting_loss.optitrack_fit(observed_data, pred_data, flag_rhand, flag_lhand)
                    loss = loss + 50*loss_opti
                
                ## loss_wrist: compute hand pose [global pose] via kinematic tree  [seq_num*52*4*4]
                if i > 10 and self.args.is_sub1=="sub1":
                    trans_mat = self.body_model.get_global_transforms_foralljoints(self.betas, self.root_orient, body_pose)
                    pose_lh = matrix_to_euler_angles(trans_mat[:,20,:3,:3],"XYZ") # [seq_num. 3. 3] -> [seq_num. 3]
                    pose_rh = matrix_to_euler_angles(trans_mat[:,21,:3,:3],"XYZ")
                    loss_wrist = MSELoss(pose_lh[flag_lhand], gt_pose_lh_euler[flag_lhand]) + \
                                    MSELoss(pose_rh[flag_rhand], gt_pose_rh_euler[flag_rhand])
                    
                    ## wrist smoothness loss
                    loss_poselh_smooth = (pose_lh[1:,:] - pose_lh[:-1,:])**2
                    loss_poserh_smooth = (pose_rh[1:,:] - pose_rh[:-1,:])**2
                    loss_pose_wrist_smooth = 0.5*torch.sum(loss_poselh_smooth + loss_poserh_smooth)
                
                    loss = loss + 300*loss_wrist + loss_pose_wrist_smooth
                                    
                loss_record[0] = loss.item()
                loss.backward()
                return loss

            smpl_optim.step(closure)

        self.body_pose = self.latent2pose(self.latent_pose).detach()
        
        self.body_pose[:,:,:30] = observed_data['mmpose'][:,:,:30] # set mm pose [lower limb] to final poses
        self.trans = torch.zeros_like(self.trans)
        self.root_orient = torch.zeros_like(self.root_orient)
        
        stage2_pred_data, _ = self.smpl_results(self.trans, self.root_orient, self.body_pose, self.betas)
        per_stage_outputs['stage2'] = stage2_pred_data

        if stages_res_out is not None:                 
            res_betas = self.betas.clone().detach().cpu().numpy()
            res_trans = self.trans.clone().detach().cpu().numpy()
            res_root_orient = self.root_orient.clone().detach().cpu().numpy()
            res_body_pose = self.body_pose.clone().detach().cpu().numpy()
            pose_hand = torch.cat([ self.left_hand_pose.squeeze(0), self.right_hand_pose.squeeze(0)],dim=1) if self.left_hand_pose!=None else None
            for bidx, res_out_path in enumerate(stages_res_out):
                cur_res_out_path = os.path.join(res_out_path, 'stage2_results.npz')
                np.savez(cur_res_out_path, betas=res_betas[bidx],
                                        trans=res_trans[bidx],
                                        root_orient=res_root_orient[bidx],
                                        pose_body=res_body_pose[bidx],
                                        pose_hand = pose_hand.cpu().numpy())
                final_loss = loss_record.detach().cpu().numpy()
                np.savetxt( os.path.join(res_out_path, '%f.txt' % final_loss), final_loss)
                print(loss_record.detach().cpu().numpy())

        # self.hand_optimize(observed_data, data_fps, lr, num_iter, lbfgs_max_iter,stages_res_out,fit_gender)
    
        final_optim_res = self.get_optim_result(self.body_pose)
        self.save_smpl_ply()
           
        return final_optim_res,  per_stage_outputs
          
    def hand_optimize(self,observed_data,
                  data_fps=30,
                  lr=1.0,
                  num_iter=[30, 70, 70],
                  lbfgs_max_iter=20,
                  stages_res_out=None,
                  fit_gender='neutral'):
        per_stage_outputs = {} # SMPL results after each stage
        # Stage III: hand position optimization 
        #
        Logger.log('Optimizing stage 3 - optitrack-based optimization %d ...' % (num_iter[2]))
        self.trans.requires_grad = False
        self.root_orient.requires_grad = False
        self.betas.requires_grad = False
        self.latent_pose.requires_grad = False
        self.body_pose.requires_grad = False
        
        n_roi_pose = 6
        roi_pose = torch.zeros((1,self.body_pose.shape[1],n_roi_pose)).to('cuda')
        roi_pose[:] = self.body_pose[:,:,-n_roi_pose:]
        fix_pose = self.body_pose[:,:,:-n_roi_pose]
        roi_pose.requires_grad = True
        fix_pose.requires_grad = False

        hand_opt_params = [roi_pose]

        hand_optim = torch.optim.LBFGS(hand_opt_params,
                                        max_iter=lbfgs_max_iter,
                                        lr=lr,
                                        line_search_fn=LINE_SEARCH)

        
        MSELoss = nn.MSELoss()
        L1loss = nn.L1Loss()
        
        # 1. gt pose
        gt_pose_lh=torch.cat((observed_data['lhand'][:,:,-1:],observed_data['lhand'][:,:,3:-1]), dim=2)
        gt_pose_rh=torch.cat((observed_data['rhand'][:,:,-1:],observed_data['rhand'][:,:,3:-1]), dim=2)
        # to matrix 
        gt_pose_lh_mat = quaternion_to_matrix(gt_pose_lh).to(torch.float32).to('cuda')
        gt_pose_rh_mat = quaternion_to_matrix(gt_pose_rh).to(torch.float32).to('cuda')
        lh_roty = torch.tensor([[0,0,-1],[0,1,0],[1,0,0]]).to(torch.float32).to('cuda')  # y -90° 
        rh_roty = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]]).to(torch.float32).to('cuda') # y 90
        gt_pose_lh_mat =  torch.matmul(gt_pose_lh_mat, lh_roty).squeeze(0)
        gt_pose_rh_mat = torch.matmul(gt_pose_rh_mat, rh_roty).squeeze(0)
        
        for i in range(30): 
            # Logger.log('ITER: %d' % (i))
            self.fitting_loss.cur_optim_step = i 
            stats_dict = None 
            def closure(): 
                hand_optim.zero_grad() 
                
                body_pose = torch.cat([fix_pose,roi_pose],dim=2)
                
                # 2. compute hand pose [global pose] via kinematic tree  [seq_num*52*4*4]
                trans_mat = self.body_model.get_global_transforms_foralljoints(self.betas, self.root_orient, body_pose)
                pose_lh = (trans_mat[:,20,:3,:3]) # [seq_num. 3. 3]
                pose_rh = (trans_mat[:,21,:3,:3])
                
                # 3. pose loss
                loss_pose = MSELoss(pose_lh, gt_pose_lh_mat) + MSELoss(pose_rh, gt_pose_rh_mat)
                
                loss = loss_pose

                loss.backward() 
                return loss 

            hand_optim.step(closure)

        self.body_pose = torch.cat([fix_pose,roi_pose],dim=2)
        self.latent_pose = self.pose2latent(self.body_pose)
        stage2_pred_data, _ = self.smpl_results(self.trans, self.root_orient, self.body_pose, self.betas)
        per_stage_outputs['stage2'] = stage2_pred_data
        final_optim_res = self.get_optim_result(self.body_pose)
        self.save_smpl_ply()
        return final_optim_res,  per_stage_outputs
    
    def cam_solvePnp(self, observed_data):
        
        obv_rhand = observed_data['rhand'][0,:,:3].cpu().numpy() 
        obv_lhand = observed_data['lhand'][0,:,:3].cpu().numpy() 
        obv_head = observed_data['sub1_head'][0,:,:3].cpu().numpy()  # world [x,y,z]
        obv_head[:,1] -= 0.25
        model_points = np.concatenate((obv_rhand,obv_lhand,obv_head),axis=0)
        
        openpose_rhand = observed_data['joints2d'][0,:,4,:].cpu().numpy() 
        openpose_lhand = observed_data['joints2d'][0,:,7,:].cpu().numpy()  
        openpose_head = observed_data['joints2d'][0,:,17,:].cpu().numpy() 
        image_points = np.concatenate((openpose_rhand[:,:2],openpose_lhand[:,:2],openpose_head[:,:2]),axis=0)
        
        confi_flag = np.concatenate((openpose_rhand[:,2],openpose_lhand[:,2],openpose_head[:,2]),axis=0)
        flag = confi_flag>0.8
        # row_cond = flag.all(0)
        
        model_points = model_points[flag,:]
        image_points = image_points[flag,:]
        
        
        cam_K = torch.tensor([[self.cam_f[0,0], 0.0, self.cam_center[0,0],0],
                [0.0, self.cam_f[0,1], self.cam_center[0,1],0],
                [0.0, 0.0, 1.0,0]]).numpy() 
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = \
            cv2.solvePnP(model_points, image_points, cam_K[:3,:3], dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        rotM = cv2.Rodrigues(rotation_vector)[0]
        # position = -np.matrix(rotM).T * np.matrix(translation_vector)
        # print("file Vector:\n {0}".format(-np.matrix(rotM).T * np.matrix(translation_vector)))
        # print(position)
        
        t_vec = torch.tensor([0,0,0,1]).unsqueeze(0)
        cam_R = torch.from_numpy(rotM)
        cam_T = torch.from_numpy(translation_vector)
        cam_RT = torch.cat((cam_R, cam_T),dim=1) 
        RT_homo = torch.cat((cam_RT, t_vec), dim=0).squeeze()
        print(RT_homo.detach())
        # print(self.cam2prior_t.detach())
        
        # cam_extri_init = cam_extri_init.detach().cpu()
        # RT_homo_seq = RT_homo.repeat(obv_rhand.shape[0],1,1)
        # K = cam_K.repeat(obv_rhand.shape[0],1,1)
        cam_K = torch.from_numpy(cam_K)
        RT_homo=RT_homo.detach().cpu().float()
        
        # # 2. re-projection visualization 
        # for frame_id in range(0,300,30):
        #     self.write2img(frame_id, observed_data, cam_K, RT_homo, openpose_rhand[:,:2],openpose_lhand[:,:2],openpose_head[:,:2],"pnp")


        # write to file
        data_folder = self.args.data_path
        data_folder = Path(data_folder)
        root_folder = str(data_folder.parent.parent.parent)      
        data_folder = str(data_folder.parent.parent)
        
        np.savetxt(data_folder + "/CamIntr.txt", cam_K.detach().cpu().numpy())
        np.savetxt(data_folder + "/CamExtr.txt", RT_homo.detach().cpu().numpy())
            
        return cam_K, RT_homo


    def cam_optimize(self, observed_data, 
                     data_fps=30,
                  lr=1,
                  num_iter=[30, 70, 70],
                  lbfgs_max_iter=20,
                  stages_res_out=None,
                  fit_gender='neutral'): 

        # 1. observation-based optimization 
        self.cam2prior_t.requires_grad = True
        cam_euler = torch.zeros((3)).to('cuda')
        cam_euler.requires_grad = True
        cam_optim_params = [cam_euler, self.cam2prior_t]
        cam_optim = torch.optim.LBFGS(cam_optim_params,
                                    max_iter=lbfgs_max_iter,
                                    lr=lr,
                                    line_search_fn=LINE_SEARCH)
        
        
        # hand motion position [right hand only] 
        obv_rhand = observed_data['rhand'][0,:,:3] # world [x,y,z]
        t_homo = torch.ones((obv_rhand.shape[0],1)).to('cuda')
        obv_rhand = torch.cat((obv_rhand, t_homo), dim=1).unsqueeze(-1)
        # openpose 2d hand position
        openpose_rhand = observed_data['joints2d'][0,:,4,:]  # rhand [u,v,conf]
        flag_openpose = openpose_rhand[:,2]<0.8
        openpose_rhand[:,2][flag_openpose] = 0.0
        
        # hand motion position [left hand] 
        obv_lhand = observed_data['lhand'][0,:,:3] # world [x,y,z]
        t_homo = torch.ones((obv_lhand.shape[0],1)).to('cuda')
        obv_lhand = torch.cat((obv_lhand, t_homo), dim=1).unsqueeze(-1)
        # openpose 2d hand position
        openpose_lhand = observed_data['joints2d'][0,:,7,:]  # lhand [u,v,conf]
        flag_openpose = openpose_lhand[:,2]<0.8
        openpose_lhand[:,2][flag_openpose] = 0.0
        
        # head motion position [right hand only] 
        obv_head = observed_data['sub1_head'][0,:,:3].cpu() # world [x,y,z]
        obv_head[:,1] -= 0.25
        obv_head = obv_head.to('cuda')
        obv_head = torch.cat((obv_head, t_homo), dim=1).unsqueeze(-1)
        # openpose 2d hand position
        openpose_head = observed_data['joints2d'][0,:,17,:]
        flag_openpose = openpose_head[:,2]<0.8
        openpose_head[:,2][flag_openpose] = 0.0
        
        # similar to example:
        ext_file = "/home/ur-5/Projects/justlx/test_data/CamExtr.txt"
        ext_mat = torch.from_numpy(np.loadtxt(ext_file)).to('cuda').float()
        
        # optimization
        MSELoss = nn.MSELoss()
        pdist = nn.PairwiseDistance(p=2)
        cam_K = torch.tensor([[self.cam_f[0,0], 0.0, self.cam_center[0,0],0],
                [0.0, self.cam_f[0,1], self.cam_center[0,1],0],
                [0.0, 0.0, 1.0,0]]).to('cuda')
        t_vec = torch.tensor([0,0,0,1]).unsqueeze(0).unsqueeze(0).to('cuda')
        u_vec = torch.tensor([0,0,1]).to('cuda').float()
        cam_K.requires_grad_(False)
        t_vec.requires_grad_(False)
        for i in range(50):
            def closure():
                cam_optim.zero_grad()
                
                cam_R = self.euler_angles_to_matrix(cam_euler,"XYZ").unsqueeze(0)
                cam_RT = torch.cat((cam_R, self.cam2prior_t.unsqueeze(0)),dim=2) 
                RT_homo = torch.cat((cam_RT, t_vec), dim=1)
                
                RT_homo_seq = RT_homo.repeat(obv_rhand.shape[0],1,1)
                K = cam_K.repeat(obv_rhand.shape[0],1,1)
                
                cam_rhand = torch.bmm(RT_homo_seq, obv_rhand)
                esti_uv = torch.bmm(K, cam_rhand)
                flag = (esti_uv[:,2,:]==0).squeeze()
                esti_uv = torch.div(esti_uv, esti_uv[:,2,:].unsqueeze(1).repeat(1,3,1))
                
                
                cam_lhand = torch.bmm(RT_homo_seq, obv_lhand)
                esti_uv_l = torch.bmm(K, cam_lhand)
                flag = (esti_uv_l[:,2,:]==0).squeeze()
                esti_uv_l = torch.div(esti_uv_l, esti_uv_l[:,2,:].unsqueeze(1).repeat(1,3,1))
                
                cam_head = torch.bmm(RT_homo_seq, obv_head)
                esti_uv_head = torch.bmm(K, cam_head)
                flag = (esti_uv_head[:,2,:]==0).squeeze()
                # openpose_head[:,2][flag] = 0.0
                # esti_uv_head[:,2,:]=1e-10
                esti_uv_head = torch.div(esti_uv_head, esti_uv_head[:,2,:].unsqueeze(1).repeat(1,3,1))

                loss_data_rhand = torch.sum(((pdist(esti_uv[:,:2,0], openpose_rhand[:,:2]) / openpose_rhand.shape[0])*openpose_rhand[:,2])[~flag])
                loss_data_lhand = torch.sum(((pdist(esti_uv_l[:,:2,0], openpose_lhand[:,:2]) / openpose_lhand.shape[0])*openpose_lhand[:,2])[~flag])
                loss_data_head = torch.sum(((pdist(esti_uv_head[:,:2,0], openpose_head[:,:2]) / openpose_head.shape[0])*openpose_head[:,2])[~flag])


                # loss_cam = obv_rhand.shape[0] * MSELoss(cam_R[0,0,:],u_vec) # +u_axis is parallel to +z_axis 
                # loss_cam += obv_rhand.shape[0] *  torch.relu(cam_R[0,2,1]) # z[1]<0
                loss_cam = obv_rhand.shape[0] * MSELoss(cam_R[0,:,:],ext_mat[:3,:3])
                
                
                loss = 3*loss_data_rhand + loss_data_lhand + loss_data_head + 50*loss_cam
                
                loss.backward()
                return loss
            cam_optim.step(closure)    
                
        
        cam_R = self.euler_angles_to_matrix(cam_euler,"XYZ").unsqueeze(0)
        cam_RT = torch.cat((cam_R, self.cam2prior_t.unsqueeze(0)),dim=2) 
        RT_homo = torch.cat((cam_RT, t_vec), dim=1).squeeze()
        print(RT_homo.detach())
        print(cam_euler.detach())
        # print(self.cam2prior_t.detach())
        
        cam_K = cam_K.detach().cpu()
        RT_homo=RT_homo.detach().cpu()
        
        # # 2. re-projection visualization 
        # for frame_id in range(0,obv_lhand.shape[0],10):
        #     self.write2img(frame_id, observed_data, cam_K, RT_homo, openpose_rhand,openpose_lhand,openpose_head,"opt")

        data_folder = self.args.data_path
        data_folder = Path(data_folder)
        root_folder = str(data_folder.parent.parent.parent)      
        data_folder = str(data_folder.parent.parent)
        
        # NAN judgement
        if torch.isnan(RT_homo).any():
            cam_K, RT_homo = self.cam_solvePnp(observed_data)
                
        # write to file
        np.savetxt(data_folder + "/CamIntr.txt", cam_K.detach().cpu().numpy())
        np.savetxt(data_folder + "/CamExtr.txt", RT_homo.detach().cpu().numpy())
            
        return cam_K, RT_homo
               
        
    def write2img(self, frame_id, observed_data, cam_K, camRT,openpose_rhand,openpose_lhand,openpose_head, name=""):
        # 2. re-projection visualization 
        image_path = self.args.data_path + '/left_%04d.jpg' % frame_id
        npimg = cv2.imread(image_path)
        
        r_p = torch.ones([4])
        rhand_point = observed_data['rhand'][0,frame_id,:3]
        r_p[:3] = rhand_point
        rhand_2d = cam_K@camRT@r_p
        rhand_2d = torch.div(rhand_2d, rhand_2d[2])
        
        l_p = torch.ones([4])
        lhand_point = observed_data['lhand'][0,frame_id,:3]
        l_p[:3] = lhand_point
        lhand_2d = cam_K@camRT@l_p
        lhand_2d = torch.div(lhand_2d, lhand_2d[2])
        
        h_p = torch.ones([4])
        head_point = observed_data['sub1_head'][0,frame_id,:3]
        h_p[:3] = head_point
        h_2d = cam_K@camRT@h_p
        h_2d = torch.div(h_2d, h_2d[2])
        
        cv2.circle(npimg, (int(rhand_2d[0].item() + 0.5), int(rhand_2d[1].item() + 0.5)), 3, [255, 0, 0], thickness=3, lineType=8, shift=0)
        cv2.circle(npimg, (int(lhand_2d[0].item() + 0.5), int(lhand_2d[1].item() + 0.5)), 3, [0, 255, 0], thickness=3, lineType=8, shift=0)
        cv2.circle(npimg, (int(h_2d[0].item() + 0.5), int(h_2d[1].item() + 0.5)), 3, [0, 0, 255], thickness=3, lineType=8, shift=0)
        
        cv2.circle(npimg, (int(openpose_rhand[frame_id,0].item() + 0.5), int(openpose_rhand[frame_id,1].item() + 0.5)), 3, [0, 255, 255], thickness=3, lineType=8, shift=0)
        cv2.circle(npimg, (int(openpose_lhand[frame_id,0].item() + 0.5), int(openpose_lhand[frame_id,1].item() + 0.5)), 3, [0, 255, 255], thickness=3, lineType=8, shift=0)
        cv2.circle(npimg, (int(openpose_head[frame_id,0].item() + 0.5), int(openpose_head[frame_id,1].item() + 0.5)), 3, [0, 255, 255], thickness=3, lineType=8, shift=0)
        cv2.imwrite('test_%04d_%s.jpg' % (frame_id,name), npimg)

    def save_smpl_ply(self):

        with torch.no_grad():
            pred_output = self.body_model.bm(
                                    body_pose=self.body_pose[0,:,:],
                                    global_orient=self.root_orient[0,:,:],
                                    transl=self.trans[0,:,:],
                                    left_hand_pose=self.left_hand_pose[0,:,:],
                                    right_hand_pose=self.right_hand_pose[0,:,:]
                                    )
        verts = pred_output.vertices.cpu().numpy()
        faces = self.body_model.bm.faces
        
        meshes_dir = os.path.join(self.args.out, "body_meshes_humor")
        print(f"save meshes to \"{meshes_dir}\"")
        os.makedirs(meshes_dir, exist_ok=True)
        
        n = len(verts)
        for ii in range(n):
            verts0 = np.array(verts[ii])
            mesh0 = trimesh.Trimesh(verts0, faces)
                
            # save mesh0
            fram_name =  str(ii)
            filename =  "%06d_humor_smplh_%s.ply" % (self.take_id, fram_name)   
            out_mesh_path = os.path.join(meshes_dir, filename)
            mesh0.export(out_mesh_path)
    
    def euler_angles_to_matrix(self, euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as Euler angles in radians to rotation matrices.

        Args:
            euler_angles: Euler angles in radians as tensor of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            self._axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]
        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
    
    def _axis_angle_rotation(self, axis: str, angle: torch.Tensor) -> torch.Tensor:
        """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def apply_cam2prior(self, data_dict, R, t, root_height, body_pose, betas, key_frame_idx, inverse=False):
        '''
        Applies the camera2prior tranformation made up of R, t to the data in data dict and
        returns a new dictionary with the transformed data.
        Right now supports: trans, root_orient.

        NOTE: If the number of timesteps in trans/root_orient is 1, this function assumes they are at key_frame_idx.
                (othherwise the calculation of cur_root_height or trans_offset in inverse case is not correct)

        key_frame_idx : the timestep used to compute cam2prior size (B) tensor
        inverse : if true, applies the inverse transformation from prior space to camera
        '''
        prior_dict = dict()
        if 'root_orient' in data_dict:
            # B x T x 3
            root_orient = data_dict['root_orient']
            B, T, _ = root_orient.size()
            R_time = R.unsqueeze(1).expand((B, T, 3, 3))
            t_time = t.unsqueeze(1).expand((B, T, 3))
            root_orient_mat = batch_rodrigues(root_orient.reshape((-1, 3))).reshape((B, T, 3, 3))
            if inverse:
                prior_root_orient_mat = torch.matmul(R_time.transpose(3, 2), root_orient_mat)
            else:
                prior_root_orient_mat = torch.matmul(R_time, root_orient_mat)
            prior_root_orient = rotation_matrix_to_angle_axis(prior_root_orient_mat.reshape((B*T, 3, 3))).reshape((B, T, 3))
            prior_dict['root_orient'] = prior_root_orient

        if 'trans' in data_dict and 'root_orient' in data_dict:
            # B x T x 3
            trans = data_dict['trans']
            B, T, _ = trans.size()
            R_time = R.unsqueeze(1).expand((B, T, 3, 3))
            t_time = t.unsqueeze(1).expand((B, T, 3))
            if inverse:
                # transform so key frame at origin
                if T > 1:
                    trans_offset = trans[np.arange(B),key_frame_idx,:].unsqueeze(1)
                else:
                    trans_offset = trans[:,0:1,:]
                trans = trans - trans_offset
                # rotates to camera frame
                trans = torch.matmul(R_time.transpose(3, 2), trans.reshape((B, T, 3, 1)))[:,:,:,0]
                # translate to camera frame
                trans = trans - t_time
            else:
                # first transform so the trans of key frame is at origin
                trans = trans + t_time
                # then rotate to canonical frame
                trans = torch.matmul(R_time, trans.reshape((B, T, 3, 1)))[:,:,:,0]
                # then apply floor offset so the root joint is at the desired height
                cur_smpl_data, _ = self.smpl_results(trans, prior_dict['root_orient'], body_pose, betas)
                if T > 1:
                    cur_root_height = cur_smpl_data['joints3d'][np.arange(B),key_frame_idx,0,2:3]
                else:
                    cur_root_height = cur_smpl_data['joints3d'][:,0,0,2:3]
                height_diff = root_height - cur_root_height
                trans_offset = torch.cat([torch.zeros((B, 2)).to(height_diff), height_diff], axis=1)
                trans = trans + trans_offset.reshape((B, 1, 3))
            prior_dict['trans'] = trans
        elif 'trans' in data_dict:
            Logger.log('Cannot apply cam2prior on translation without root orient data!')
            exit()

        return prior_dict
  
    def get_optim_result(self, body_pose=None):
        '''
        Collect final outputs into a dict.
        '''
        if body_pose is None:
            body_pose = self.latent2pose(self.latent_pose)
        optim_result = {
            'trans' : self.trans.clone().detach(),
            'root_orient' : self.root_orient.clone().detach(),
            'pose_body' : body_pose.clone().detach(),
            'betas' : self.betas.clone().detach(),
            'latent_pose' : self.latent_pose.clone().detach()   
        }
        # optim_result['latent_motion'] = self.latent_motion.clone().detach()
        if self.left_hand_pose!=None:
            pose_hand = torch.cat([ self.left_hand_pose, self.right_hand_pose],dim=2)
            optim_result['pose_hand'] = pose_hand.clone().detach()
        
        if self.optim_floor:
            ground_plane = parse_floor_plane(self.floor_plane)
            optim_result['floor_plane'] = ground_plane.clone().detach()
        
        return optim_result

    def latent2pose(self, latent_pose):
        '''
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        '''
        B, T, _ = latent_pose.size()
        latent_pose = latent_pose.reshape((-1, self.latent_pose_dim))
        body_pose = self.pose_prior.decode(latent_pose, output_type='matrot')
        body_pose = rotation_matrix_to_angle_axis(body_pose.reshape((B*T*J_BODY, 3, 3))).reshape((B, T, J_BODY*3))
        return body_pose

    def pose2latent(self, body_pose):
        '''
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        '''
        B, T, _ = body_pose.size()
        body_pose = body_pose.reshape((-1, J_BODY*3))
        latent_pose_distrib = self.pose_prior.encode(body_pose)
        latent_pose = latent_pose_distrib.mean.reshape((B, T, self.latent_pose_dim))
        return latent_pose

    def smpl_results(self, trans, root_orient, body_pose, beta):
        '''
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        beta : B x D
        '''
        B, T, _ = trans.size()
        if T == 1:
            # must expand to use with body model
            trans = trans.expand((self.batch_size, self.seq_len, 3))
            root_orient = root_orient.expand((self.batch_size, self.seq_len, 3))
            body_pose = body_pose.expand((self.batch_size, self.seq_len, J_BODY*3))
        elif T != self.seq_len:
            # raise NotImplementedError('Only supports single or all steps in body model.')
            pad_size = self.seq_len - T
            trans, root_orient, body_pose = self.zero_pad_tensors([trans, root_orient, body_pose], pad_size)


        betas = beta.reshape((self.batch_size, 1, self.num_betas)).expand((self.batch_size, self.seq_len, self.num_betas))
        pose_hand = torch.cat([ self.left_hand_pose.squeeze(0), self.right_hand_pose.squeeze(0)],dim=1) if self.left_hand_pose!=None else None
        smpl_body = self.body_model(pose_body=body_pose.reshape((self.batch_size*self.seq_len, -1)), 
                                    pose_hand=pose_hand, 
                                    betas=betas.reshape((self.batch_size*self.seq_len, -1)),
                                    root_orient=root_orient.reshape((self.batch_size*self.seq_len, -1)),
                                    trans=trans.reshape((self.batch_size*self.seq_len, -1))
                                    )
        # body joints
        joints3d = smpl_body.Jtr.reshape((self.batch_size, self.seq_len, -1, 3))[:, :T]
        body_joints3d = joints3d[:,:,:len(SMPL_JOINTS),:]
        added_joints3d = joints3d[:,:,len(SMPL_JOINTS):,:]
        # ALL body vertices
        points3d = smpl_body.v.reshape((self.batch_size, self.seq_len, -1, 3))[:, :T]
        # SELECT body vertices
        verts3d = points3d[:, :T, KEYPT_VERTS, :]

        pred_data = {
            'joints3d' : body_joints3d,
            'points3d' : points3d,
            'verts3d' : verts3d,
            'joints3d_extra' : added_joints3d, # hands and selected OP vertices (if applicable) 
            'faces' : smpl_body.f # always the same, but need it for some losses
        }
        
        return pred_data, smpl_body

    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x T x D and pad temporal dimension
        '''
        B = pad_list[0].size(0)
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
        return new_pad_list
    
    
# camextr: numpy arr [4,4]
def isgoodcamextrix(camextr):
    if np.isnan(camextr).any():
        return False
    if (camextr[0,3]<0) & (camextr[1,3]>0) & (camextr[2,3]>0) or (camextr[0,3]>0) & (camextr[1,3]<0) & (camextr[2,3]<0):
        return True
    else:
        return False 