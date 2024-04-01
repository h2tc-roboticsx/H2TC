'''
Test-time optimization to fit to observations using HuMoR as a motion prior.

This is a 3-stage optimization. Stages 1 & 2 are initialization that DON'T use the motion prior,
and stage 3 is the main optimization that uses HuMoR.
'''

from pathlib import Path
import sys, os, glob
import cv2
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
sys.path.append(os.path.join(cur_file_path, '.'))

import importlib, time, math, shutil, json
import traceback

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from utils.logging import Logger, cp_files

from h2tc_fit_dataset_mm import H2TCFitDataset
from utils.logging import mkdir
from config import parse_args
from fitting_utils import NSTAGES, DEFAULT_FOCAL_LEN, load_vposer, save_optim_result, save_rgb_stitched_result
from motion_optimizer import MotionOptimizer

from human_tools.body_model import BodyModel
from human_tools.body_model import SMPLX_PATH, SMPLH_PATH

from fitting_utils import vis_results_lx

def main(args, config_file):
    res_out_path = None
    if args.out is not None:
        mkdir(args.out)
        # create logging system
        fit_log_path = os.path.join(args.out, 'fit_' + str(int(time.time())) + '.log')
        Logger.init(fit_log_path)

        if args.save_results or args.save_stages_results:
            res_out_path = os.path.join(args.out, 'results_out')

    # save arguments used
    Logger.log('args: ' + str(args))
    # and save config
    cp_files(args.out, [config_file])
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    B = args.batch_size
    if args.amass_batch_size > 0:
        B = args.amass_batch_size
    if args.prox_batch_size > 0:
        B = args.prox_batch_size
    if B == 3:
        Logger.log('Cannot use batch size 3, setting to 2!') # NOTE: bug with pytorch 3x3 matmul weirdness
        B = 2
    dataset = None
    data_fps = args.data_fps
    im_dim = (1080, 1080)
    rgb_img_folder = None
    rgb_vid_name = None
   
    img_folder = args.data_path
    video_preprocess_path = args.out+'/rgb_preprocess'

    # read images
    use_custom_keypts = args.op_keypts is not None
    img_paths = glob.glob(img_folder + "/left*.[jp][pn]g")
    img_paths.sort()
    img_path = img_paths[0]
    img_shape = cv2.imread(img_path).shape

    # Create dataset by splitting the video up into overlapping clips
    vid_name = '.'.join(args.data_path.split('/')[-1].split('.')[:-1])
    dataset = H2TCFitDataset(joints2d_path=None,
                                cam_mat=None,
                                seq_len=args.rgb_seq_len,
                                overlap_len=args.rgb_overlap_len,
                                img_path=img_folder,
                                load_img=False,
                                masks_path=None,
                                mask_joints=args.mask_joints2d,
                                planercnn_path=args.rgb_planercnn_res,
                                video_name=vid_name,
                                is_sub1= args.is_sub1,
                                zed_path=None,
                                args = args,
                            )
    cam_mat = dataset.cam_mat

    data_fps = args.fps
    im_dim = tuple(img_shape[:-1][::-1])
    rgb_img_folder = img_folder
    rgb_vid_name = vid_name

    data_loader = DataLoader(dataset, 
                            batch_size=B,
                            shuffle=args.shuffle,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=lambda _: np.random.seed())

    # weights for optimization loss terms
    loss_weights = {
        'joints2d' : args.joint2d_weight,
        'joints3d' : args.joint3d_weight,
        'joints3d_rollout' : args.joint3d_rollout_weight,
        'verts3d' : args.vert3d_weight,
        'points3d' : args.point3d_weight,
        'pose_prior' : args.pose_prior_weight,
        'shape_prior' : args.shape_prior_weight,
        'motion_prior' : args.motion_prior_weight,
        'init_motion_prior' : args.init_motion_prior_weight,
        'joint_consistency' : args.joint_consistency_weight,
        'bone_length' : args.bone_length_weight,
        'joints3d_smooth' : args.joint3d_smooth_weight,
        'contact_vel' : args.contact_vel_weight,
        'contact_height' : args.contact_height_weight,
        'floor_reg' : args.floor_reg_weight,
        'rgb_overlap_consist' : args.rgb_overlap_consist_weight
    }

    max_loss_weights = {k : max(v) for k, v in loss_weights.items()}
    all_stage_loss_weights = []
    for sidx in range(NSTAGES):
        stage_loss_weights = {k : v[sidx] for k, v in loss_weights.items()}
        all_stage_loss_weights.append(stage_loss_weights)
        
    use_joints2d = max_loss_weights['joints2d'] > 0.0

    # must always have pose prior to optimize in latent space
    pose_prior, _ = load_vposer(args.vposer)
    pose_prior = pose_prior.to(device)
    pose_prior.eval()

    motion_prior = None

    # load in prior on the initial motion state if given
    init_motion_prior = None
    if motion_prior is not None and max_loss_weights['init_motion_prior'] > 0.0:
        Logger.log('Loading initial motion state prior from %s...' % (args.init_motion_prior))
        gmm_path = os.path.join(args.init_motion_prior, 'prior_gmm.npz')
        init_motion_prior = dict()
        if os.path.exists(gmm_path):
            gmm_res = np.load(gmm_path)
            gmm_weights = torch.Tensor(gmm_res['weights']).to(device)
            gmm_means = torch.Tensor(gmm_res['means']).to(device)
            gmm_covs = torch.Tensor(gmm_res['covariances']).to(device)
            init_motion_prior['gmm'] = (gmm_weights, gmm_means, gmm_covs)
        if len(init_motion_prior.keys()) == 0:
            Logger.log('Could not find init motion state prior at given directory!')
            exit()

    if args.data_type == 'RGB' and args.save_results:
        all_res_out_paths = []

    fit_errs = dict()
    prev_batch_overlap_res_dict = None
    for i, data in enumerate(data_loader):
        start_t = time.time()
        # these dicts have different data depending on modality
        # MUST have at least name
        observed_data, gt_data = data
        # both of these are a list of tuples, each list index is a frame and the tuple index is the seq within the batch
        obs_img_paths = None if 'img_paths' not in observed_data else observed_data['img_paths'] 
        obs_mask_paths = None if 'mask_paths' not in observed_data else observed_data['mask_paths']
        observed_data = {k : v.to(device) for k, v in observed_data.items() if isinstance(v, torch.Tensor)}
        for k, v in gt_data.items():
            if isinstance(v, torch.Tensor):
                gt_data[k] = v.to(device)
        cur_batch_size = observed_data[list(observed_data.keys())[0]].size(0)
        T = observed_data['rhand'].size(1)

        seq_names = []
        for gt_idx, gt_name in enumerate(gt_data['name']):
            seq_name = gt_name + '_' + str(int(time.time())) + str(gt_idx)
            # seq_name = gt_name 
            Logger.log(seq_name)
            seq_names.append(seq_name)

        cur_z_init_paths = []
        cur_z_final_paths = []
        cur_res_out_paths = []
        for seq_name in seq_names:
            # set current out paths based on sequence name
            if res_out_path is not None:
                # cur_res_out_path = os.path.join(res_out_path, seq_name)
                cur_res_out_path = res_out_path
                mkdir(cur_res_out_path)
                cur_res_out_paths.append(cur_res_out_path)
        cur_res_out_paths = cur_res_out_paths if len(cur_res_out_paths) > 0 else None
        if cur_res_out_paths is not None and args.data_type == 'RGB' and args.save_results:
            all_res_out_paths += cur_res_out_paths
        cur_z_init_paths = cur_z_init_paths if len(cur_z_init_paths) > 0 else None
        cur_z_final_paths = cur_z_final_paths if len(cur_z_final_paths) > 0 else None

        # get body model
        # load in from given path
        Logger.log('Loading SMPL model from %s...' % (args.smpl))
        body_model_path = args.smpl
        fit_gender = body_model_path.split('/')[-2]
        num_betas = 16 if 'betas' not in gt_data else gt_data['betas'].size(2)
        body_model = BodyModel(bm_path=body_model_path,
                                num_betas=num_betas,
                                batch_size=cur_batch_size*T,
                                use_vtx_selector=use_joints2d).to(device)

        if body_model.model_type != 'smplh':
            print('Only SMPL+H model is supported for current algorithm!')
            exit()

        gt_body_paths = None
        # if 'gender' in gt_data:
        #     gt_body_paths = []
        #     for cur_gender in gt_data['gender']:
        #         gt_body_path = None
        #         if args.gt_body_type == 'smplh':
        #             gt_body_path = os.path.join(SMPLH_PATH, '%s/model.npz' % (cur_gender))
        #         gt_body_paths.append(gt_body_path)

        cam_mat = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].to(device)

        #  save meta results information about the optimized bm and GT bm (gender)
        if args.save_results:
            for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
                cur_meta_path = os.path.join(cur_res_out_path, 'meta.txt')
                with open(cur_meta_path, 'w') as f:
                    f.write('optim_bm %s\n' % (body_model_path))
                    if gt_body_paths is None:
                        f.write('gt_bm %s\n' % (body_model_path))
                    else:
                        f.write('gt_bm %s\n' % (gt_body_paths[bidx]))

        # create optimizer
        optimizer = MotionOptimizer(device,
                                    body_model,
                                    num_betas,
                                    cur_batch_size,
                                    dataset.data_len,
                                    [k for k in observed_data.keys()],
                                    all_stage_loss_weights,
                                    pose_prior,
                                    motion_prior,
                                    init_motion_prior,
                                    use_joints2d,
                                    cam_mat,
                                    args.robust_loss,
                                    args.robust_tuning_const,
                                    args.joint2d_sigma,
                                    stage3_tune_init_state=args.stage3_tune_init_state,
                                    stage3_tune_init_num_frames=args.stage3_tune_init_num_frames,
                                    stage3_tune_init_freeze_start=args.stage3_tune_init_freeze_start,
                                    stage3_tune_init_freeze_end=args.stage3_tune_init_freeze_end,
                                    stage3_contact_refine_only=args.stage3_contact_refine_only,
                                    use_chamfer=('points3d' in observed_data),
                                    im_dim=im_dim,
                                    args=args)

        # run optimizer
        try:
            optimizer.run(observed_data,
                            data_fps=data_fps,
                            lr=args.lr,
                            num_iter=args.num_iters,
                            lbfgs_max_iter=args.lbfgs_max_iter,
                            stages_res_out=cur_res_out_paths,
                            fit_gender=fit_gender)




            elapsed_t = time.time() - start_t
            Logger.log('Optimized sequence %d in %f s' % (i, elapsed_t))

        except Exception as e:
            Logger.log('Caught error in current optimization! Skipping...') 
            Logger.log(traceback.format_exc()) 

        if i < (len(data_loader) - 1):
            del optimizer
        del body_model
        del observed_data
        del gt_data
        torch.cuda.empty_cache()

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    main(args, config_file)