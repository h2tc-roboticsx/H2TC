
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import shutil, glob
import os.path as osp
from pathlib import Path
import cv2
import numpy as np
import json
import torch

from human_tools.body_model import BodyModel

from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues, convert_to_rotmat
from utils.logging import mkdir, Logger

NSTAGES = 3 # number of stages in the optimization
DEFAULT_FOCAL_LEN = (699.78, 699.78) # fx, fy

from typing import Optional

import torch
import torch.nn.functional as F


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def read_keypoints(keypoint_fn, rgbd_id="rgbd0", is_sub1="sub1"):
    '''
    Only reads body keypoint data of first person.
    '''
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data['people']) == 0:
        print('WARNING: Found no keypoints in %s! Returning zeros!' % (keypoint_fn))
        return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)

    body_keypoints = np.array(data['people'][0]['pose_keypoints_2d'],
                                    dtype=np.float32).reshape([-1, 3])
    if len(data['people'])==2:
        body_0 = np.array(data['people'][0]['pose_keypoints_2d'],
                                    dtype=np.float32).reshape([-1, 3])
        body_1 = np.array(data['people'][1]['pose_keypoints_2d'],
                                    dtype=np.float32).reshape([-1, 3])
        if rgbd_id=="rgbd0":
            x_body_0 = np.mean(body_0[:,0])
            x_body_1 = np.mean(body_1[:,0])
            # lx: read the sub1 left person
            if is_sub1=="sub1":
                body_keypoints = body_0 if x_body_0<x_body_1 else body_1
            else: # read the sub2 right person
                body_keypoints = body_1 if x_body_0<x_body_1 else body_0
            
        else: # rgbd1
            #lx: find the highest-score one
            y_l_body_0, y_u_body_0 = np.min(body_0[:,1]),np.max(body_0[:,1])
            y_l_body_1, y_u_body_1 = np.min(body_1[:,1]),np.max(body_1[:,1])
            body_keypoints = body_0 if (y_u_body_0 - y_l_body_0)>(y_u_body_1 - y_l_body_1) else body_1
                
    # person_data = data['people'][0]
    # body_keypoints = np.array(person_data['pose_keypoints_2d'],
    #                             dtype=np.float32)
    # body_keypoints = body_keypoints.reshape([-1, 3])

    return body_keypoints

def resize_points(points_arr, num_pts):
    '''
    Either randomly subsamples or pads the given points_arr to be of the desired size.
    - points_arr : N x 3
    - num_pts : desired num point 
    '''
    is_torch = isinstance(points_arr, torch.Tensor)
    N = points_arr.size(0) if is_torch else points_arr.shape[0]
    if N > num_pts:
        samp_inds = np.random.choice(np.arange(N), size=num_pts, replace=False)
        points_arr = points_arr[samp_inds]
    elif N < num_pts:
        while N < num_pts:
            pad_size = num_pts - N
            if is_torch:
                points_arr = torch.cat([points_arr, points_arr[:pad_size]], dim=0)
                N = points_arr.size(0)
            else:
                points_arr = np.concatenate([points_arr, points_arr[:pad_size]], axis=0)
                N = points_arr.shape[0]
    return points_arr

def compute_plane_intersection(point, direction, plane):
    '''
    Given a ray defined by a point in space and a direction, compute the intersection point with the given plane.
    Detect intersection in either direction or -direction so the given ray may not actually intersect with the plane.

    Returns the intersection point as well as s such that point + s*direction = intersection_point. if s < 0 it means
    -direction intersects.

    - point : B x 3
    - direction : B x 3
    - plane : B x 4 (a, b, c, d) where (a, b, c) is the normal and (d) the offset.
    '''
    plane_normal = plane[:,:3]
    plane_off = plane[:,3]
    s = (plane_off - bdot(plane_normal, point)) / bdot(plane_normal, direction)
    itsct_pt = point + s.reshape((-1, 1))*direction
    return itsct_pt, s

def bdot(A1, A2, keepdim=False):
    ''' 
    Batched dot product.
    - A1 : B x D
    - A2 : B x D.
    Returns B.
    '''
    return (A1*A2).sum(dim=-1, keepdim=keepdim) 

def parse_floor_plane(floor_plane):
    '''
    Takes floor plane in the optimization form (Bx3 with a,b,c * d) and parses into
    (a,b,c,d) from with (a,b,c) normal facing "up in the camera frame and d the offset.
    '''
    floor_offset = torch.norm(floor_plane, dim=1, keepdim=True)
    floor_normal = floor_plane / floor_offset
    
    # in camera system -y is up, so floor plane normal y component should never be positive
    #       (assuming the camera is not sideways or upside down)
    neg_mask = floor_normal[:,1:2] > 0.0
    floor_normal = torch.where(neg_mask.expand_as(floor_normal), -floor_normal, floor_normal)
    floor_offset = torch.where(neg_mask, -floor_offset, floor_offset)
    floor_plane_4d = torch.cat([floor_normal, floor_offset], dim=1)

    return floor_plane_4d

def load_planercnn_res(res_path):
    '''
    Given a directory containing PlaneRCNN plane detection results, loads the first image result 
    and heuristically finds and returns the floor plane.
    '''
    planes_param_path = glob.glob(res_path + '/*_plane_parameters_*.npy')[0]
    planes_mask_path = glob.glob(res_path + '/*_plane_masks_*.npy')[0]
    planes_params = np.load(planes_param_path)
    planes_masks = np.load(planes_mask_path)
    
    # heuristically determine the ground plane
    #   the plane with the most labeled pixels in the bottom N rows
    nrows = 10
    label_count = np.sum(planes_masks[:, -nrows:, :], axis=(1, 2))
    floor_idx = np.argmax(label_count)
    valid_floor = False
    floor_plane = None
    while not valid_floor:
        # loop until we find a plane with many pixels on the bottom
        #       and doesn't face in the complete wrong direction
        # we assume the y component is larger than any others
        # i.e. that the floor is not > 45 degrees relative rotation from the camera
        floor_plane = planes_params[floor_idx]
        # transform to our system
        floor_plane = np.array([floor_plane[0], -floor_plane[2], floor_plane[1]])
        # determine 4D parameterization
        # for this data we know y should always be negative
        floor_offset = np.linalg.norm(floor_plane)
        floor_normal = floor_plane / floor_offset
        if floor_normal[1] > 0.0:
            floor_offset *= -1.0
            floor_normal *= -1.0
        a, b, c = floor_normal
        d = floor_offset
        floor_plane = np.array([a, b, c, d])

        valid_floor = np.abs(b) > np.abs(a) and np.abs(b) > np.abs(c)
        if not valid_floor:
            label_count[floor_idx] = 0
            floor_idx = np.argmax(label_count)

    return floor_plane


def compute_cam2prior(floor_plane, trans, root_orient, joints):
    '''
    Computes rotation and translation from the camera frame to the canonical coordinate system
    used by the motion and initial state priors.
    - floor_plane : B x 3
    - trans : B x 3
    - root_orient : B x 3
    - joints : B x J x 3
    '''
    B = floor_plane.size(0)
    if floor_plane.size(1) == 3:
        floor_plane_4d = parse_floor_plane(floor_plane)
    else:
        floor_plane_4d = floor_plane
    floor_normal = floor_plane_4d[:,:3]
    floor_trans, _ = compute_plane_intersection(trans, -floor_normal, floor_plane_4d)

    # compute prior frame axes within the camera frame
    # up is the floor_plane normal
    up_axis = floor_normal
    # right is body -x direction projected to floor plane
    root_orient_mat = batch_rodrigues(root_orient)
    body_right = -root_orient_mat[:, :, 0]
    floor_body_right, s = compute_plane_intersection(trans, body_right, floor_plane_4d)
    right_axis = floor_body_right - floor_trans 
    # body right may not actually intersect - in this case must negate axis because we have the -x
    right_axis = torch.where(s.reshape((B, 1)) < 0, -right_axis, right_axis)
    right_axis = right_axis / torch.norm(right_axis, dim=1, keepdim=True)
    # forward is their cross product
    fwd_axis = torch.cross(up_axis, right_axis)
    fwd_axis = fwd_axis / torch.norm(fwd_axis, dim=1, keepdim=True)

    prior_R = torch.stack([right_axis, fwd_axis, up_axis], dim=2)
    cam2prior_R = prior_R.transpose(2, 1)

    # translation takes translation to origin plus offset to the floor
    cam2prior_t = -trans

    _, s_root = compute_plane_intersection(joints[:,0], -floor_normal, floor_plane_4d)
    root_height = s_root.reshape((B, 1))

    return cam2prior_R, cam2prior_t, root_height

def apply_robust_weighting(res, robust_loss_type='bisquare', robust_tuning_const=4.6851):
    '''
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    '''
    robust_choices = ['none', 'bisquare']
    if robust_loss_type not in robust_choices:
        print('Not a valid robust loss: %s. Please use %s' % (robust_loss_type, str(robust_choices)))
    
    w = None
    detach_res = res.clone().detach() # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == 'none':
        w = torch.ones_like(detach_res)
    elif robust_loss_type == 'bisquare':
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w

def robust_std(res):
    ''' 
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    '''
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B,1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std

def bisquare_robust_weights(res, tune_const=4.6851):
    '''
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    '''
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    # NOTE: this should use absolute value, it's ok right now since only used for 3d point cloud residuals
        #   which are guaranteed positive, but generally this won't work)
    outlier_mask = norm_res >= 1.0

    # print(torch.sum(outlier_mask))
    # print('Outlier frac: %f' % (float(torch.sum(outlier_mask)) / res.size(1)))

    w = (1.0 - norm_res**2)**2
    w[outlier_mask] = 0.0

    return w

def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def log_cur_stats(stats_dict, loss, iter=None):
    Logger.log('LOSS: %f' % (loss.cpu().item()))
    print('----')
    for k, v in stats_dict.items():
        if isinstance(v, float):
            Logger.log('%s: %f' % (k, v))
        else:
            Logger.log('%s: %f' % (k, v.cpu().item()))
    if iter is not None:
        print('======= iter %d =======' % (int(iter)))
    else:
        print('========')

def save_optim_result(cur_res_out_paths, optim_result, per_stage_results, gt_data, observed_data, data_type,
                      optim_floor=True,
                      obs_img_paths=None,
                      obs_mask_paths=None):
    # final optim results
    res_betas = optim_result['betas'].cpu().numpy()
    res_trans = optim_result['trans'].cpu().numpy()
    res_root_orient = optim_result['root_orient'].cpu().numpy()
    res_body_pose = optim_result['pose_body'].cpu().numpy()
    res_hand_pose = optim_result['pose_hand'].cpu().numpy()
    res_contacts = None
    res_floor_plane = None
    if 'contacts' in optim_result:
        res_contacts = optim_result['contacts'].cpu().numpy()
    if 'floor_plane' in optim_result:
        res_floor_plane = optim_result['floor_plane'].cpu().numpy()
    for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
        cur_res_out_path = os.path.join(cur_res_out_path, 'stage2_results.npz')
        save_dict = { 
            'betas' : res_betas[bidx],
            'trans' : res_trans[bidx],
            'root_orient' : res_root_orient[bidx],
            'pose_body' : res_body_pose[bidx],
        }
        if res_hand_pose is not None:
            save_dict['pose_hand'] = res_hand_pose[bidx]
        if res_contacts is not None:
            save_dict['contacts'] = res_contacts[bidx]
        if res_floor_plane is not None:
            save_dict['floor_plane'] = res_floor_plane[bidx]
        np.savez(cur_res_out_path, **save_dict)

    # in prior coordinate frame
    if 'stage3' in per_stage_results and optim_floor:
        res_trans = per_stage_results['stage3']['prior_trans'].detach().cpu().numpy()
        res_root_orient = per_stage_results['stage3']['prior_root_orient'].detach().cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'stage3_results_prior.npz')
            save_dict = { 
                'betas' : res_betas[bidx],
                'trans' : res_trans[bidx],
                'root_orient' : res_root_orient[bidx],
                'pose_body' : res_body_pose[bidx]
            }
            if res_contacts is not None:
                save_dict['contacts'] = res_contacts[bidx]
            np.savez(cur_res_out_path, **save_dict)

    # ground truth
    save_gt = 'betas' in gt_data and \
                'trans' in gt_data and \
                'root_orient' in gt_data and \
                'pose_body' in gt_data
    if save_gt:
        gt_betas = gt_data['betas'].cpu().numpy()
        if data_type not in ['PROX-RGB', 'PROX-RGBD']:
            gt_betas = gt_betas[:,0] # only need frame 1 for e.g. 3d data since it's the same over time.
        gt_trans = gt_data['trans'].cpu().numpy()
        gt_root_orient = gt_data['root_orient'].cpu().numpy()
        gt_body_pose = gt_data['pose_body'].cpu().numpy()
        gt_contacts = None
        if 'contacts' in gt_data:
            gt_contacts = gt_data['contacts'].cpu().numpy()
        cam_mat = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            gt_res_name = 'proxd_results.npz' if data_type in ['PROX-RGB', 'PROX-RGBD'] else 'gt_results.npz'
            cur_gt_out_path = os.path.join(cur_res_out_path, gt_res_name)
            save_dict = { 
                'betas' : gt_betas[bidx],
                'trans' : gt_trans[bidx],
                'root_orient' : gt_root_orient[bidx],
                'pose_body' : gt_body_pose[bidx]
            }
            if gt_contacts is not None:
                save_dict['contacts'] = gt_contacts[bidx]
            if cam_mat is not None:
                save_dict['cam_mtx'] = cam_mat[bidx]
            np.savez(cur_gt_out_path, **save_dict)

            # if these are proxd results also need to save a GT with cam matrix
            if data_type in ['PROX-RGB', 'PROX-RGBD']:
                cur_gt_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
                np.savez(cur_gt_out_path, cam_mtx=cam_mat[bidx])

    elif 'joints3d' in gt_data:
        # don't have smpl params, but have 3D joints (e.g. imapper)
        gt_joints = gt_data['joints3d'].cpu().numpy()
        cam_mat = occlusions = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].cpu().numpy()
        if 'occlusions' in gt_data:
            occlusions = gt_data['occlusions'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
            save_dict = { 
                'joints3d' : gt_joints[bidx]
            }
            if cam_mat is not None:
                save_dict['cam_mtx'] = cam_mat[bidx]
            if occlusions is not None:
                save_dict['occlusions'] = occlusions[bidx]
            np.savez(cur_res_out_path, **save_dict)
    elif 'cam_matx' in gt_data:
        # need the intrinsics even if we have nothing else
        cam_mat = gt_data['cam_matx'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
            save_dict = { 
                'cam_mtx' : cam_mat[bidx]
            }
            np.savez(cur_res_out_path, **save_dict)

    # observations
    obs_out = {k : v.cpu().numpy() for k, v in observed_data.items() if k != 'prev_batch_overlap_res'}
    for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
        obs_out_path = os.path.join(cur_res_out_path, 'observations.npz')
        cur_obs_out = {k : v[bidx] for k, v in obs_out.items() if k not in ['RGB']}
        if obs_img_paths is not None:
            cur_obs_out['img_paths'] = [frame_tup[bidx] for frame_tup in obs_img_paths]
            # print(cur_obs_out['img_paths'])
        if obs_mask_paths is not None:
            cur_obs_out['mask_paths'] = [frame_tup[bidx] for frame_tup in obs_mask_paths]
        np.savez(obs_out_path, **cur_obs_out)    


# lx: visualize the overlap rendering results
from motion_modeling.util_tool.viz import viz_smpl_seq, viz_smpl_seq_multiperson, create_video, create_multi_comparison_images
from matplotlib.image import imread
def vis_results_lx(cur_res_out_paths, optim_result, per_stage_results, gt_data, observed_data, data_type,
                      device,
                      optim_floor=True,
                      obs_img_paths=None,
                      obs_mask_paths=None,
                      args=None):
    # load in image frames
    T = len(obs_img_paths)
    D_IMH, D_IMW = 720,1280
    IMW, IMH = None, None
    img_arr = np.zeros((T, D_IMH, D_IMW, 3), dtype=np.float32)
    for imidx, img_path in enumerate(obs_img_paths):
        img = cv2.imread(str(img_path[0]))
        IMH, IMW, _ = img.shape
        # img = cv2.resize(img, (D_IMW, D_IMH), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:, :, ::-1] / 255.0
        img_arr[imidx] = img
    # get camera intrinsics
    cam_fx = gt_data['cam_matx'][0,0, 0]
    cam_fy = gt_data['cam_matx'][0,1, 1]
    cam_cx = gt_data['cam_matx'][0,0, 2]
    cam_cy = gt_data['cam_matx'][0,1, 2]
    cam_intrins = (cam_fx, cam_fy, cam_cx, cam_cy)
   
   
    # print(cam_intrins)
    x_frac = float(D_IMW) / IMW
    y_frac = float(D_IMH) / IMH
    cam_intrins_down = (cam_fx*x_frac, cam_fy*y_frac, cam_cx*x_frac, cam_cy*y_frac)
    #
    # Qualitative evaluation
    #
    cur_qual_out_path = cur_res_out_paths[0]
    mkdir(cur_qual_out_path)
    
    
    # humor model
    cur_meta_path = os.path.join(cur_res_out_paths[0], 'meta.txt')
    optim_bm_path = gt_bm_path = None
    with open(cur_meta_path, 'r') as f:
        optim_bm_str = f.readline().strip()
        optim_bm_path = optim_bm_str.split(' ')[1]
        gt_bm_str = f.readline().strip()
        gt_bm_path = gt_bm_str.split(' ')[1]
    pred_bm = None
    pred_bm = BodyModel(bm_path=optim_bm_path,
                        num_betas=optim_result['betas'].size(1),
                        batch_size=T).to(device)
    # recover and render body model:
    cur_stages_res = None
    cur_stages_res = dict()
    STAGES_RES_NAMES = [ 'stage2_results']
    for stage_name in  STAGES_RES_NAMES:
        stage_res = load_res(cur_qual_out_path, stage_name + '.npz')
        if stage_res is None:
            print('Could not find results for stage %s of %s, skipping...' % (stage_name, cur_qual_out_path))
            continue
        cur_stages_res[stage_name] = prep_res(stage_res, device, T)
    
    # get camera extrinsic file
    take_folder = str((Path(args.data_path)).parent.parent)
    extri_file = osp.join(take_folder,"CamExtr.txt")
    cam_RT = torch.from_numpy(np.loadtxt(extri_file)).float().to('cuda')
    cam_RT = cam_RT.repeat(optim_result['root_orient'].shape[1],1,1)
    rotation = cam_RT[:,:3,:3]
    translation = cam_RT[:,:3,3]

    stages_body = None
    if per_stage_results is not None:
        stages_body = dict()
        for k, v in cur_stages_res.items():

            stages_body[k] = run_smpl(v, pred_bm)
            points =  stages_body[k].v
            
            # world to camera
            points = torch.einsum('bij,bkj->bki', rotation, points)
            points = points + translation.unsqueeze(1)
            stages_body[k].v = points
            
            
            # get body smpl joints
            stage_body_joints = stages_body[k].Jtr[:, :22]
            stage_body_joints = torch.einsum('bij,bkj->bki', rotation, stage_body_joints)
            stage_body_joints = stage_body_joints + translation.unsqueeze(1)
            cur_stages_res[k]['joints3d_smpl'] = stage_body_joints
    IM_EXTN = "jpg"
    FPS = args.fps
    for k, stage_body in stages_body.items():
        stage_out_path = os.path.join(cur_qual_out_path, k)
        viz_smpl_seq(stage_body, imw=D_IMW, imh=D_IMH, fps=FPS,
                    render_body=True,
                    render_bodies_static=None,
                    render_joints=False,
                    render_skeleton=False,
                    render_ground=False,
                    ground_plane=False,
                    ground_alpha=1.0,
                    body_alpha=None,
                    static_meshes=None,
                    points_seq=None,
                    point_color=[0.0, 1.0, 0.0],
                    use_offscreen=True,
                    out_path=stage_out_path,
                    wireframe=False,
                    RGBA=True,
                    point_rad=0.004,
                    follow_camera=False,
                    camera_intrinsics=cam_intrins_down,
                    img_seq=img_arr,
                    mask_seq=None,
                    img_extn=IM_EXTN)
        create_video(stage_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), stage_out_path + '.mp4', FPS)
        # shutil.rmtree(stage_out_path)

def vis_results_twosubjs(take_folder, device='cuda'):
    
    # get camera intrinsics
    intri_file = osp.join(take_folder,"CamIntr.txt")
    cam_intri = np.loadtxt(intri_file)
    cam_fx = cam_intri[0, 0]
    cam_fy = cam_intri[1, 1]
    cam_cx = cam_intri[0, 2]
    cam_cy = cam_intri[1, 2]
    cam_intrins = (cam_fx, cam_fy, cam_cx, cam_cy)
    
  
    D_IMH, D_IMW = 720,1280
    IMW, IMH = 1280, 720
    # print(cam_intrins)
    x_frac = float(D_IMW) / IMW
    y_frac = float(D_IMH) / IMH
    cam_intrins_down = (cam_fx*x_frac, cam_fy*y_frac, cam_cx*x_frac, cam_cy*y_frac)

    # out folder
    cur_qual_out_path = os.path.join(take_folder,'processed/rgbd0/humor_out_two_subjs')
    mkdir(cur_qual_out_path)
    
    # human model
    cur_meta_path = os.path.join(take_folder,'processed/rgbd0/humor_out_v3_sub2/results_out/meta.txt')
    optim_bm_path = gt_bm_path = None
    with open(cur_meta_path, 'r') as f:
        optim_bm_str = f.readline().strip()
        optim_bm_path = optim_bm_str.split(' ')[1]
        gt_bm_str = f.readline().strip()
        gt_bm_path = gt_bm_str.split(' ')[1]
    pred_bm = None
    T = 200
    pred_bm = BodyModel(bm_path=optim_bm_path,
                        num_betas=16,
                        batch_size=T).to(device)
    
    # recover and render body model:
    cur_stages_res = None
    cur_stages_res = dict()
    subj_folders = [ 'humor_out_v3','humor_out_v3_sub2']
    for subj in  subj_folders:
        folder = osp.join(take_folder,'processed/rgbd0',subj,'results_out')
        stage_res = load_res(folder, 'stage2_results.npz')
        if stage_res is None:
            print('Could not find results for stage %s of %s, skipping...' % (subj, folder))
            continue
        cur_stages_res[subj] = prep_res(stage_res, device, T)
    
    # get camera extrinsic file
    extri_file = osp.join(take_folder,"CamExtr.txt")
    cam_RT = torch.from_numpy(np.loadtxt(extri_file)).float().to('cuda')
    cam_RT = cam_RT.repeat(cur_stages_res['humor_out_v3']['root_orient'].shape[0],1,1)
    rotation = cam_RT[:,:3,:3]
    translation = cam_RT[:,:3,3]
    
    stages_body = None
    stages_body = dict()
    for k, v in cur_stages_res.items():
        if k == 'humor_out_v3_sub2':
            v['pose_hand'] = None
        stages_body[k] = run_smpl(v, pred_bm)
        points =  stages_body[k].v
        
        # world to camera
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)
        stages_body[k].v = points
        
        
        # get body smpl joints
        stage_body_joints = stages_body[k].Jtr[:, :22]
        stage_body_joints = torch.einsum('bij,bkj->bki', rotation, stage_body_joints)
        stage_body_joints = stage_body_joints + translation.unsqueeze(1)
        cur_stages_res[k]['joints3d_smpl'] = stage_body_joints


    IM_EXTN = "jpg"
    FPS = 60
    stage_out_path = os.path.join(cur_qual_out_path, 'humor_out_two_subjects')
    viz_smpl_seq_multiperson(stages_body, 
                             imw=D_IMW, imh=D_IMH, fps=FPS,
                    render_body=True,
                    render_bodies_static=None,
                    render_joints=False,
                    render_skeleton=False,
                    render_ground=False,
                    ground_plane=False,
                    ground_alpha=1.0,
                    body_alpha=None,
                    static_meshes=None,
                    points_seq=None,
                    point_color=[0.0, 1.0, 0.0],
                    use_offscreen=True,
                    out_path=stage_out_path,
                    wireframe=False,
                    RGBA=False,
                    point_rad=0.004,
                    follow_camera=False,
                    camera_intrinsics=cam_intrins_down,
                    img_extn=IM_EXTN)
    create_video(stage_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), stage_out_path + '.mp4', FPS)
    # shutil.rmtree(stage_out_path)        

def save_rgb_stitched_result(seq_intervals, all_res_out_paths, res_out_path, device,
                                body_model_path, num_betas, use_joints2d):
    import cv2
    seq_overlaps = [0]
    for int_idx in range(len(seq_intervals)-1):
        prev_end = seq_intervals[int_idx][1]
        cur_start = seq_intervals[int_idx+1][0]
        seq_overlaps.append(prev_end - cur_start)

    # if arbitray RGB video data, stitch together to save full sequence output
    all_res_dirs = all_res_out_paths
    print(all_res_dirs)

    final_res_out_path = os.path.join(res_out_path, 'final_results')
    mkdir(final_res_out_path)

    concat_cam_res = None
    concat_contacts = None
    concat_ground_planes = None
    concat_joints2d = None
    concat_img_paths = None
    gt_cam_mtx = None
    for res_idx, res_dir in enumerate(all_res_dirs):
        # camera view
        cur_stage3_res = load_res(res_dir, 'stage3_results.npz')
        cur_contacts = torch.Tensor(cur_stage3_res['contacts']).to(device)
        if concat_ground_planes is None: 
            concat_ground_planes = torch.Tensor(cur_stage3_res['floor_plane']).to(device).reshape((1, -1))
        else:
            concat_ground_planes = torch.cat([concat_ground_planes, torch.Tensor(cur_stage3_res['floor_plane']).to(device).reshape((1, -1))], dim=0)
        cur_stage3_res = {k : v for k, v in cur_stage3_res.items() if k in ['betas', 'trans', 'root_orient', 'pose_body']}
        cur_stage3_res = prep_res(cur_stage3_res, device, cur_stage3_res['trans'].shape[0])
        if concat_cam_res is None: 
            concat_cam_res = cur_stage3_res
            concat_contacts = cur_contacts
        else:
            for k, v in concat_cam_res.items():
                concat_cam_res[k] = torch.cat([concat_cam_res[k], cur_stage3_res[k][seq_overlaps[res_idx]:]], dim=0)
            concat_contacts = torch.cat([concat_contacts, cur_contacts[seq_overlaps[res_idx]:]], dim=0)

        # gt
        if gt_cam_mtx is None:
            gt_res = load_res(res_dir, 'gt_results.npz')
            gt_cam_mtx = gt_res['cam_mtx']

        # obs
        cur_obs = load_res(res_dir, 'observations.npz')
        if concat_joints2d is None:
            concat_joints2d = cur_obs['joints2d']
        else:
            concat_joints2d = np.concatenate([concat_joints2d, cur_obs['joints2d'][seq_overlaps[res_idx]:]], axis=0)
        if concat_img_paths is None:
            concat_img_paths = list(cur_obs['img_paths'])
        else:
            concat_img_paths = concat_img_paths + list(cur_obs['img_paths'][seq_overlaps[res_idx]:])
        
        # ignore if we don't have an interval for this directory (was an extra due to even batching requirement)
        if res_idx >= len(seq_overlaps):
            break

    # copy meta
    src_meta_path = os.path.join(all_res_dirs[0], 'meta.txt')
    shutil.copyfile(src_meta_path, os.path.join(final_res_out_path, 'meta.txt'))

    #  gt results (cam matx)
    np.savez(os.path.join(final_res_out_path, 'gt_results.npz'), cam_mtx=gt_cam_mtx)

    # obs results (joints2d and img_paths)
    np.savez(os.path.join(final_res_out_path, 'observations.npz'), joints2d=concat_joints2d, img_paths=concat_img_paths)

    #  save the actual results npz for viz later
    concat_res_out_path = os.path.join(final_res_out_path, 'stage3_results.npz')
    res_betas = concat_cam_res['betas'].clone().detach().cpu().numpy()
    res_trans = concat_cam_res['trans'].clone().detach().cpu().numpy()
    res_root_orient = concat_cam_res['root_orient'].clone().detach().cpu().numpy()
    res_body_pose = concat_cam_res['pose_body'].clone().detach().cpu().numpy()
    res_floor_plane = concat_ground_planes[0].clone().detach().cpu().numpy() # NOTE: saves estimate from first subsequence
    res_contacts = concat_contacts.clone().detach().cpu().numpy()
    np.savez(concat_res_out_path, betas=res_betas,
                                trans=res_trans,
                                root_orient=res_root_orient,
                                pose_body=res_body_pose,
                                floor_plane=res_floor_plane,
                                contacts=res_contacts)

    # get body model
    num_viz_frames = concat_cam_res['trans'].size(0)
    viz_body_model = BodyModel(bm_path=body_model_path,
                            num_betas=num_betas,
                            batch_size=num_viz_frames,
                            use_vtx_selector=use_joints2d).to(device)
    viz_body = run_smpl(concat_cam_res, viz_body_model)
    
    # transform full camera-frame sequence into a shared prior frame based on a single ground plane
    viz_joints3d = viz_body.Jtr
    # compute the transformation based on t=0 and the first sequence floor plane
    cam2prior_R, cam2prior_t, cam2prior_root_height = compute_cam2prior(concat_ground_planes[0].unsqueeze(0),
                                                                        concat_cam_res['trans'][0].unsqueeze(0),
                                                                        concat_cam_res['root_orient'][0].unsqueeze(0),
                                                                        viz_joints3d[0].unsqueeze(0))
    # transform the whole sequence
    input_data_dict = {kb : vb.unsqueeze(0) for kb, vb in concat_cam_res.items() if kb in ['trans', 'root_orient', 'pose_body', 'betas']}
    viz_prior_data_dict = apply_cam2prior(input_data_dict, cam2prior_R, cam2prior_t, cam2prior_root_height, 
                                            input_data_dict['pose_body'],
                                            input_data_dict['betas'],
                                            0,
                                            viz_body_model)
    concat_prior_res = {
        'trans' : viz_prior_data_dict['trans'][0],
        'root_orient' : viz_prior_data_dict['root_orient'][0],
        'pose_body' : concat_cam_res['pose_body'],
        'betas' : concat_cam_res['betas']
    }

    # save pose prior frame
    concat_prior_res_out_path = os.path.join(final_res_out_path, 'stage3_results_prior.npz')
    res_betas = concat_prior_res['betas'].clone().detach().cpu().numpy()
    res_trans = concat_prior_res['trans'].clone().detach().cpu().numpy()
    res_root_orient = concat_prior_res['root_orient'].clone().detach().cpu().numpy()
    res_body_pose = concat_prior_res['pose_body'].clone().detach().cpu().numpy()
    res_contacts = concat_contacts.clone().detach().cpu().numpy()
    np.savez(concat_prior_res_out_path, betas=res_betas,
                                trans=res_trans,
                                root_orient=res_root_orient,
                                pose_body=res_body_pose,
                                contacts=res_contacts)


def load_res(result_dir, file_name):
    '''
    Load np result from our model or GT
    '''
    res_path = os.path.join(result_dir, file_name)
    if not os.path.exists(res_path):
        return None
    res = np.load(res_path)
    res_dict = {k : res[k] for k in res.files}
    return res_dict

def prep_res(np_res, device, T):
    '''
    Load np result dict into dict of torch objects for use with SMPL body model.
    '''
    betas = np_res['betas']
    betas = torch.Tensor(betas).to(device)
    if len(betas.size()) == 1:
        num_betas = betas.size(0)
        betas = betas.reshape((1, num_betas)).expand((T, num_betas))
    else:
        num_betas = betas.size(1)
        assert(betas.size(0) == T)
    trans = np_res['trans']
    trans = torch.Tensor(trans).to(device)
    root_orient = np_res['root_orient']
    root_orient = torch.Tensor(root_orient).to(device)
    pose_body = np_res['pose_body']
    pose_body = torch.Tensor(pose_body).to(device)
    pose_hand = np_res['pose_hand']
    pose_hand = torch.Tensor(pose_hand).to(device)

    res_dict = {
        'betas' : betas,
        'trans' : trans,
        'root_orient' : root_orient,
        'pose_body' : pose_body,
        'pose_hand' : pose_hand
    }

    for k, v in np_res.items():
        if k not in ['betas', 'trans', 'root_orient', 'pose_body','pose_hand']:
            res_dict[k] = v
    return res_dict

def run_smpl(res_dict, body_model):
    smpl_body = body_model(pose_body=res_dict['pose_body'], 
                            pose_hand=res_dict['pose_hand'], 
                            betas=res_dict['betas'],
                            root_orient=res_dict['root_orient'],
                            trans=res_dict['trans'],)
                            # left_hand_pose=res_dict['pose_hand'][:,:45],
                            # right_hand_pose=res_dict['pose_hand'][:,45:])
    
    return smpl_body

def apply_cam2prior(data_dict, R, t, root_height, body_pose, betas, key_frame_idx, body_model, inverse=False):
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
            cur_smpl_body = body_model(pose_body=body_pose.reshape((-1, body_pose.size(2))), 
                                    pose_hand=None, 
                                    betas=betas.reshape((-1, betas.size(2))),
                                    root_orient=prior_dict['root_orient'].reshape((-1, 3)),
                                    trans=trans.reshape((-1, 3)))
            smpl_joints3d = cur_smpl_body.Jtr.reshape((B, T, -1, 3))
            if T > 1:
                cur_root_height = smpl_joints3d[np.arange(B),key_frame_idx,0,2:3]
            else:
                cur_root_height = smpl_joints3d[:,0,0,2:3]
            height_diff = root_height - cur_root_height
            trans_offset = torch.cat([torch.zeros((B, 2)).to(height_diff), height_diff], axis=1)
            trans = trans + trans_offset.reshape((B, 1, 3))
        prior_dict['trans'] = trans
    elif 'trans' in data_dict:
        Logger.log('Cannot apply cam2prior on translation without root orient data!')
        exit()

    return prior_dict

# lx: if error, need to check 
def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    Adapted from https://github.com/mkocabas/VIBE/blob/master/lib/models/spin.py
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs, 2): Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

OP_NUM_JOINTS = 25
OP_IGNORE_JOINTS = [1, 9, 12] # neck and left/right hip
OP_EDGE_LIST = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
# indices to map an openpose detection to its flipped version
OP_FLIP_MAP = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]

#
# The following 2 functions are borrowed from VPoser (https://github.com/nghorbani/human_body_prior).
# See their license for usage restrictions.
#
def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_vposer(expr_dir, vp_model='snapshot'):
    '''
    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if vp_model == 'snapshot':

        vposer_path = sorted(glob.glob(os.path.join(expr_dir, 'vposer_*.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('VPoser', vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        vposer_pt = getattr(module, 'VPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        vposer_pt = vp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    vposer_pt.eval()

    return vposer_pt, ps

if __name__=='__main__':
    # root_folders = ["/home/ur-5/Projects/justlx/website_data/home_data",
    #                 "/home/ur-5/Projects/justlx/website_data/case_data"]
    # root_out = "/home/ur-5/Projects/justlx/website_data/out_videos"
    # if not os.path.exists(root_out):
    #     os.makedirs(root_out, exist_ok=True)
    # # take_folders = [
    # # "/home/ur-5/Projects/justlx/website_data/case_data/home_data/001078",
    # #  "/home/ur-5/Projects/justlx/website_data/case_data/home_data/008699",
    # #  "/home/ur-5/Projects/justlx/website_data/case_data/home_data/009823",
    # #  "/home/ur-5/Projects/justlx/website_data/case_data/home_data/010045",
    # #  "/home/ur-5/Projects/justlx/website_data/case_data/002870",
    # #  "/home/ur-5/Projects/justlx/website_data/case_data/002873",
    # #  ]
    # for root in root_folders:
    #     take_folders = os.listdir(root)
    #     take_folders.sort()
    #     for take in take_folders:
    #         if take !="008699" and take !="010045":
    #             continue
    #         take_folder = os.path.join(root, take)
    #         orig_images_folder = os.path.join(take_folder, 'processed/rgbd0')
    #         two_subjs_folder = os.path.join(take_folder, 'processed/rgbd0/humor_out_two_subjs')
    #         out_folder = os.path.join(root_out, take, 'images')
    #         # create extracted two subj's motion video
    #         if not os.path.exists(os.path.join(two_subjs_folder, 'humor_out_two_subjects.mp4')):
    #             vis_results_twosubjs(take_folder)
    #         # create combined original images and extracted motion video
    #         create_comparison_images_vert_lx(img1_dir=orig_images_folder, img2_dir=two_subjs_folder+'/humor_out_two_subjects', out_dir=out_folder, text1=None, text2=None,extn1='*.png',extn2='*.jpg',frame_num=200)
    #         create_video(out_folder + '/frame_%08d.jpg', root_out+'/'+ take + '.mp4', 60)
    
    # combine home videos
    img_dirs = [
        "/home/ur-5/Projects/justlx/website_data/out_videos/008699/images",
        "/home/ur-5/Projects/justlx/website_data/out_videos/010045/images",
        "/home/ur-5/Projects/justlx/website_data/out_videos/002873/images",
        "/home/ur-5/Projects/justlx/website_data/out_videos/005916/images",
    ]
    out_dir = "/home/ur-5/Projects/justlx/website_data/out_videos"
    out_dir_images = "/home/ur-5/Projects/justlx/website_data/out_videos/home_images"
    IM_EXTN = 'jpg'
    create_multi_comparison_images(img_dirs, 
                            out_dir_images,
                            None,
                            extn=IM_EXTN)
    create_video(out_dir_images + '/frame_%08d.' + '%s' % (IM_EXTN), out_dir + 'home_video.mp4', 60)