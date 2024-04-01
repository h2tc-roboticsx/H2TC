import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

import trimesh

from smplx import SMPL, SMPLH, SMPLX
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct
from smplx.lbs import (
    lbs, vertices2joints, batch_rodrigues, blend_shapes)
from typing import NewType
Tensor = NewType('Tensor', torch.Tensor)

SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

SMPLH_PATH = "./human_tools/smplh_humor"
SMPLX_PATH = './body_models/smplx'
SMPL_PATH = './body_models/smpl'
VPOSER_PATH = './body_models/vposer_v1_0'
male_bm_path = os.path.join(SMPLH_PATH, 'male/smplh_male.npz')
female_bm_path = os.path.join(SMPLH_PATH, 'female/smplh_female.npz')

smpl_connections = [[11, 8], [8, 5], [5, 2], [2, 0], [10, 7], [7, 4], [4, 1], [1, 0], 
                [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [12, 13], [13, 16], [16, 18], 
                [18, 20], [12, 14], [14, 17], [17, 19], [19, 21]]

# chosen virtual mocap markers that are "keypoints" to work with
KEYPT_VERTS = [4404, 920, 3076, 3169, 823, 4310, 1010, 1085, 4495, 4569, 6615, 3217, 3313, 6713,
            6785, 3383, 6607, 3207, 1241, 1508, 4797, 4122, 1618, 1569, 5135, 5040, 5691, 5636,
            5404, 2230, 2173, 2108, 134, 3645, 6543, 3123, 3024, 4194, 1306, 182, 3694, 4294, 744]

def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

#
# From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# Please see license for usage restrictions.
#
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps SMPL to OpenPose

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=10,
                 use_vtx_selector=False,
                 model_type='smplh',
                 flat_hand_mean=True):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        '''
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        data_struct = None
        if '.npz' in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate([data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM-B))], axis=-1) # super hacky way to let smplh use 16-size beta
        kwargs = {
                'model_type' : model_type,
                'data_struct' : data_struct,
                'num_betas': num_betas,
                'batch_size' : batch_size,
                'num_expression_coeffs' : num_expressions,
                'vertex_ids' : cur_vertex_ids,
                'use_pca' : False,
                'flat_hand_mean' : flat_hand_mean
        }
        assert(model_type in ['smpl', 'smplh', 'smplx'])
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        assert(dmpls is None)
        out_obj = self.bm(
                betas=betas,
                global_orient=root_orient,
                body_pose=pose_body,
                left_hand_pose=None if pose_hand is None else pose_hand[:,:(SMPLH.NUM_HAND_JOINTS*3)],
                right_hand_pose=None if pose_hand is None else pose_hand[:,(SMPLH.NUM_HAND_JOINTS*3):],
                transl=trans,
                expression=expression,
                jaw_pose=pose_jaw,
                leye_pose=None if pose_eye is None else pose_eye[:,:3],
                reye_pose=None if pose_eye is None else pose_eye[:,3:],
                return_full_pose=True,
                **kwargs
        )

        out = {
            'v' : out_obj.vertices,
            'f' : self.bm.faces_tensor,
            'betas' : out_obj.betas,
            'Jtr' : out_obj.joints,
            'pose_body' : out_obj.body_pose,
            'full_pose' : out_obj.full_pose
        }
        if self.model_type in ['smplh', 'smplx']:
            out['pose_hand'] = torch.cat([out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1)
        if self.model_type == 'smplx':
            out['pose_jaw'] = out_obj.jaw_pose
            out['pose_eye'] = pose_eye
        

        if not self.use_vtx_selector:
            # don't need extra joints
            out['Jtr'] = out['Jtr'][:,:self.num_joints+1] # add one for the root

        if not return_dict:
            out = Struct(**out)

        return out


    def get_global_transforms_foralljoints(self, betas, global_orient, body_pose, left_hand_pose=None, right_hand_pose=None):
        
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.bm.global_orient).squeeze(0)
        body_pose = (body_pose if body_pose is not None else self.bm.body_pose).squeeze(0)
        betas = betas if betas is not None else self.bm.betas
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.bm.left_hand_pose).squeeze(0)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.bm.right_hand_pose).squeeze(0)
        full_pose = torch.cat([global_orient, body_pose,
                               left_hand_pose,
                               right_hand_pose], dim=1)
        full_pose += self.bm.pose_mean
        
        batch_size = max(betas.shape[0], full_pose.shape[0])
        betas = betas.repeat((batch_size,1))
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        v_shaped = self.bm.v_template + blend_shapes(betas, self.bm.shapedirs)

        # Get the joints
        # NxJx3 array
        J = vertices2joints(self.bm.J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        # ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
                [batch_size, -1, 3, 3])


        # A = self.my_batch_rigid_transform(rot_mats, self.bm.parents, dtype=dtype)
        _, A = self.batch_rigid_transform(rot_mats, J, self.bm.parents, dtype=dtype)
        return A
    
    def batch_rigid_transform(
    self,
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
    ) -> Tensor:
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor BxN
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
        """

        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        # joints_homogen = F.pad(joints, [0, 0, 0, 1])

        # rel_transforms = transforms - F.pad(
        #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, transforms
    
    def my_batch_rigid_transform(self, rot_mats, parents, dtype=torch.dtype) -> torch.Tensor:
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        parents : torch.tensor BxN
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        transforms : torch.tensor BxNx4x4
            The global rigid transformations
            for all the joints
        """
        transforms_mat = rot_mats.reshape(-1, 3, 3)

if __name__ == "__main__":
    SMPLH_HUMOR_MODEL = "./human_tools/smplh_humor/male/model.npz"
    smpl_poses = torch.zeros([1,63]).to(torch.float32).to("cuda")
    root_orient = torch.zeros([1,3]).to(torch.float32).to("cuda")
    smpl_transl = torch.zeros([1,3]).to(torch.float32).to("cuda")
    smplh_model = BodyModel(SMPLH_HUMOR_MODEL, \
            num_betas=16, \
            batch_size = smpl_poses.shape[0],\
                flat_hand_mean = True,\
                ).to("cuda")
    with torch.no_grad():
        pred_output = smplh_model.bm(
                                body_pose=smpl_poses,
                                global_orient=root_orient,
                                transl=smpl_transl)
    verts = pred_output.vertices.cpu().numpy()
    faces = smplh_model.bm.faces
    
    n = len(verts)
    id = 0
    for ii in range(n):
        verts0 = np.array(verts[ii])
        mesh0 = trimesh.Trimesh(verts0, faces)
            
        # save mesh0
        fram_name =  str(ii)
        filename =  "smplh_rest_pose_normalhand.ply"                                                    
        out_mesh_path = filename
        mesh0.export(out_mesh_path)
    