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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

smplh_model = SMPLH(constants.SMPLH_MODEL_PATH, \
        gender='male',\
            batch_size = 1,\
                flat_hand_mean=True,\
                 use_pca=False, 
                ).to(device)

# run smplh deformation
with torch.no_grad():
    pred_output = smplh_model()
pred_vertices = pred_output.vertices
pred_vertices = pred_vertices.cpu().numpy()
    
faces=smplh_model.faces
verts0 = np.array(pred_vertices[0])
mesh0 = trimesh.Trimesh(verts0, faces)
    
# save mesh0
filename =  "smpl_restpose_body.obj"                                                           
out_mesh_path = osp.join('./', filename)
mesh0.export(out_mesh_path)