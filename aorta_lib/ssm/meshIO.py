import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
import io


def pv2torch3d(pvMesh):
    nPointsPerFace = pvMesh.faces[0]
    faces = pvMesh.faces.reshape((-1, nPointsPerFace + 1))
    faces = faces[:, -nPointsPerFace:]
    return np.array(pvMesh.points), np.array(faces)

def createInputData(src_mesh_pv, trg_mesh_pv, device):

    trg_mesh_pv_temp = trg_mesh_pv.copy()
    src_mesh_pv_temp = src_mesh_pv.copy()

    mean = trg_mesh_pv_temp.points.mean()
    std = trg_mesh_pv_temp.points.std()

    trg_mesh_pv_temp.points = ( trg_mesh_pv_temp.points - mean ) / std
    src_mesh_pv_temp.points = ( src_mesh_pv_temp.points - mean ) / std

    verts, faces = pv2torch3d(trg_mesh_pv_temp)
    faces = torch.from_numpy(faces).to(device)
    verts = torch.from_numpy(verts).to(device).to(torch.float32)
    #trg_mesh = Meshes(verts=[verts], faces=[faces])
    trg_mesh = Pointclouds(points=[verts])

    verts, faces = pv2torch3d(src_mesh_pv_temp)
    faces = torch.from_numpy(faces).to(device)
    verts = torch.from_numpy(verts).to(device).to(torch.float32)
    #src_mesh = Meshes(verts=[verts], faces=[faces])
    src_mesh = Pointclouds(points=[verts])

    return src_mesh, trg_mesh, mean, std
