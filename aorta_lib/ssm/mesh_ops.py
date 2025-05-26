import numpy as np
import torch
import pytorch3d
import pyvista as pv
import pymeshlab as ml
import pyacvd

from pytorch3d.structures import Meshes

import scipy.sparse as ssparse

def laplacian_numpy(verts, edges):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j     ====================> deg(i)
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge =============> -1
    L[i, j] =    0        , otherwise     ====================> 0
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    # Aggiungo gli edge simmetrici che normalmente non ci sono
    e0, e1 = edges[:,0], edges[:,1]
    idx01 = np.c_[e0, e1].T  # (E, 2)
    idx10 = np.c_[e1, e0].T  # (E, 2)
    idx = np.c_[idx01, idx10]

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = np.ones(idx.shape[1], dtype=np.float32)
    A = ssparse.coo_matrix((ones, (idx[0], idx[1])), shape=(V,V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = np.squeeze(np.array(A.sum(1)))

    # indici della diagonale
    idx = np.arange(V, dtype=int)
    idx = np.c_[idx, idx].T
    L = - A + ssparse.coo_matrix((deg, (idx[0], idx[1])), shape=(V,V))

    return L


def laplacian(verts, edges):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j     ====================> deg(i)
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge =============> -1
    L[i, j] =    0        , otherwise     ====================> 0
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    # Aggiungo gli edge simmetrici che normalmente non ci sono
    e0, e1 = edges.unbind(1)
    #e0, e1 = edges[:,0], edges[:,1]
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # indici della diagonale
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    L = - A + torch.sparse.FloatTensor(idx, deg, (V, V))

    return L



def discrete_dirichlet_energy(meshes):

    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals

    with torch.no_grad():
        L = laplacian(verts_packed, edges_packed)

    e = 0.5 * ( verts_packed.transpose(0,1).mm(L.mm(verts_packed)) ).trace()

    return e


def upsample_morph(origCoarseMesh, newCoarseMesh, fineMesh):
    fineMeshTemp = fineMesh.copy()
    origCoarseMeshTemp = origCoarseMesh.copy()
    origCoarseMeshTemp.clear_data()
    newCoarseMeshTemp = newCoarseMesh.copy()

    #radius = origCoarseMeshTemp.points.ptp() * 0.007
    radius = 3.
    disp = newCoarseMeshTemp.points - origCoarseMeshTemp.points
    origCoarseMeshTemp['disp'] = disp
    fineMeshTemp = fineMeshTemp.interpolate(origCoarseMeshTemp, radius=radius)
    fineMeshTemp.points = fineMeshTemp.points + fineMeshTemp['disp']
    return fineMeshTemp



def computePolylineBarycenter(polyline):
    """
    barycenter is computed taking into account the segments
    connecting polyline vertices by weighting point positions
    with thehalf-length of their neighboring segments
    """

    segmentIdx = polyline.cells.reshape((-1, 3))[:, 1:]
    segmentIdx = segmentIdx[segmentIdx[:, 0].argsort()]
    segmentIdx = np.c_[segmentIdx, segmentIdx[segmentIdx[:, 1].argsort()]]
    segmentIdx = segmentIdx.reshape((-1, 2))
    edgeLenght = np.sqrt(((polyline.points[segmentIdx[:, 0]] -
                           polyline.points[segmentIdx[:, 1]])**2).sum(axis=1))
    position_weights = (edgeLenght[::2] + edgeLenght[1::2]) / 2.
    position_weights = np.c_[position_weights, position_weights,
                             position_weights]
    barycenter = (polyline.points *
                  position_weights).sum(axis=0) / position_weights.sum(axis=0)

    return barycenter



def get_bou_idxs(model_):
    model = model_.copy()
    model['oldIdx'] = np.arange(model.points.shape[0])

    boundary_edges = model.extract_feature_edges(boundary_edges=True,
                                             non_manifold_edges=False,
                                             feature_edges=False,
                                             manifold_edges=False)
    boundary_edge_list = list(boundary_edges.split_bodies())
    nBEdges = len(boundary_edge_list)  # number of Boundary Edges
    #assert nBEdges == 5, nBEdges

    centers = np.empty([nBEdges, 3])
    for i in np.arange(nBEdges):
        centers[i] = computePolylineBarycenter(boundary_edge_list[i])
    z_idx = centers[:, 2].argsort()
    boundary_edge_list = [boundary_edge_list[i] for i in z_idx]
    centers = centers[z_idx]
    io_idx = centers[:2, 1].argsort(
    )  # ordine dei tre sovraortici in base alla coordinata y
    sa_idx = centers[2:, 1].argsort(
    ) + 2  # ordine dei tre sovraortici in base alla coordinata y
    edges_idx = np.r_[io_idx, sa_idx]
    boundary_edge_list = [boundary_edge_list[i] for i in edges_idx]
    centers = centers[edges_idx]

    origEdgeIdx = [boundary_edge_list[i]['oldIdx'] for i in range(len(boundary_edge_list))]

    return origEdgeIdx


def mesh_edge_loss(meshes, target_length = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = (torch.sqrt(((v0 - v1)**2.).sum(1)) - target_length) ** 2.0
    loss = loss * weights

    return loss.sum() / N


import trimesh

def pvToTrimesh(pvMesh):
    return trimeshMesh

def trimeshToPv(trimeshMesh):
    return pvMesh


#### ATTENZIONE: PERMUTA GLI INDICI DELLA MESH
#### DURANTE L'__INIT__ DELL'OGGETTO TRIMESH
def taubin_smooth(
        mesh_pv,
        lamb=0.5,
        nu=0.53,
        iterations=100,
        equal_weight=False
):

    mesh_trimesh = trimesh.Trimesh(
        vertices = mesh_pv.points,
        faces = mesh_pv.faces.reshape(-1,4)[:,1:],
    )
    lap = trimesh.smoothing.laplacian_calculation(
            mesh_trimesh,
            equal_weight=equal_weight)
    smoothed_mesh = trimesh.smoothing.filter_taubin(
            mesh_trimesh,
            lamb=lamb,
            nu=nu,
            iterations=iterations,
            laplacian_operator=lap)

    n_faces = mesh_trimesh.faces.shape[0]
    temp = np.array([3]*n_faces)
    faces = np.c_[temp, mesh_trimesh.faces]
    pvMesh = pv.PolyData(mesh_trimesh.vertices, faces.reshape(-1))
    return pvMesh


import vtk

def windowedSincSmooth(mesh, iters=20, passband=0.01):
    smoothed = vtk.vtkWindowedSincPolyDataFilter()
    smoothed.SetInputData(mesh)
    smoothed.SetNumberOfIterations(iters)
    smoothed.SetPassBand(passband)
    smoothed.SetBoundarySmoothing(False)
    smoothed.SetNonManifoldSmoothing(True)
    smoothed.SetNormalizeCoordinates(True)
    smoothed.Update()
    return pv.PolyData(smoothed.GetOutput())

def laplacianSmooth(mesh, iters, relaxFactor):
    smoothed = vtk.vtkSmoothPolyDataFilter()
    smoothed.SetInputData(mesh)
    smoothed.SetNumberOfIterations(iters)
    smoothed.SetBoundarySmoothing(False)
    smoothed.SetRelaxationFactor(relaxFactor)
    smoothed.Update()
    return pv.PolyData(smoothed.GetOutput())

def pv_to_ml(m):
  m_ml = ml.Mesh(m.points, m.faces.reshape(-1,4)[:,1:])
  # create a new MeshSet
  ms = ml.MeshSet()
  ms.add_mesh(m_ml, "mesh")
  return ms

def ml_to_pv(ms):
  m_ml = ms.current_mesh()
  face = m_ml.face_matrix()
  tmp = np.array([3]*face.shape[0])
  faces = np.c_[(tmp,m_ml.face_matrix())].reshape(-1)
  m = pv.PolyData(m_ml.vertex_matrix(), faces)
  return m

def mesh_metrics(m):
  edges = m.extract_all_edges()
  edges = edges.compute_cell_sizes()
  print(f"mean edge len  : {edges['Length'].mean()}")
  return

def remesh(m, target_edge_len):
  """https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing_isotropic_explicit_remeshing"""
  #mesh_metrics(m)
  ms = pv_to_ml(m)
  ms.remeshing_isotropic_explicit_remeshing(
    iterations = 10,
    targetlen = ml.AbsoluteValue(target_edge_len),
    adaptive = False,
    selectedonly = False,
    featuredeg = 30.,
    swapflag = True,
    smoothflag = True,
    reprojectflag = True,
  )
  m = ml_to_pv(ms)
  #mesh_metrics(m)
  return m

def remesh_with_n_points(m, n):
  clus = pyacvd.Clustering(m.copy())
  clus.subdivide(1)
  clus.cluster(n)
  m_ = clus.create_mesh()
  return m_

def interpolate_with_nearest_neighbour(m_, target_, array_name, cell_data):
  """
  m_ : mesh's points onto which the array will be interpolated
  target_ : mesh from which the array will be read
  
  ## note ##:
  Source array can be defined either on points or cells of <target_>.
  Then, it will be interpolated on m_'s points.
  """
  target = target_.copy()
  arr = target[array_name].copy()
  target.clear_data()
  if cell_data:
    target = pv.PolyData(target.cell_centers())
  target[array_name] = arr
  m = m_.copy()
  m = m.interpolate(target, strategy='closest_point', n_points=1, pass_cell_data=False)
  m[array_name] = m[array_name].astype(target[array_name].dtype)
  return np.array(m[array_name].copy())

def get_chambers_idxs(m):
  tags = np.unique(m['GroupIds'])
  idxs = []
  for tag in tags:
    idxs.append(np.where(m['GroupIds']==tag)[0])
  return idxs
