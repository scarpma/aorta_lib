import numpy as np
import scipy.spatial

EPSILON = 1e-5

def seg_tria_intersection(P0, P1, V0, V1, V2):
    """
    P0 first point of the segment
    P1 second point of the segment
    V0, V1, V2 vertices of triangle
    D normalized direction of segment
    l length of the segment
    u,v barycentric coordinated of intersection (inside triangle
    if 0 < u < 1. and 0. < v < 1.)
    t coordinate along segment of the intersection (inside segment
    if 0. < t < l.)
    3d space
    """
    O = P0
    D = P1 - P0
    l = np.sqrt((D**2.).sum(-1))
    D = (D.T / l).T

    E1 = V1 - V0 # edge 1 of triangle
    E2 = V2 - V0 # edge 2 of triangle
    P = np.cross(D, E2)
    det = np.einsum('...i,...i',E1, P)
    # if det is near 0, ray lies in triangle's plane
    det_bool = np.abs(det) > EPSILON

    # EPSILON2 must be >0 because otherwise segments that start
    # or finish in one of the triangle vertices are counted as
    # intersections
    EPSILON2 = 0.001

    inv_det = 1 / det
    T = O - V0
    u = np.einsum('...i,...i', T, P) * inv_det
    u_bool = np.logical_and(u > 0.+EPSILON2, u < 1.-EPSILON2)

    Q = np.cross(T, E1)
    v = np.einsum('...i,...i', D, Q) * inv_det
    v_bool = np.logical_and(v > 0.+EPSILON2, u+v < 1.-EPSILON2)

    t = np.einsum('...i,...i', E2, Q) * inv_det
    t_bool = np.logical_and(t > 0., t < 1.)

    return det_bool * u_bool * v_bool * t_bool


#def tria_tria_intersection(tria1_coo, tria2_coo):
#    """
#    this is the correct implementation to check if two triangles intersect.
#    given two triangles, they interset if
#        or one edge on each intersect
#        or two edges of one intersect the other one
#    """
#
#
#    if tria1_coo.ndim == 2:
#        tria1_coo = tria1_coo[(None)]
#    if tria2_coo.ndim == 2:
#        tria2_coo = tria2_coo[(None)]
#
#    i1 = 0
#    i1 += seg_tria_intersection(
#        tria1_coo[:,0], tria1_coo[:,1],
#        tria2_coo[:,0], tria2_coo[:,1], tria2_coo[:,2],
#    )
#    i1 += seg_tria_intersection(
#        tria1_coo[:,1], tria1_coo[:,2],
#        tria2_coo[:,0], tria2_coo[:,1], tria2_coo[:,2],
#    )
#    i1 += seg_tria_intersection(
#        tria1_coo[:,2], tria1_coo[:,0],
#        tria2_coo[:,0], tria2_coo[:,1], tria2_coo[:,2],
#    )
#
#    i2 = 0
#    i2 += seg_tria_intersection(
#        tria2_coo[:,0], tria2_coo[:,1],
#        tria1_coo[:,0], tria1_coo[:,1], tria1_coo[:,2],
#    )
#    i2 += seg_tria_intersection(
#        tria2_coo[:,1], tria2_coo[:,2],
#        tria1_coo[:,0], tria1_coo[:,1], tria1_coo[:,2],
#    )
#    i2 += seg_tria_intersection(
#        tria2_coo[:,2], tria2_coo[:,0],
#        tria1_coo[:,0], tria1_coo[:,1], tria1_coo[:,2],
#    )

    return np.logical_or(np.logical_or(i1 == 2, i2 == 2), np.logical_and(i1 == 1, i2 == 1))

def tria_tria_intersection(tria1_coo, tria2_coo):

    """
    detects if edges of tria1 intersect tria2. Note that this is different from
    detecting if the two triangles intersect. I can use this shorter version of the
    intersection function because I loop among all triangles, so if only triangle2'edges
    intersect triangle1, the intersection will be spotted later in the loop
    If not in a loop, given two triangles, they interset iff:
        or one edge on each intersect
        or two edges of one intersect the other one
    """

    tria1_coo = tria1_coo.astype(np.float32)
    tria2_coo = tria2_coo.astype(np.float32)

    if tria1_coo.ndim == 2:
        tria1_coo = tria1_coo[(None)]
    if tria2_coo.ndim == 2:
        tria2_coo = tria2_coo[(None)]

    i = 0
    i += seg_tria_intersection(
        tria1_coo[:,0], tria1_coo[:,1],
        tria2_coo[:,0], tria2_coo[:,1], tria2_coo[:,2],
    )
    i += seg_tria_intersection(
        tria1_coo[:,1], tria1_coo[:,2],
        tria2_coo[:,0], tria2_coo[:,1], tria2_coo[:,2],
    )
    i += seg_tria_intersection(
        tria1_coo[:,2], tria1_coo[:,0],
        tria2_coo[:,0], tria2_coo[:,1], tria2_coo[:,2],
    )

    return i>0

def remove_neig_cells(tria_idx, cell_idxs, near_cells):

    near_points_idxs = cell_idxs[near_cells]
    idx1, idx2, idx3 = cell_idxs[tria_idx]
    non_neig_cells_bool = np.logical_not(np.any(
        np.array([
            near_points_idxs[:,0] == idx1,
            near_points_idxs[:,0] == idx2,
            near_points_idxs[:,0] == idx3,
            near_points_idxs[:,1] == idx1,
            near_points_idxs[:,1] == idx2,
            near_points_idxs[:,1] == idx3,
            near_points_idxs[:,2] == idx1,
            near_points_idxs[:,2] == idx2,
            near_points_idxs[:,2] == idx3,
        ]),
        axis=0,
    ))

    return near_cells[non_neig_cells_bool]



def check_intersections(mesh, r=None):

    cell_idxs = np.array(mesh.faces.reshape((-1,4))[:,1:])
    cell_coords = np.array(mesh.points[cell_idxs])
    cell_center_coords = np.array(mesh.cell_centers().points)
    kdtree = scipy.spatial.KDTree(cell_center_coords)

    if r is None:
        max_distance_between_center_and_vertices = np.sqrt((
            (cell_center_coords[:,None,:] - mesh.points[cell_idxs])**2
        ).sum(-1)).max()
        r = max_distance_between_center_and_vertices * 1.05

    near_cells = kdtree.query_ball_point(cell_center_coords, r, return_sorted=False, workers=-1)

    tria1_idxs_tot, tria2_idxs_tot = [], []
    for tria1_idx, tria2_idxs in enumerate(near_cells):
        #tria2_idxs_non_neig = list(remove_neig_cells(tria1_idx, cell_idxs, np.array(tria2_idxs)))
        tria2_idxs_non_neig = tria2_idxs
        tria1_idxs_tot += [tria1_idx] * len(tria2_idxs_non_neig)
        tria2_idxs_tot += tria2_idxs_non_neig
    tria1_idxs_tot = np.array(tria1_idxs_tot)
    tria2_idxs_tot = np.array(tria2_idxs_tot)

    tria1_coo = cell_coords[tria1_idxs_tot]
    tria2_coo = cell_coords[tria2_idxs_tot]

    o = tria_tria_intersection(tria1_coo, tria2_coo)

    return tria1_idxs_tot[o].size
