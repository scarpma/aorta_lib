import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')




## taken from aortaDataset
import numpy as np
import pyvista as pv
import vtk
import os.path as osp
import shutil
from io import StringIO
import os
import os.path as osp



def save_multiblock_stl(multiblock, filename):

    names = multiblock.keys()
    oname, ext = osp.splitext(filename)
    assert ext == '.stl'
    ofiles = [f'{oname}_{ii}' + '.stl' for ii in range(len(names))]

    for ii, subpart in enumerate(multiblock):
        subpart.save(ofiles[ii], binary=False)
        change_first_line_of_file(ofiles[ii], f'solid {names[ii]}')

    total_stl = ''
    for fn in ofiles:
        f = open(fn)
        total_stl += f.read()
        f.close()

    with open(oname + '.stl', 'w') as f:
        f.write(total_stl)

    for fn in ofiles:
        os.remove(fn)

    return



def change_first_line_of_file(filename, new_first_line):

    fr = open(filename, 'r')
    first_line = fr.readline()
    fr.close()
    first_line_len = len(first_line)

    new_first_line_len = len(new_first_line)
    spaces_num = first_line_len - new_first_line_len
    new_first_line = new_first_line + ' '*(spaces_num-1) + '\n'
    fw = StringIO(new_first_line)
    fr = open(filename, 'r+')
    shutil.copyfileobj(fw, fr)
    fr.close()
    fw.close()
    return



def order_bous(bous):

    nBEdges = len(bous)  # number of Boundary Edges
    assert nBEdges == 5, nBEdges

    centers = np.empty([nBEdges, 3])
    for i in np.arange(nBEdges):
        centers[i] = computePolylineBarycenter(bous[i])
    z_idx = centers[:, 2].argsort()
    bous = [bous[i] for i in z_idx]
    centers = centers[z_idx]
    io_idx = centers[:2, 1].argsort(
    )  # ordine dei tre sovraortici in base alla coordinata y
    sa_idx = (0.5*(centers[2:, 1] + centers[2:, 0])).argsort(
    ) + 2  # ordine dei tre sovraortici in base alla coordinata x + y
    edges_idx = np.r_[io_idx, sa_idx]
    bous = [bous[i] for i in edges_idx]

    bous = pv.MultiBlock(bous)
    bous.set_block_name(0, 'inlet')
    bous.set_block_name(1, 'outlet')
    bous.set_block_name(2, 'sa1')
    bous.set_block_name(3, 'sa2')
    bous.set_block_name(4, 'sa3')

    return bous


def compute_normals_vtk(surface):
    ## gives correctly outward oriented normals
    ## even for surfaces with sharp edges by
    ## splitting in more points where angles are
    ## sharp
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.AutoOrientNormalsOff()
    normals.ConsistencyOn()
    normals.SplittingOn()
    normals.SetFeatureAngle(60)
    normals.AutoOrientNormalsOn()
    #  normals.GetOutput().ReleaseDataFlagOn()
    normals.UpdateInformation()
    normals.Update()
    normals = normals.GetOutput()
    normals = pv.PolyData(normals)

    return normals


def adjust_bou_cap_normal_to_outward(cap, surface_):

    surface = surface_.copy()
    surface['oldIdx'] = np.full(surface.points.shape[0], -1, dtype=int)
    cap['oldIdx'] = np.arange(0, cap.points.shape[0], dtype=int)
    ### WITHOUT MERGE_POINTS=TRUE IT DIDN'T GET THE RIGHT NORMALS
    combined = pv.MultiBlock([surface, cap]).combine(merge_points=1).extract_surface()
    combined = compute_normals_vtk(combined)
    centerpoint_idx, = np.where(combined['oldIdx']==cap.points.shape[0]-1)
    #assert centerpoint_idx.shape[0] == 1
    centerpoint_normal = combined['Normals'][centerpoint_idx].mean(0)

    cap = cap.compute_normals(cell_normals=False)
    mean_normal = cap['Normals'].mean(0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    sign = np.sign(mean_normal.dot(centerpoint_normal))
    if sign < 0:
        print("reversing normals of cap")
        cap.flip_normals()
    return cap


def compute_bou_mean_outward_normal(bou, surface_):

    surface = surface_.copy()
    cap = createCapFromBoundary(bou)
    surface['oldIdx'] = np.full(surface.points.shape[0], -1, dtype=int)
    cap['oldIdx'] = np.arange(0, cap.points.shape[0], dtype=int)
    ### WITHOUT MERGE_POINTS=TRUE IT DIDN'T GET THE RIGHT NORMALS
    combined = pv.MultiBlock([surface, cap]).combine(merge_points=1).extract_surface()
    combined = compute_normals_vtk(combined)
    centerpoint_idx, = np.where(combined['oldIdx']==cap.points.shape[0]-1)
    #assert centerpoint_idx.shape[0] == 1
    centerpoint_normal = combined['Normals'][centerpoint_idx].mean(0)

    cap = cap.compute_normals(cell_normals=False)
    mean_normal = cap['Normals'].mean(0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    sign = np.sign(mean_normal.dot(centerpoint_normal))
    mean_normal = sign * mean_normal

    return mean_normal



def compute_polyline_mean_normal(bou, sign):

    cap = createCapFromBoundary(bou)
    cap = cap.compute_normals(cell_normals=False)
    mean_normal = cap['Normals'].mean(0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    mean_normal = np.sign(sign*mean_normal[2]) * mean_normal

    return mean_normal

## # OLD VERSION
## def compute_polyline_mean_normal(polyline, sign):
##     # sign is the vertical ortientation of the normal
##     center = computePolylineBarycenter(polyline)
##     raggi = polyline.points - center
##     # compute cross products between random pairs of raggi
##     rng = np.random.default_rng()
##     n_cross_prod = polyline.points.shape[0] // 2 * 2
##     randIdx = rng.choice(raggi.shape[0], size=n_cross_prod, replace=False)
##     randIdx1 = randIdx.reshape(2, -1)[0]
##     randIdx2 = randIdx.reshape(2, -1)[1]
## 
##     vector0 = polyline.points[-1] - center
##     cross = np.zeros(3)
##     for i in range(1,polyline.points.shape[0]):
##         vector1  = polyline.points[i] - center
##         cross += np.cross(vector1, vector0)
##         vector0 = vector1
## 
##     # normalize cross product
##     cross = cross / np.linalg.norm(cross)
##     cross = np.sign(cross[2]*sign) * cross
## 
##     return cross
##
## # OLD VERSION
## def compute_polyline_mean_normal(polyline, sign):
##     # sign is the vertical ortientation of the normal
##     center = computePolylineBarycenter(polyline)
##     raggi = polyline.points - center
##     # compute cross products between random pairs of raggi
##     rng = np.random.default_rng()
##     n_cross_prod = polyline.points.shape[0] // 2 * 2
##     randIdx = rng.choice(raggi.shape[0], size=n_cross_prod, replace=False)
##     randIdx1 = randIdx.reshape(2, -1)[0]
##     randIdx2 = randIdx.reshape(2, -1)[1]
##     cross_prods = np.cross(raggi[randIdx1], raggi[randIdx2])
##     # normalize cross products
##     cross_prods_mag = np.sqrt((cross_prods * cross_prods).sum(1))
##     cross_prods = cross_prods / np.repeat(cross_prods_mag, 3).reshape(-1, 3)
##     #define one direction for every vector
##     if np.sign(cross_prods[0, 2]) != sign:
##         cross_prods[0] = cross_prods[0] * -1.
##     directions = (cross_prods * cross_prods[0]).sum(1)
##     directions = directions / np.abs(directions)
##     cross_prods = (cross_prods.T * directions).T
##     mean_normal = cross_prods.mean(0)
##     return mean_normal



def compute_ortnorm_basis(bou, surface):

    bouCenter = computePolylineBarycenter(bou)
    normal = compute_bou_mean_outward_normal(bou, surface)
    outest_point_idx = (bou.points - bouCenter).dot(normal).argmax()

    # create orthonormal axes "a" and "b" with respect to "normal"
    # the first axis "a" is parallel to the direction from the center
    # to the 'outest' point of "bou", i.e. the one which have the
    # greatest component parallel to the bou normal (outwards)
    a = bou.points[outest_point_idx] - bouCenter
    a = a - normal.dot(a) * normal
    a = a - normal.dot(a) * normal
    a = a - normal.dot(a) * normal
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    b /= np.linalg.norm(b)

    return normal, a, b, bouCenter, outest_point_idx



def getIdxCorrespondence(bouPoints, a, b, bouCenter):

    """
    Calcola l'ordine secondo cui i punti del
    bordo diventano consecutivi. Questo perché
    in generale i punti di un bordo hanno ordine
    casuale. Essendo però una linea, posso
    definire un ordine tra i punti, dato dal
    l'ordine tra gli angoli a cui ogni punto è
    associato. L'angolo di ogni punto è calcolato
    come l'angolo tra il segmento (a;center) ed il
    segmento (center; punto), dove a è uno dei
    punti del bordo scelto appositamente come
    il punto più 'esterno' del bordo.
    """

    x = a.dot( (bouPoints - bouCenter).T )
    y = b.dot( (bouPoints - bouCenter).T )
    alpha = np.arctan2(y, x)
    alpha = alpha + (alpha < 0) * 2 * np.pi

    return np.argsort(alpha)



def attach_and_smooth_extension(bou, ext, n_interp_rings, surface):

    ts = np.linspace(0, 1, n_interp_rings+1)
    n_pts = bou.points.shape[0]
    normal, a, b, bouCenter, outest_point_idx = compute_ortnorm_basis(bou, surface)
    bou_angle_order = getIdxCorrespondence(
        bou.points, a=a, b=b, bouCenter=bouCenter)

    # interpolo i punti dei vari anelli dell'estensione
    # verso i punti del bordo dell'aorta. Questo per avere
    # una transizione piú smooth. L'ordine dei punti del
    # bordo è diverso dall'ordine dei punti dell'estensione.

    # Per fare un'interpolazione corretta devo associare i
    # punti ad un sistema di riferimento comune. Ho scelto quello
    # angolare dato dalla funzione "getIdxCorrespondence".

    # tramuto l'ordine dei punti del bordo in quello del sistema
    # di riferimento della funzione. Poi tramuto questo ordine in
    # quello dell'estensione (in particolare dell'anello in questione)
    # calcolando l'inverso dell'ordine dell'estensione (con argsort()).

    for kk in range(n_interp_rings):
        # source points to be changed
        extPoints = ext.points[kk*n_pts:(kk+1)*n_pts]
        # order of these points by reference system
        ext_angle_order = getIdxCorrespondence(
            extPoints, a=a, b=b, bouCenter=bouCenter)
        # inverted transf. from ref. system to current ring 
        angle_ext_order = ext_angle_order.argsort()

        # target points
        if kk > 0:
            trgPoints = ext.points[(kk-1)*n_pts:(kk)*n_pts]
        else :
            trgPoints = bou.points

        trg_angle_order = getIdxCorrespondence(
            trgPoints, a=a, b=b, bouCenter=bouCenter)

        extPoints_edited = (1-ts[kk])*\
            trgPoints[trg_angle_order][angle_ext_order] + ts[kk]*extPoints
        ext.points[kk*n_pts:(kk+1)*n_pts] = extPoints_edited

    return ext



def extendPolyline(bou, length, surface):

    """
    input: polyline (chiusa a loop)

    Crea un'estensione cilindrica di una superficie.
    Il cilindro ha direzione pari alla normale del bordo
    'bou' in ingresso. Il numero di punti per ogni anello
    del cilindro è pari al numero di punti sul bordo 'bou'
    inizialmente fornito.
    """

    assert check_if_closed(bou)

    normal, a, b, bouCenter, outest_point_idx = compute_ortnorm_basis(bou, surface)

    #import glob
    #paths = glob.glob('bouCenter_*.vtp')
    #if len(paths) == 0:
    #    num = 0
    #else:
    #    paths = sorted(paths)
    #    num = int(osp.splitext(paths[-1])[0].split('_')[-1]) + 1
    #p = pv.PolyData(bouCenter[np.newaxis, :])
    #p['normal'] = normal[np.newaxis, :]
    #p['a'] = a[np.newaxis, :]
    #p['b'] = b[np.newaxis, :]
    #p.save(f'bouCenter_{num}.vtp')

    radius = np.mean(np.sqrt(((bou.points - bouCenter)**2.).sum(1)))

    # 1e-12 is added to prevent round errors in np.arange
    dangle = 2 * np.pi / bou.points.shape[0] + 1.e-12
    dabscissa = (np.sqrt(3.) / 2 ) * 2. * np.sin(dangle / 2.) * radius
    outest_point_idx = (bou.points - bouCenter).dot(normal).argmax()
    abscissa_initial_offset = (bou.points - bouCenter).dot(normal)[outest_point_idx] +\
        dabscissa #/ 3.

    angles = np.arange(0, 2*np.pi, dangle)
    abscissas = np.arange(abscissa_initial_offset, \
                    abscissa_initial_offset + length, dabscissa)
    ringPoints = np.zeros((angles.shape[0] * len(abscissas), 3))

    for kk in range(len(abscissas)):
        for ii in range(len(angles)):
            shift_angle = dangle / 2. * kk#(kk + 1) # with kk+1 starts not aligned with
                                                    # axis "a"
            ringPoints[ii + kk*len(angles)] = bouCenter + abscissas[kk] * normal +\
                radius * (b * np.sin(angles[ii] + shift_angle) +\
                np.cos(angles[ii] + shift_angle) * a)

    Ncells = bou.points.shape[0] * (len(abscissas) - 1) * 2
    cells = np.zeros((Ncells,3), dtype=int)

    # up triangles
    for kk in range(len(abscissas) - 1):
        for ii in range(len(angles) - 1):
            cells[(ii + kk*(len(angles))), 0] = ii     + kk*len(angles)
            cells[(ii + kk*(len(angles))), 1] = ii + 1 + kk*len(angles)
            cells[(ii + kk*(len(angles))), 2] = ii     + (kk+1)*len(angles)
        ii += 1
        cells[(ii + kk*len(angles)), 0] = ii     + kk*len(angles)
        cells[(ii + kk*len(angles)), 1] = 0      + kk*len(angles)
        cells[(ii + kk*len(angles)), 2] = ii     + (kk+1)*len(angles)

    ## # down triangles
    offset = len(angles) * (len(abscissas) - 1)
    for kk in range(len(abscissas) - 1):
        ii = 0
        cells[(ii + kk*len(angles)) + offset, 0] = ii     + kk*len(angles)
        cells[(ii + kk*len(angles)) + offset, 1] = ii     + (kk+1)*len(angles)
        cells[(ii + kk*len(angles)) + offset, 2] = len(angles) - 1 + (kk+1)*len(angles)
        for ii in range(1, len(angles)):
            cells[(ii + kk*(len(angles))) + offset, 0] = ii     + kk*len(angles)
            cells[(ii + kk*(len(angles))) + offset, 1] = ii     + (kk+1)*len(angles)
            cells[(ii + kk*(len(angles))) + offset, 2] = ii - 1 + (kk+1)*len(angles)

    faces = np.c_[np.full(cells.shape[0], 3), cells]
    return pv.PolyData(ringPoints, faces)



def createCapFromBoundary(bou):

    try:
        bouCells = bou.cells.reshape(-1,3)[:,1:]
    except AttributeError:
        bouCells = bou.lines.reshape(-1,3)[:,1:]

    num_cells = bouCells.shape[0]
    num_points = bou.points.shape[0]

    cappedPoints = np.zeros(shape=(num_points + 1, 3))
    cappedPoints[:-1] = bou.points
    cappedPoints[-1] = computePolylineBarycenter(bou)

    cappedCells = np.zeros((num_cells,3), dtype=int)
    cappedCells[:,:-1] = bouCells
    cappedCells[:,-1] = num_points

    cappedCells = np.c_[np.full(num_cells, 3), cappedCells]
    return pv.PolyData(cappedPoints, cappedCells.reshape(-1))



def check_if_closed(polyline):
    n_points = polyline.points.shape[0]
    try:
        n_edges = polyline.cells.size / 3
    except AttributeError:
        n_edges = polyline.lines.size / 3

    return n_points == n_edges


def computePolylineBarycenter(polyline_):
    """
    barycenter is computed taking into account the segments
    connecting polyline vertices by weighting point positions
    with thehalf-length of their neighboring segments
    """
    polyline = polyline_.copy()
    assert len(polyline.split_bodies()) == 1
    try:
        segmentIdx = polyline.cells.reshape((-1, 3))[:, 1:]
    except AttributeError:
        segmentIdx = polyline.lines.reshape((-1, 3))[:, 1:]

    # vicini di ogni punto i per i da 0 a N.
    # neigs[i] contiene i vicini del punto i
    neigs = np.c_[segmentIdx[segmentIdx[:, 0].argsort()],\
                  segmentIdx[segmentIdx[:, 1].argsort()]][:,1:3]
    points = polyline.points
    weights = ( np.sqrt(((points - points[neigs[:,0]])**2).sum(1)) +\
           np.sqrt(((points - points[neigs[:,1]])**2.).sum(1)) ) / 2.

    weights3 = weights.repeat(3).reshape(-1, 3)

    barycenter = (polyline.points * weights3).sum(axis=0) / weights.sum(axis=0)

    return barycenter
    #return polyline_.points.mean(0)


def cap_polyline(polyline):
    temp = polyline.copy()
    order = temp.lines.reshape((-1, 3))[:, 1:]
    center = computePolylineBarycenter(temp)
    points = np.zeros((temp.points.shape[0] + 1, 3))
    points[:-1] = temp.points
    points[-1] = center
    faces = np.empty(shape=(4, points.shape[0] - 1), dtype=int)
    faces[0, :] = 3
    faces[1, :] = points.shape[0]
    faces[2, :] = order[:, 0]
    faces[3, :] = order[:, 1]
    faces = faces.T.reshape(-1)
    return pv.PolyData(points, faces)

def extract_submodels(model):
    # identify each block of the model (disc and supraortics)
    # in order to create cuts for each single one. This is done
    # by looking at the GroupIds array created by vmtk.
    submodels = model.split_bodies()
    a = [submodel['GroupIds'] for submodel in submodels]
    gIds = []
    submodelIdx = []
    for kk, submodel_gids in enumerate(a):
        if submodel_gids.size < 50:
            print('Found small isolated region. This '+\
            'will be ignored.')
            continue
        if np.all(submodel_gids == submodel_gids[0]):
            gIds.append(submodel_gids[0])
        else:
            first_gid = submodel_gids == submodel_gids[0]
            second_gid = submodel_gids != submodel_gids[0]
            assert np.all(second_gid == second_gid[0])
            if second_gid.size > first_gid.size:
                first_gid, second_gid = second_gid, first_gid
            assert second_gid.size / first_gid.size < 0.1

            print('Found small region with different \
            groupId. This will be ignored.')
            gIds.append(first_gid[0])
        submodelIdx.append(kk)

    return submodels, gIds, submodelIdx


def identify_sa_and_aorta_from_submodels(submodels, gIds, submodelIdx):
    sa3_gid = gIds[-1]
    sa3_idx = submodelIdx[-1]
    sa2_gid = gIds[-2]
    sa2_idx = submodelIdx[-2]
    sa1_gid = gIds[-3]
    sa1_idx = submodelIdx[-3]
    aorta_gids = gIds[:-3]
    aorta_idxs = submodelIdx[:-3]

    # sometimes supraortic vessels have more than one gid,
    # so aorta_idxs will contain some parts of supraortic
    # vessels. If this is the case, we must be sure that
    # the descending aorta is in the end of aorta_idxs (we
    # will not remove the small pieces of supraortics, but we'll
    # just reorder the array)
    #
    # perform the check by:
    # checking that min z coord of last submodel is less than
    # min z coord of penultimate submodel
    #
    # AND
    #
    # checking that min z coord of highest boundary of last submodel
    # is not equal to min z coord of highest boundary of penultimate
    # submodel.

    boundary_edges_last_submodel = list(submodels[aorta_idxs[-1]].extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False).split_bodies())
    assert len(boundary_edges_last_submodel) == 2, len(boundary_edges_last_submodel)
    lowest_z_0 = boundary_edges_last_submodel[0].points.min(0)[2]
    lowest_z_1 = boundary_edges_last_submodel[1].points.min(0)[2]
    last_submodel_lowest_z_of_highest_bou = max([lowest_z_0, lowest_z_1])
    boundary_edges_penultimate_submodel = list(submodels[aorta_idxs[-2]].extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False).split_bodies())
    assert len(boundary_edges_penultimate_submodel) == 2, len(boundary_edges_penultimate_submodel)
    lowest_z_0 = boundary_edges_penultimate_submodel[0].points.min(0)[2]
    lowest_z_1 = boundary_edges_penultimate_submodel[1].points.min(0)[2]
    penultimate_submodel_lowest_z_of_highest_bou = max([lowest_z_0, lowest_z_1])

    while submodels[aorta_idxs[-1]].points.min(0)[2] > submodels[aorta_idxs[-2]].points.min(0)[2] and \
            np.abs(last_submodel_lowest_z_of_highest_bou - penultimate_submodel_lowest_z_of_highest_bou)>0.1:
        #print('found a supraortic in aorta_idxs.')
        #print('before rolling idx', aorta_idxs)
        #print('before rolling gid', aorta_gids)
        aorta_idxs = np.roll(aorta_idxs,1)
        aorta_gids = np.roll(aorta_gids,1)
        #print('after rolling idx', aorta_idxs)
        #print('after rolling gid', aorta_gids)
    #print('gid, idx')
    #print('aorta_gids ', aorta_gids, aorta_idxs)
    #print('sa1_gid ', sa1_gid, sa1_idx)
    #print('sa2_gid ', sa2_gid, sa2_idx)
    #print('sa3_gid ', sa3_gid, sa3_idx)
    return aorta_idxs, aorta_gids, sa1_idx, sa1_gid, sa2_idx, sa2_gid, sa3_idx, sa3_gid


def extract_aorta_and_sa(model):
    submodels, gIds, submodelIdx = extract_submodels(model)
    aorta_idxs, aorta_gids, sa1_idx, sa1_gid, sa2_idx, sa2_gid, sa3_idx, sa3_gid = \
            identify_sa_and_aorta_from_submodels(submodels, gIds, submodelIdx)
    names = ['aorta', 'sa1', 'sa2', 'sa3']
    return submodels, aorta_idxs, sa1_idx, sa2_idx, sa3_idx, names


def measure_lengths(model, arrayName):

    submodels, aorta_idxs, sa1_idx, sa2_idx, sa3_idx, names = extract_aorta_and_sa(model)
    contours = pv.MultiBlock()
    # now each block of the model has been identified (disc, sa1 sa2, sa3).
    # we can go forward to create cuts for each one separately.
    # ciclo sulla discendente e dal primo all'ultimo sovraortico
    boundaries = []
    for kk, idx in enumerate([aorta_idxs[-1], sa1_idx, sa2_idx, sa3_idx]):
        # find correct abscissa value for each block of the model
        submodel = submodels[idx]
        submodel['oldPointIdx'] = np.arange(submodel.points.shape[0])
        boundary_edges = submodel.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False)
        boundary_edge_list = list(boundary_edges.split_bodies())
        nBEdges = len(boundary_edge_list)  # number of Boundary Edges

        # sometimes a small boundary is created since vmtk clips erroneously.
        # this eliminates the small hole in the mesh
        if nBEdges == 3:
            n_points = [bou.points.shape[0] for bou in boundary_edge_list]
            n_points = np.array(n_points)
            smallest_bou_idx = n_points.argmin()
            boundary_edge_list.pop(smallest_bou_idx)
            nBEdges = 2
            print('found a small hole in ', names[kk])
        elif nBEdges == 2:
            pass
        else:
            assert False

        centers = np.empty([nBEdges, 3])
        for i in np.arange(nBEdges):
            centers[i] = computePolylineBarycenter(boundary_edge_list[i])
        # qui, per ogni submodel (ovvero discendente, sa1 2 o 3) cerco di individuare
        # il boundary che ha coordinata z piu bassa (piu alta per la discendente).
        # Questo mi serve per capire il valore di abscissa da cui misurare la
        # lunghezza a cui fare il taglio.
        z_idx = centers[:, 2].argsort()
        if kk == 0: z_idx = z_idx[::-1] # per la discendente prendo il bordo alto
        boundary_edge_list = [boundary_edge_list[i] for i in z_idx]
        boundaries.append(boundary_edge_list)

    return [np.abs(bou[1][arrayName].mean() - bou[0][arrayName].mean()) for bou in boundaries]


def computeContours(model, arrayName, lenghts):

    submodels, aorta_idxs, sa1_idx, sa2_idx, sa3_idx, names = extract_aorta_and_sa(model)
    contours = pv.MultiBlock()
    # now each block of the model has been identified (disc, sa1 sa2, sa3).
    # we can go forward to create cuts for each one separately.
    # ciclo sulla discendente e dal primo all'ultimo sovraortico
    for kk, idx in enumerate([aorta_idxs[-1], sa1_idx, sa2_idx, sa3_idx]):
        # find correct abscissa value for each block of the model
        submodel = submodels[idx]
        submodel['oldPointIdx'] = np.arange(submodel.points.shape[0])
        boundary_edges = submodel.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False)
        boundary_edge_list = list(boundary_edges.split_bodies())
        nBEdges = len(boundary_edge_list)  # number of Boundary Edges

        # sometimes a small boundary is created since vmtk clips erroneously.
        # this eliminates the small hole in the mesh
        if nBEdges == 3:
            n_points = [bou.points.shape[0] for bou in boundary_edge_list]
            n_points = np.array(n_points)
            smallest_bou_idx = n_points.argmin()
            boundary_edge_list.pop(smallest_bou_idx)
            nBEdges = 2
            print('found a small hole in ', names[kk])
        elif nBEdges == 2:
            pass
        else:
            assert False

        centers = np.empty([nBEdges, 3])
        for i in np.arange(nBEdges):
            centers[i] = computePolylineBarycenter(boundary_edge_list[i])
        # qui, per ogni submodel (ovvero discendente, sa1 2 o 3) cerco di individuare
        # il boundary che ha coordinata z piu bassa (piu alta per la discendente).
        # Questo mi serve per capire il valore di abscissa da cui misurare la
        # lunghezza a cui fare il taglio.
        z_idx = centers[:, 2].argsort()
        if kk == 0: z_idx = z_idx[::-1] # per la discendente prendo il bordo alto
        boundary_edge_list = [boundary_edge_list[i] for i in z_idx]

        abscissa = boundary_edge_list[0][arrayName].mean()
        #tallest_bou_point_idx = boundary_edge_list[0].points[:,2].argmax()
        #abscissa = boundary_edge_list[0][arrayName][tallest_bou_point_idx]

        # compute contour
        contour = submodel.contour(isosurfaces=1,
                                   scalars=arrayName,
                                   compute_scalars=True,
                                   rng=(abscissa + lenghts[kk] - 2.,
                                        abscissa + lenghts[kk] + 2.),
                                   preference='cell',
                                   method='contour')

        # check per vedere se i contorni trovati sono
        # delle linee chiuse (a volte vengono delle
        # linee spurie)
        #contour.save('contour.vtk')
        if len(contour.split_bodies()) > 1:
            for line in contour.split_bodies():
                closed_bool = check_if_closed(line)
                if closed_bool:
                    contour = line
                    break

        contours.append(contour)

    return contours


def clip_surface_with_contours(surface,
                               contours,
                               height,
                               delta_radius,
                               signs):

    # create cylinders
    cyls = pv.MultiBlock()
    print('Computing cylynders to cut model')
    for kk, contour in enumerate(contours):
        center = computePolylineBarycenter(contour)
        radius = np.max(np.sqrt(((contour.points - center)**2.).sum(1)))
        normal = compute_polyline_mean_normal(contour, signs[kk])
        cyls.append(
            #pv.Cylinder(center + normal * height / 2.,
            #        normal,
            #        radius=radius + delta_radius[kk],
            #        height=height,
            #        resolution=100,
            #        capping=True))
            pv.CylinderStructured(
                    center= center + normal * height / 2.,
                    direction = normal,
                    radius=(np.linspace(1.0, radius + delta_radius[kk], 10)),
                    height=height,
                    theta_resolution=20,
                    z_resolution=5,).extract_surface())

    clipped = surface
    for ii, cyl in enumerate(cyls):
        print(f'Clipping with cylinder{ii}')
        clipped = clipped.clip_surface(cyl, invert=False, compute_distance=True)
        bodies = clipped.split_bodies()
        n_points = [body.points.shape[0] for body in bodies]
        idx = np.argmax(n_points)
        clipped = bodies[idx]

    # clean surface and re-triangulate
    # since the cut produces bad polygons, we will have
    # borders with points possibly very near to each
    # other. One way to correct this is to do a "clean"
    # with a large tolerance (will merge points nearer
    # than the tolerance). But I found that choosing 
    # the tolerance to be 0.3 is too much because then
    # non manifold triangles are produced near cuts
    print(f'Cleaning surface and re-triangulating')
    clipped_cleaned = clipped.extract_surface().clean(
        absolute=True, tolerance=0.1).triangulate()

    return clipped_cleaned, cyls



### added 25/01/22


from scipy.fft import rfft, rfftfreq, irfft

def smoothedPolylineFFT(polyline, cutoff=0.5, closed=False):

    """
    ho aggiunto qualche hack momentaneo perché c'era un problema
    con le polyline con numero di punti dispari.
    sarebbe da sistemare il bug fix per renderlo pulito.
    al momento se il numero di punti è dispari, viene aggiunto un
    punto uguale all'ultimo. Poi viene tolto.
    """

    def _filter(y, cutoff):
        # Pretty much followed the instructions here.
        # https://realpython.com/python-scipy-fft/
        assert(0<=cutoff<=1)
        dt = 1/(len(y)-1)
        t = np.linspace(0,1,len(y))
        xf = rfftfreq(len(t), d=dt)
        yf = rfft(y)
        yf[int(cutoff*len(yf)):] = 0
        return irfft(yf)

    points = polyline.points

    # Extend signal to avoid artifacts at beginning and end of the line.
    if closed:
        ns = points.shape[0]
        points = np.tile(points, (3, 1))
        if points.shape[0]%2==1:
            DISPARI = True
            points = np.vstack([points, points[-1]])
        else:
            DISPARI = False
        previous_shape = points.shape[0]
    else:
        ns = points.shape[0]//2
        points = np.concatenate([np.tile(points[0], (ns,1)),
                                 points,
                                 np.tile(points[-1], (ns,1))])
    points = np.apply_along_axis(_filter, axis=0, arr=points,
                                 cutoff=cutoff)

    # Undo signal extension
    ##points = points[ns:-ns+1]
    if DISPARI:
        points = points[ns:-ns-1]
    else:
        points = points[ns:-ns]


    result = pv.PolyData(points, lines=np.r_[points.shape[0], np.arange(0,points.shape[0])])

    assert polyline.points.shape[0] == result.points.shape[0]

    return result



def order_cell_array(cell):

    # non altera la connettività della mesh

    cells = cell.reshape(-1,3)[:,1:]
    cell0sorted = cells[cells[:,0].argsort()]

    idxs = []
    pairs = []
    cur_id = cell0sorted[0,0]
    idxs.append(cur_id)
    for i in range(len(cell0sorted)):
        next_id = cell0sorted[cell0sorted[cur_id,1],0]
        idxs.append(next_id)
        pairs.append([cur_id, next_id])
        cur_id = next_id

    idxs = np.array(idxs)
    new_lines = np.c_[np.full(len(pairs), 2), np.array(pairs)].reshape(-1)

    return new_lines



def order_polyline(polyline):

    # altera la connettività della mesh

    try:
        polyline.cells
    except:
        polyline.cells = polyline.lines

    new_cells = order_cell_array(polyline.cells)
    new_point_order = new_cells.reshape(-1,3)[:,1]
    n_pts = new_point_order.shape[0]
    new_lines = np.c_[np.full(n_pts,2), np.arange(0,n_pts), np.r_[np.arange(1, n_pts), 0]]
    new_bou = pv.PolyData(polyline.points[new_point_order], lines=new_lines)

    return new_bou, new_point_order



def smooth_bou(bou_, surface, cutoff=0.4):

    bou = bou_.copy()
    orderedBou, idx = order_polyline(bou)
    new_bou = pv.PolyData(smoothedPolylineFFT(orderedBou, closed=True, cutoff=cutoff))
    points = new_bou.points[idx.argsort()]
    smoothedBou = pv.PolyData(points, lines=bou.cells)

    return smoothedBou



from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic

def binned_stat(x, values, nbins, y=None):

    if y is None:
        bin_means, bin_edges, binnumber = binned_statistic(x, values, statistic='mean', bins=nbins)
        bin_stds, _, _ = binned_statistic(x, values, statistic='std', bins=nbins)
        bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_center, bin_means, bin_stds, binnumber-1
    else:
        bin_means, bin_edgesx, bin_edgesy, binnumber = binned_statistic_2d(x,y,values, statistic='mean', bins=nbins)
        bin_stds, _, _, _ = binned_statistic_2d(x,y,values, statistic='mean', bins=nbins)
        bin_centerx = (bin_edgesx[:-1] + bin_edgesx[1:]) / 2
        bin_centery = (bin_edgesy[:-1] + bin_edgesy[1:]) / 2
        return bin_centerx, bin_centery, bin_means, bin_stds, binnumber-1





