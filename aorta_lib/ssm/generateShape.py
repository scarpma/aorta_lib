import numpy as np
import h5py
import pyvista as pv
import glob
import os.path as osp
from aorta_lib.ssm import check_intersections


#SSM_PATH = "SSM_47shapes_relaxed_corrected.npy"
SSM_PATH = "SSM_47shapes_relaxed_corrected.h5"
#SSM_PATH = "SSM_47shapes_relaxed_corrected_SparsePCA.npy"
#SSM_PATH = "SSM_47shapes_raw.h5"
#SSM_PATH = "SSM_47.h5"
MAX_DEV = 2.99
DEV = 0.8 # std della distribuzione da cui estrarre i pesi
SAVE_DIR = 'SSM_output'
refModelPath = 'V2_to_V2.vtp'


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform



def GPa(points, eps):
    """
    Inputs:
        ------------
        points
            matrix NxMxD
                N number of samples
                M number of points per sample
                D number of dimensions
        eps
            threshold for the maximum movement of mean shape
            to control convergence

    """
    delta_mean = 1.e10
    mean = points.mean(0)
    points_new = points.copy()

    while delta_mean > eps :
        for ii in range(points.shape[0]):
            source_pts = points[ii]
            d, new_source_pts, T = procrustes(mean, source_pts, scaling=False, reflection=False)
            points_new[ii] = new_source_pts.reshape(points.shape[1:])

        delta_mean = np.sqrt(((mean - points_new.mean(0))**2. ).sum())
        print(delta_mean)
        points = points_new
        mean = points_new.mean(0)

    return points_new


def load_CFD_model(path):

    ##TODO

    return cfdModel


def load_SSM_h5(filename, refModelPath):
    f = h5py.File(filename, "r")
    #keys = list(f.keys())
    #print(keys)
    #f['model'].keys()

    meanShapePoints = np.array(f['model']['mean'])
    meanShapePoints = meanShapePoints.reshape(-1,3)
    faces = pv.read(refModelPath).faces
    meanShape_pv = pv.PolyData(meanShapePoints, faces)
    #meanShapepv.save('meanShape.vtp')

    pca_variance = np.array(f['model']['pcaVariance'])
    pca_basis = np.array(f['model']['pcaBasis'])
    pca_basis = pca_basis.reshape(pca_variance.shape[0], meanShapePoints.shape[0], 3)

    return meanShape_pv, pca_basis, pca_variance


def load_SSM(filename):
    data = np.load(filename, allow_pickle=True)[()]
    SSM, refPvModel = data['SSM'], data['refPvModel']
    return SSM, refPvModel


def sampleWeightsDistribution(size, std, abs_max):
    assert abs_max > 0
    assert std > 0
    assert size > 0
    w = np.random.normal(size=size) * std
    w = np.clip(w, -abs_max, abs_max)
    return w


def generateAortaPoints(meanShapePoints, pca_variance, pca_basis, w):
    points = meanShapePoints + \
        np.einsum('i,ijk->jk', w * np.sqrt(pca_variance) , pca_basis)
    return points

#def generateAortaPoints(SSM, w, refPvModel):
#    w = w[None,:]
#    points = SSM['model'].inverse_transform(w * np.sqrt(SSM['model'].explained_variance_))[0]
#    points = points.reshape(refPvModel.points.shape) * SSM['points_std'] + SSM['points_mean']
#    return points

def generateAortaCore(pca_basis, pca_variance, meanShape_pv, w):

    newModelPoints = generateAortaPoints(
        meanShape_pv.points,
        pca_variance,
        pca_basis,
        w)

    newModel_pv = pv.PolyData(newModelPoints, meanShape_pv.faces)

    return newModel_pv

#def generateAortaCore(SSM, w, refPvModel):
#    newModelPoints = generateAortaPoints(SSM, w, refPvModel)
#    newModel = refPvModel.copy()
#    newModel.points = newModelPoints
#    return newModel


#def reduceShapeCore(shape_pv, pca_basis, meanShape_pv, pca_variance):
#
#    point_std = np.array([20.03514457, 38.64412316, 35.46009813])
#
#    w = 1 / np.sqrt(pca_variance) * \
#        np.einsum('jk,ijk->i', shape_pv.points - meanShape_pv.points,
#        pca_basis / point_std**2)
#    w[-1] = 0
#
#    return w

def reduceShapeCore(SSM, pvModel):
    points = (pvModel.points - SSM['points_mean'] ) / SSM['points_std']
    points = points.reshape(-1)[None]
    w = SSM['model'].transform(points) / np.sqrt(SSM['model'].explained_variance_)
    return w[0]


def check_self_intersections(a):
    b = a.copy()
    b = b.compute_normals(
        consistent_normals=True,
        cell_normals=False,
        split_vertices=True
    )
    b.points = b.points + b['Normals']*1.
    c = a.intersection(b)
    return c[0].n_points



def get_existent_output_shapes_numbered():
    paths = glob.glob(osp.join(SAVE_DIR, '*.vtp'))
    modelNames = [osp.splitext(osp.split(path)[-1])[0] for path in paths]
    paths = [paths[i] for i in range(len(paths)) if not 'V' in modelNames[i] and not 'A' in modelNames[i] and not 'RN' in modelNames[i]]
    if len(paths) == 0:
        paths = ['/home/-1.vtp']
    numbers = [int(osp.splitext(osp.split(path)[-1])[0]) for path in paths]
    numbers = sorted(numbers)
    return numbers


def compute_output_filename():
    numbers = get_existent_output_shapes_numbered()
    return '{:05d}'.format(numbers[-1] + 1)






if __name__ == '__main__':

    filename = SSM_PATH

    meanShape_pv, pca_basis, pca_variance = load_SSM_h5(filename, refModelPath)

    def check_all(newModel_pv):
        nonManifold_edges = newModel_pv.extract_feature_edges(
                boundary_edges=False,
                non_manifold_edges=True,
                feature_edges=False,
                manifold_edges=False).points.shape[0]

        if nonManifold_edges > 0:
            print(f"{kk} iteration: {nonManifold_edges} non manifold edges")
            return nonManifold_edges

        self_intersections = check_intersections.check_intersections(newModel_pv)
        #self_intersections = check_self_intersections(newModel_pv)
        #if self_intersections > 20:
        if self_intersections > 0:
            print(f"{kk} iteration: {self_intersections} self intersections")
            return self_intersections

        return 0

    def generateAorta():
        kk = 1
        while True:
            w = sampleWeightsDistribution(
                pca_basis.shape[0],
                DEV,
                MAX_DEV)

            newModel_pv = generateAortaCore(pca_basis, pca_variance, meanShape_pv, w)

            # CHECKS
            inters = check_all(newModel_pv)
            kk += 1
            if inters == 0:
                break

        return newModel_pv

    def generateAorta2(w):
        newModel_pv = generateAortaCore(pca_basis, pca_variance, meanShape_pv, w)
        # CHECKS
        inters = check_all(newModel_pv)
        if inters>0:
            print(f"SELF_INTERSECTIONS: {inters}")
        return newModel_pv

    import scipy.stats
    ndim = 4
    sampler = scipy.stats.qmc.LatinHypercube(ndim, centered=True)
    n = 5
    bound = 3 # standard deviations
    ws = (sampler.random(n) - 0.5 - 1/(2*n) ) * 2 * bound
    print(ws)
    ws = np.around(ws.astype(np.float32), 3)
    ws = np.c_[ws, np.zeros((n,47-ndim))]

    import matplotlib.pyplot as plt
    plt.xlabel("Mode 0")
    plt.ylabel("Mode 1")
    plt.title('Drawn samples')
    plt.scatter(ws[:,0], ws[:,1])
    plt.show()

    names = []
    for i in range(ws.shape[0]):
        name = ''.join([f'{idx}_{coef:.2f},' for idx, coef in enumerate(ws[i]) if coef!=0])[:-1]
        if name == '': name = '0_0'
        names.append(name)

    for i in range(ws.shape[0]):
        print(ws[i])
        print(names[i], '\n')

    for kk, w in enumerate(ws):
        print(kk)
        m = generateAorta2(w)
        name = names[kk]
        filename = osp.join(SAVE_DIR, name + '.vtp')
        print(filename)
        if not osp.exists(filename):
            m.save(filename)
        else:
            print(f"{filename} already exists")


