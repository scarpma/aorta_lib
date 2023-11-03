import vtk
from vmtk import vmtkscripts
from vmtk import vtkvmtk
from vmtk import pypes
import pyvista as pv
from aorta_lib import utils
import numpy as np


def vmtkAnalysis(ifile, odir, from_zero=True, onlycl=False, skip=True):

    name = osp.splitext(osp.basename(ifile))[0]

    cl_file = osp.join(odir, name + '_cl.vtp')
    if skip:
        if osp.isfile(cl_file):
            print(cl_file, ' exists. skipping')
            return 0
    vor_file = osp.join(odir, name + '_vor.vtp')
    rs_file = osp.join(odir, name + '_rs.vtp')
    clipped_file = osp.join(odir, name + '_clipped.vtp')
    metrics_file = osp.join(odir, name + '_metrics.vtp')
    mapping_file = osp.join(odir, name + '_mapping.vtp')
    clproj_file = osp.join(odir, name + '_clproj.vtp')
    patching_file = osp.join(odir, name + '_patching.vtp')
    distance_file = osp.join(odir, name + '_distance.vtp')
    img_file = osp.join(odir, name + '_img.vti')

    m = pv.read(ifile)
    m = m.clean(absolute=True, tolerance=0.001)
    bous = m.extract_feature_edges(boundary_edges=True,
                                             non_manifold_edges=False,
                                             feature_edges=False,
                                             manifold_edges=False)
    bous = list(bous.split_bodies())

    bous = utils.order_bous(bous)

    centers = np.zeros((len(bous), 3))
    for i in np.arange(len(bous)):
        centers[i] = utils.computePolylineBarycenter(bous[i])

    #sourceids = '{}'.format(edges_idx[0])
    #targetids = '{} {} {} {}'.format(*edges_idx[1:])
    #targetids = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(*centers[1],*centers[2],*centers[3],*centers[4])
    #sourceids = '{} {} {}'.format(*centers[0])
    targetids = list(centers[1:].reshape((-1)))
    sourceids = list(centers[0])

    #centers_file = osp.join(odir, name + '_centers.vtp')
    #pv.PolyData(centers).save(centers_file)


def centerline(m, sa=False, resample=False):

    #m = m.clean(absolute=True, tolerance=0.001)
    bous = m.extract_feature_edges(boundary_edges=True,
                                   non_manifold_edges=False,
                                   feature_edges=False,
                                   manifold_edges=False)
    bous = list(bous.split_bodies())
    bous = utils.order_bous(bous)

    centers = np.zeros((len(bous), 3))
    for i in np.arange(len(bous)):
        centers[i] = utils.computePolylineBarycenter(bous[i])

    #sourceids = '{}'.format(edges_idx[0])
    #targetids = '{} {} {} {}'.format(*edges_idx[1:])
    #targetids = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(*centers[1],*centers[2],*centers[3],*centers[4])
    #sourceids = '{} {} {}'.format(*centers[0])
    sourceids = list(centers[0])
    if sa:
        targetids = list(centers[1:].reshape((-1)))
    else:
        targetids = list(centers[1].reshape((-1)))

    # vmtkSurfaceCapper
    SurfaceCapper = vmtkscripts.vmtkSurfaceCapper()
    #Viewer = vmtkscripts.vmtkCenterlineViewer()
    SurfaceCapper.Surface = m
    SurfaceCapper.Method = 'centerpoint'
    SurfaceCapper.Interactive = 0
    print('capping')
    SurfaceCapper.Execute()
    m_capped = SurfaceCapper.Surface

    # vmtkCenterlines
    Centerlines = vmtkscripts.vmtkCenterlines()
    Centerlines.Surface = m_capped
    Centerlines.AppendEndPoints = 1
    Centerlines.SeedSelectorName = "pointlist"
    Centerlines.SourcePoints = sourceids
    Centerlines.TargetPoints = targetids
    Centerlines.CheckNonManifold = 1
    if resample:
        Centerlines.Resampling = 1
        Centerlines.ResamplingStepLength = resample

    print('computing centerlines')
    Centerlines.Execute()

    # vmtkCenterlineAttributes
    temp = vmtkscripts.vmtkCenterlineAttributes()
    temp.Centerlines = Centerlines.Centerlines
    print('computing centerlines attributes')
    temp.Execute()
    Centerlines = temp
    del temp

    # vmtkBranchExtractor
    temp = vmtkscripts.vmtkBranchExtractor()
    temp.Centerlines = Centerlines.Centerlines
    print('splitting centerlines')
    temp.Execute()
    Centerlines = temp
    del temp

    # vmtkCenterlineGeometry
    temp = vmtkscripts.vmtkCenterlineGeometry()
    temp.Centerlines = Centerlines.Centerlines
    print('computing centerlines geometry')
    temp.Execute()
    Centerlines = temp
    del temp

    centerlines =  pv.PolyData(Centerlines.Centerlines)
    #voronoi =  pv.PolyData(Centerlines.VoronoiDiagram)

    return centerlines#, voronoi

def bif_rs(centerlines):
    bif_rs = vmtkscripts.vmtkBifurcationReferenceSystems()
    bif_rs.Centerlines = centerlines
    print('computing bifurcation rs')
    bif_rs.Execute()
    bif_rs = pv.PolyData(bif_rs.ReferenceSystems)
    return bif_rs


def branch_clipper(m, centerlines, clip_value=0.):
    #errors = 1
    #clip_value = 0.
    #count = 0
    #delta = 5.0
    #while errors != 0 and count < 8:
    branch_clipper = vmtkscripts.vmtkBranchClipper()
    branch_clipper.Surface = m
    branch_clipper.Centerlines = centerlines
    branch_clipper.ClipValue = clip_value
    #branch_clipper.UseRadiusInformation = False
    print('clipping surface with clipvalue = ', clip_value)
    branch_clipper.Execute()
    clipped_pv = pv.PolyData(branch_clipper.Surface)
    #errors = check_clipping(clipped_pv)
    #clip_value += delta
    #count += 1
    #if count > 4:
    #    clip_value = 0.
    #    delta /= 2.
    return clipped_pv


def branch_metrics(m, centerlines):
    branch_metrics = vmtkscripts.vmtkBranchMetrics()
    branch_metrics.Surface = m
    branch_metrics.ComputeAngularMetric = 1
    branch_metrics.ComputeAbscissaMetric = 1
    branch_metrics.Centerlines = centerlines
    print('computing branch metrics')
    branch_metrics.Execute()
    metrics = pv.PolyData(branch_metrics.Surface)
    return metrics

def proj_cldata(m, centerlines, use_radius : int):
    m_clproj = vmtkscripts.vmtkSurfaceCenterlineProjection()
    m_clproj.Surface = m
    m_clproj.Centerlines = centerlines
    assert (use_radius == 0 or use_radius == 1)
    m_clproj.UseRadiusInformation = use_radius
    print('projecting cl data to surface')
    m_clproj.Execute()
    return pv.PolyData(m_clproj.Surface)


"""
    m_mapped = vmtkscripts.vmtkBranchMapping()
    m_mapped.Surface = branch_metrics.Surface
    m_mapped.Centerlines = centerlines
    m_mapped.ReferenceSystems = bif_rs
    print('computing branch mappings')
    m_mapped.Execute()
    pv.PolyData(m_mapped.Surface).save(mapping_file)



    ## distance_arg = f"vmtkdistancetocenterlines \
    ## -ifile {ifile} \
    ## -centerlinesfile {cl_file} \
    ## -combined {1} \
    ## -projectarrays {1} \
    ## -ofile {distance_file}"

    ## patching_arg = f"vmtkbranchpatching \
    ## -ifile {mapping_file} \
    ## -groupidsarray GroupIds \
    ## -longitudinalmappingarray HarmonicMapping \
    ## -circularmappingarray AngularMetric \
    ## -longitudinalpatchsize 5 \
    ## -circularpatches 60 \
    ## -ofile {patching_file} \
    ## -patcheddatafile {img_file}"

    ## # apro l'immagine .vti e la converto in un dizionario python
    ## imageReader = vmtkscripts.vmtkImageReader()
    ## imageReader.InputFileName = img_file
    ## imageReader.Execute()
    ## imageNumpyAdaptor = vmtkscripts.vmtkImageToNumpy()
    ## imageNumpyAdaptor.Image = imageReader.Image
    ## imageNumpyAdaptor.Execute()
    ## numpyImage = imageNumpyAdaptor.ArrayDict
    ##
    ## # accedo all'immagine vera e propria nel dizionario
    ## img_array = numpyImage['PointData']['WSS'].squeeze()
    ## # plot in .png
    ## plt.imshow(img_array, aspect=1)
    ## plt.savefig('immagine.png', dpi=150)

    return 0
"""


if __name__ == "__main__":
    m = pv.read('../examples/alignedDatasetBiomarkers_new/A2.vtp')
    #cl = centerline(m, sa=False, resample=1.)
    cl = centerline(m, sa=True, resample=1.)
    cl.save("cl.vtp")
    #bif_rs = bif_rs(cl)
    clipped = branch_clipper(m, cl, clip_value=0.)
    metrics = branch_metrics(clipped, cl)
    metrics = proj_cldata(metrics, cl, 1)
    metrics.save('metrics.vtp')





