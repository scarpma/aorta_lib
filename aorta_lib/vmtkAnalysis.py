import sys
import vtk
from vmtk import vmtkscripts
from vmtk import vtkvmtk
from vmtk import pypes
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os.path as osp
from pathlib import Path
import os
import glob

from aorta_lib import utils


def check_clipping(clipped_pv):
    errors = 0
    submodels = clipped_pv.split_bodies()
    a = [submodel['GroupIds'] for submodel in submodels]
    gIds = []
    submodelIdx = []
    for kk, submodel_gids in enumerate(a):
        # check for cutted small regions
        if submodel_gids.size < 30:
            print('Found small isolated region.')
            errors += 1
            break
        if np.all(submodel_gids==submodel_gids[0]):
            pass
        # check for uncutted small regions with
        # different GroupIds
        else:
            print('Found small isolated region.')
            errors += 1
            break

        # check for holes and manifoldness
        boundary_edges = submodels[kk].extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False)
        boundary_edges_list = list(boundary_edges.split_bodies())
        if len(boundary_edges_list) != 2:
            print('Found something without 2 boundaries')
            errors += 1
            break
        nonManifold_edges = submodels[kk].extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=True,
            feature_edges=False,
            manifold_edges=False)
        if nonManifold_edges.points.shape[0] != 0:
            print('Found something non manifold')
            errors += 1
            break

    return errors


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

    if from_zero or onlycl:
        SurfaceCapper = vmtkscripts.vmtkSurfaceCapper()
        Centerlines = vmtkscripts.vmtkCenterlines()
        #Viewer = vmtkscripts.vmtkCenterlineViewer()
        SurfaceCapper.Surface = m
        SurfaceCapper.Method = 'centerpoint'
        SurfaceCapper.Interactive = 0
        print('capping')
        SurfaceCapper.Execute()

        m_capped = SurfaceCapper.Surface
        Centerlines.Surface = m_capped
        Centerlines.AppendEndPoints = 1
        Centerlines.SeedSelectorName = "pointlist"
        Centerlines.SourcePoints = sourceids
        Centerlines.TargetPoints = targetids
        Centerlines.CheckNonManifold = 1
        print('computing centerlines')
        Centerlines.Execute()
        if not onlycl: pv.PolyData(Centerlines.VoronoiDiagram).save(vor_file)

        temp = vmtkscripts.vmtkCenterlineAttributes()
        temp.Centerlines = Centerlines.Centerlines
        print('computing centerlines attributes')
        temp.Execute()
        Centerlines = temp
        del temp

        temp = vmtkscripts.vmtkBranchExtractor()
        temp.Centerlines = Centerlines.Centerlines
        print('splitting centerlines')
        temp.Execute()
        Centerlines = temp
        del temp

        temp = vmtkscripts.vmtkCenterlineGeometry()
        temp.Centerlines = Centerlines.Centerlines
        print('computing centerlines geometry')
        temp.Execute()
        Centerlines = temp
        del temp
        pv.PolyData(Centerlines.Centerlines).save(cl_file)
        if onlycl: return 0
        #Viewer.Centerlines = Centerlines.Centerlines
        #Viewer.Execute()

        bif_rs = vmtkscripts.vmtkBifurcationReferenceSystems()
        bif_rs.Centerlines = Centerlines.Centerlines
        print('computing bifurcation rs')
        bif_rs.Execute()
        pv.PolyData(bif_rs.ReferenceSystems).save(rs_file)

        centerlines = Centerlines.Centerlines
        bif_rs = bif_rs.ReferenceSystems
    else:
        centerlines = pv.read(cl_file)
        bif_rs = pv.read(rs_file)

    errors = 1
    clip_value = 0.
    count = 0
    delta = 5.0
    while errors != 0 and count < 8:
        branch_clipper = vmtkscripts.vmtkBranchClipper()
        branch_clipper.Surface = m
        branch_clipper.Centerlines = centerlines
        branch_clipper.ClipValue = clip_value
        #branch_clipper.UseRadiusInformation = False
        print('clipping surface with clipvalue = ', clip_value)
        branch_clipper.Execute()
        clipped_pv = pv.PolyData(branch_clipper.Surface)
        errors = check_clipping(clipped_pv)
        clip_value += delta
        count += 1
        clipped_pv.save(clipped_file)
        if count > 4:
            clip_value = 0.
            delta /= 2.


    branch_metrics = vmtkscripts.vmtkBranchMetrics()
    branch_metrics.Surface = branch_clipper.Surface
    branch_metrics.ComputeAngularMetric = 1
    branch_metrics.ComputeAbscissaMetric = 1
    branch_metrics.Centerlines = centerlines
    print('computing branch metrics')
    branch_metrics.Execute()
    pv.PolyData(branch_metrics.Surface).save(metrics_file)

    m_mapped = vmtkscripts.vmtkBranchMapping()
    m_mapped.Surface = branch_metrics.Surface
    m_mapped.Centerlines = centerlines
    m_mapped.ReferenceSystems = bif_rs
    print('computing branch mappings')
    m_mapped.Execute()
    pv.PolyData(m_mapped.Surface).save(mapping_file)

    m_clproj = vmtkscripts.vmtkSurfaceCenterlineProjection()
    m_clproj.Surface = branch_metrics.Surface
    m_clproj.Centerlines = centerlines
    m_clproj.UseRadiusInformation = 1
    print('projecting cl data to surface')
    m_clproj.Execute()
    pv.PolyData(m_mapped.Surface).save(clproj_file)



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


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        prog='VMTK_Analysis',
        description="",
    )

    parser.add_argument(
        '-i',
        type=str,
        nargs='+',
        required=True,
        dest='ifile',
    )

    parser.add_argument(
        '-o',
        type=str,
        required=True,
        dest='odir',
    )

    args = parser.parse_args()

    odir = Path(args.odir)
    if not odir.exists():
        odir.mkdir()
    else:
        if input(f'Output dir {odir} already exists. Continue ?').lower()=='y':
            pass
        else:
            sys.exit(1)

    ifiles = sorted(args.ifile)

    for kk, ifile in enumerate(ifiles):
        print('*' * 30)
        print(kk, ifile)
        print('*' * 30)
        ## if input('y/n : ')=='y':
        ##     odir = osp.split(ifile)[0]
        ##     vmtkAnalysis(ifile, odir, from_zero=True, onlycl=False, skip=True)
        ## else:
        ##     continue
        vmtkAnalysis(ifile, odir, from_zero=True, onlycl=False, skip=False)

