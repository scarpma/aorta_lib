import pyvista as pv
import numpy as np
import vtk
from vtk import vtkMatrix4x4
from vtk import vtkMatrix3x3
from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter



def arrayFromVTKMatrix(vmatrix):
    """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
    The returned array is just a copy and so any modification in the array will not affect the input matrix.
    To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
    :py:meth:`updateVTKMatrixFromArray`.
    """
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray



def icp(model, refModel):
    # trova una trasformazione rigida con l'algoritmo icp
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(model)
    icp.SetTarget(refModel)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(20)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    # applica la trasformazione trovata al modello
    icpTransformFilter = vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(model)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()
    matrix = arrayFromVTKMatrix(icp.GetMatrix())

    regModel = pv.PolyData(icpTransformFilter.GetOutput())

    return regModel, matrix

