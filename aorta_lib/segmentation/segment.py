#!/usr/bin/env python
# coding: utf-8

log_l1_s = '='*5 + ' '
log_l1_e = ' '+ '='*5
indentation = '  '
def print_l1(s):
    print(log_l1_s + s + log_l1_e)
    return
def print_l2(s):
    print(indentation + s)
    return

print_l1("Loading libraries")

# # Datalist

import numpy as np
import torch
import monai
import ignite
import os.path as osp
from pathlib import Path
import matplotlib.pyplot as plt
import os
import subprocess
import argparse

import pyvista as pv
import vtk

import data
import utils

def marching_cubes(array):
    #array[:] = array[::-1]
    #array[:,:] = array[:,::-1]
    array = array.squeeze()
    assert len(array.shape)==3
    a = pv.wrap(array)
    contour = a.contour(
            isosurfaces=1,
            rng=(0.5,1),
            method='marching_cubes',
    )
    return contour

def windowedSincSmooth(mesh, iters=20, passband=0.01):
    smoothed = vtk.vtkWindowedSincPolyDataFilter()
    smoothed.SetInputData(mesh)
    smoothed.SetNumberOfIterations(iters)
    smoothed.SetPassBand(passband)
    smoothed.SetBoundarySmoothing(False)
    smoothed.SetFeatureEdgeSmoothing(False)
    smoothed.SetNonManifoldSmoothing(True)
    smoothed.SetNormalizeCoordinates(True)
    smoothed.Update()
    return pv.PolyData(smoothed.GetOutput())

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')

def segment(net_state_dict_path, image_paths, odir=None, overwrite_odir=False):

    if odir:
        odir = Path(odir)
        if overwrite_odir:
            run_command('rm -r '+str(odir))
        print_l1(f"Creating output folder: {odir}")
        odir.mkdir(exist_ok=False)

    net_state_dict_path = Path(net_state_dict_path).expanduser().absolute()

    pixdim = [1,1,1] # mm
    vol_size = (192,192,192)
    smoothing_factor = 0.5 # between 0 and 1
    device = torch.device("cuda")
    
    print_l1("Creating patient datalist")
    datalist = []
    for p in image_paths:
        pp = Path(p).expanduser().absolute()
        assert pp.exists(), pp
        name = pp.stem.split('.')[0]
        datalist.append(dict(name=name,image=p))
        
    for ii, sample in enumerate(datalist):
        print(f"{ii:4d}: {sample['name']}")
    
    # # Transforms
    keys = ['image']
    monai.utils.misc.set_determinism(seed=218341029)

    trans = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys),
        monai.transforms.EnsureChannelFirstd(keys),
        monai.transforms.Orientationd(keys, axcodes='RAS'),
        monai.transforms.Spacingd(keys, pixdim=pixdim, mode=['bilinear', 'nearest'] if len(keys)==2 else ['bilinear']),
        monai.transforms.ScaleIntensityRanged(
            'image',
            a_min=-350,
            a_max=800,
            b_min=0,
            b_max=1,
        ),
        #monai.transforms.CropForegroundd(keys, source_key='image'),
        monai.transforms.EnsureTyped(keys)
    ])
    
    print_l1("Creating dataset")
    ds = monai.data.Dataset(
        data=datalist,
        transform=trans,
    )
    print_l1("Creating data loader")
    loader = monai.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
    )
    
    # # Create Model, Loss, Optimizer, Trainer
    
    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    print_l1("Building neural network")
    UNet_meatdata = dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=monai.networks.layers.Norm.BATCH
    )
    model = monai.networks.nets.UNet(**UNet_meatdata).to(device)
    print_l1(f"Loading weights from path: {net_state_dict_path}")
    net_state_dict = torch.load(net_state_dict_path)['model']
    model.load_state_dict(net_state_dict)
    
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=vol_size,
        sw_batch_size=16,
        overlap=0.10,
        #mode='constant',
        mode='gaussian',
        sw_device=device,
        device='cpu',
        progress=True,
    )

    predictions = []
    for ii, data in enumerate(loader):
    
        print_l2(f"segmenting patient {ii}/{len(loader)} {data['name']}")
        image = data['image'].to(device)
        with torch.no_grad():
            y_pred = inferer(image, model)
    
        y_pred = monai.data.decollate_batch(y_pred)
        assert len(y_pred) == 1
        y_pred = y_pred[0] # choose batch n. 0 from list
        # apply softmax and choose aorta channel (no background)
        y_pred = monai.transforms.Activations(softmax=True)(y_pred)[1]
        y_pred = monai.transforms.AsDiscrete(threshold=0.5)(y_pred).unsqueeze(0) # [1, H, W, D]
    
        data = monai.data.decollate_batch(data)
        assert len(data) == 1
        data = data[0]
    
        y_pred_np = y_pred[0].detach().cpu().numpy()
        y_pred_np = monai.transforms.get_largest_connected_component_mask(y_pred_np)
    
        surface = marching_cubes(y_pred_np)
        surface.clear_data()
        surface = surface.transform(data['image'].meta['affine'].numpy())
        surface.points[:,:-1] = -surface.points[:,:-1] # LPS. Comment this to save in RAS (i think)
        surface = windowedSincSmooth(surface, iters=20, passband=10**(-4.*smoothing_factor))

        predictions.append(dict(name=datalist[ii]['name'], pred=y_pred_np,surface=surface))
    
        if odir:
            saver = monai.transforms.SaveImage(
                #output_dir='./',
                output_dir=odir,
                output_postfix='seg',
                output_ext='.nii.gz',
                resample=False,
                output_dtype=np.float32,
                separate_folder=False,
            )
            name = osp.basename(data['image'].meta['filename_or_obj']).split('.')[0]
            saver(y_pred, data['image'].meta)
            surface.save(osp.join(odir, f'{name}_seg.vtp'))
            os.symlink(datalist[ii]['image'], osp.join(odir, osp.split(datalist[ii]['image'])[-1]))
            fig = utils.plot_seg(datalist[ii]['image'], surface)
            fig.savefig(osp.join(odir, f'{name}.pdf'), dpi=300)

    
    return predictions
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segment a CT scan using a neural network')
    parser.add_argument('image_paths', type=str, nargs='+',
                        help='paths of the images to segment')
    parser.add_argument('--state_dict_path', type=str,
                        default='~/Martino/aorta_segmentation_v3/MMAR_TRIAL/logs_kfold0_slidingWindow/val__checkpoint_key_metric=0.9626.pt',
                        help='path of the state dictionary of the neural network to use')
    parser.add_argument('--odir', type=str, default=None,
                        help='directory to save results')
    parser.add_argument('--overwrite',action='store_true',
                        help='wether to overwrite odir or not')
    args = parser.parse_args()


    segment(
        net_state_dict_path=args.state_dict_path,
        image_paths=args.image_paths,
        odir=args.odir,
        overwrite_odir=args.overwrite,
    )


