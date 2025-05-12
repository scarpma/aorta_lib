#!/usr/bin/env python
# coding: utf-8
#
#    _   _   _____    _____     _____   _____  
#   | \ | | |  __ \  |  __ \   / ____| |  __ \ 
#   |  \| | | |__) | | |__) | | |  __  | |  | |
#   | . ` | |  _  /  |  ___/  | | |_ | | |  | |
#   | |\  | | | \ \  | |      | |__| | | |__| |
#   |_| \_| |_|  \_\ |_|       \_____| |_____/ 
#                                              
#                                              

import numpy as np
import os
import os.path as osp
import glob
import sys
import torch
import pytorch3d
import pyvista as pv
import pyacvd

import mesh_ops
import optimizers_gpu_cg as optimizers_cg
import optimizers_cpu_direct as optimizers_direct

import meshIO
import plotter_lib
import loss as loss_lib

import ICP

from tqdm import tqdm
import time


import matplotlib.pyplot as plt


def register(
    src_cloud,
    trg_cloud,
    Niter,
    lossObj,
    save_period=25,
    first_iter=0,
    losses=[],
    src_mesh_pv=None,
    plotter=None,
    lamb=500,
    p='2',
    lr=0.01,
    mean=0.,
    std=0.,
    odir=None,
):

    device = src_cloud.device
    deform_verts = torch.full(src_cloud.points_packed().shape, 0.0, device=device, requires_grad=True)

    if src_cloud.points_packed().shape[0] > 8000 :
        ## RUN WITH CG ON GPU IF WE HAVE MORE THAN 8000 VERTICES
        print("GRADIENT PRECONDITIONING: RUNNING WITH CG ON GPU (PARALLEL)")
        optimizer = optimizers_cg.myAdam([deform_verts], lamb=lamb, p=p, init_src_cloud=src_cloud, pv_src_mesh=src_mesh_pv, lr=lr)
    else :
        ## RUN WITH DIRECT METHOD ON CPU (SCIPY) IF WE HAVE LESS THAN 8000 VERTICES
        print("GRADIENT PRECONDITIONING: RUNNING WITH DIRECT METHOD ON CPU (SEQUENTIAL)")
        optimizer = optimizers_direct.myAdam([deform_verts], lamb=lamb, p=p, init_src_cloud=src_cloud, pv_src_mesh=src_mesh_pv, lr=lr)

    # Plot period for the losses
    loop = tqdm(range(first_iter, Niter+first_iter), ascii=True, ncols=80)

    for i in loop:

        # set to zero the gradients
        optimizer.zero_grad()

        # deform the mesh according to last iteration
        new_src_cloud = src_cloud.offset(deform_verts)

        # compute loss
        loss = lossObj.compute_loss(new_src_cloud)
        loop.set_description('total_loss = %.7f' % loss)

        # optimization step
        loss.backward()
        optimizer.step()

        # logs    
        losses.append(loss.item())
        if plotter is not None: plotter.plot(i, losses)
        if save_period != 0 and i % save_period == 0:
            new_verts = new_src_cloud.points_list()[0].detach().cpu().numpy() * std + mean
            faces = src_mesh_pv.faces
            src_deformed_pv = pv.PolyData(new_verts, faces)
            src_deformed_pv.save(osp.join(odir, f'reg_{i}.vtp'))

    return new_src_cloud.points_list()[0].detach().cpu().numpy(), losses



def create_pv_meshes(
    src_mesh_path,
    trg_mesh_path,
    decimate_values,
    initial_icp=False,
):

    trg_meshes_pv = []
    src_meshes_pv = []
    trg_meshes_pv.append(pv.read(trg_mesh_path))
    src_meshes_pv.append(pv.read(src_mesh_path))

    transform = None
    if initial_icp:
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('z',180, degrees=True)
        r = r.as_matrix()
        r_ = np.zeros((4,4))
        r_[:3,:3] = r
        r_[-1,-1] = 1.
        trg_meshes_pv[0] = trg_meshes_pv[0].transform(r_)
        trg_meshes_pv[0], transform = ICP.icp(trg_meshes_pv[0], src_meshes_pv[0])
        transform = transform @ r_


    for dec in decimate_values[::-1]:
        # target meshes creation
        clus = pyacvd.Clustering(trg_meshes_pv[-1].copy())
        clus.subdivide(1)
        clus.cluster(dec)
        trg_meshes_pv.insert(0, clus.create_mesh())

        # source meshes creation
        clus = pyacvd.Clustering(src_meshes_pv[-1].copy())
        clus.subdivide(1)
        clus.cluster(dec)
        src_meshes_pv.insert(0, clus.create_mesh())

    return src_meshes_pv, trg_meshes_pv, transform



def reg_multiscale(
    src_mesh_path,
    trg_mesh_path,
    decimate_values=[4000,18000],
    Niters=[700,1000,1500],
    lambs=[120,80,50],
    p='1',
    lrs=[0.007, 0.01, 0.02],
    save_periods=0,
    odir='.',
    plot_period=0,
    device=torch.device('cuda'),
    wsa=0.1,
    wio=0.1,
    initial_icp=False,
):

    if isinstance(save_periods, int):
        save_periods = [save_periods]*len(Niters)

    if any([save_period > 0 for save_period in save_periods]):
        assert odir != '.', f"save_periods != 0 implies outputdir ({odir}) to not be '.'"
        if not osp.isdir(odir):
            os.makedirs(odir)

    src_name = osp.basename(src_mesh_path).split('.')[0]
    trg_name = osp.basename(trg_mesh_path).split('.')[0]

    #### INITIALIZE TIMINGS ####
    print(time.ctime())
    times = []
    times.append(time.time())

    #### CREATE MESHES AT LOWER RESOLUTION AND FIND BOUNDARIES ####
    src_meshes_pv, trg_meshes_pv, transform = create_pv_meshes(
        src_mesh_path,
        trg_mesh_path,
        decimate_values,
        initial_icp)
    #trg_bou_idxs = []
    #for ii in range(len(src_meshes_pv)):
    #    trg_bou_idxs.append(mesh_ops.get_bou_idxs(trg_meshes_pv[ii]))

    odir_iterations = osp.join(odir, src_name+'_to_'+trg_name)
    os.makedirs(odir_iterations, exist_ok=True)
    if odir is not None:
        trg_meshes_pv[-1].save(osp.join(odir_iterations,trg_name+'_target.vtp'))
        src_meshes_pv[-1].save(osp.join(odir_iterations,src_name+'_source.vtp'))

    #### INITIALIZE SECONDARY TOOLS ####
    prev_loss = []
    means = np.empty(len(trg_meshes_pv))
    stds = np.empty(len(trg_meshes_pv))
    plotter = plotter_lib.Plotter(plot_period=plot_period) if plot_period > 0 else None

    #### MAIN LOOP ####
    for i in range(len(src_meshes_pv)):
        if i > 0:
            src_mesh_pv_coarser_reg = src_meshes_pv[0].copy()
            src_mesh_pv_coarser_reg.points = new_verts * stds[i-1] + means[i-1]
            upmorph = mesh_ops.upsample_morph(
                origCoarseMesh=src_meshes_pv[i-1],
                newCoarseMesh=src_mesh_pv_coarser_reg,
                fineMesh=src_meshes_pv[i])
            # SMOOTH
            upmorph = mesh_ops.laplacianSmooth(upmorph, 300, 0.01)
        else:
            upmorph = src_meshes_pv[i]

        src_bou_idxs = mesh_ops.get_bou_idxs(upmorph)

        src_cloud, trg_cloud, mean, std = meshIO.createInputData(upmorph, trg_meshes_pv[i], device)
        means[i] = mean
        stds[i] = std

        #lossObj = loss_lib.Loss_fn_bou_match(trg_cloud, src_bou_idxs, trg_bou_idxs[i], wsa=wsa, wio=wio)
        lossObj = loss_lib.Loss_fn_bou_match_nonSym(trg_cloud)

        print("NUMBER OF VERTICES: ", src_cloud.points_packed().shape[0])
        new_verts, loss = register(
            src_cloud=src_cloud,
            trg_cloud=trg_cloud,
            lossObj=lossObj,
            Niter=Niters[i],
            lamb=lambs[i],
            p=p,
            lr=lrs[i],
            save_period=save_periods[i],
            odir=odir_iterations,
            first_iter=Niters[:i].sum(),
            losses=prev_loss,
            src_mesh_pv=upmorph,
            plotter=plotter,
            mean=means[i],
            std=stds[i],
        )
        prev_loss = loss
        times.append(time.time())

    loss__ = np.array(loss)
    np.savetxt(osp.join(odir_iterations, f"loss.txt"), np.c_[np.arange(loss__.shape[0]), loss__])

    #### APPLY LAST CHANGES TO MESH ####
    final_src_mesh_pv = src_meshes_pv[-1]
    final_src_mesh_pv.points = new_verts * stds[-1] + means[-1]

    #### REPORT TIMINGS ####
    print(time.ctime())
    for i in range(1, len(times)):
        print(f'reg #{i} duration: {(times[i] - times[i-1])/60.:.1f}')
    print(f'total duration:    {(times[-1] - times[0])/60.:.1f}')

    return final_src_mesh_pv, transform




if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(
        prog='NRPGD',
        description="Non rigid registration between surfaces using"+
        "Pytorch and Pytorch3d. Optinal additional constraints for"+
        "landmark registration",
    )

    parser.add_argument(
        '-s', '--sourcePath',
        type=str,
        required=True,
        dest='srcPath',
    )
    parser.add_argument(
        '-t', '--targetPath',
        type=str,
        required=True,
        dest='trgPath',
    )
    parser.add_argument(
        '-o', '--odir',
        type=str,
        required=False,
        default='.',
        help='output directory relative to .',
        dest='odir',
    )
    parser.add_argument(
        '-d', '--decimateValues',
        type=int,
        required=False,
        default=[4000,18000],
        nargs='+',
        #nargs='*',
        help='number of points of the lower resolution meshes',
        dest='decimate_values',
    )
    parser.add_argument(
        '-N', '--iterations',
        type=int,
        required=False,
        default=[700,1000,1500],
        nargs='+',
        help='number of iterations at each mesh resolution (from low to high)',
        dest='Niters',
    )
    parser.add_argument(
        '-l', '--lambdas',
        type=int,
        required=False,
        default=[120,80,50],
        nargs='+',
        help='lambda values for each resolution (from low to high)',
        dest='lambs',
    )
    parser.add_argument(
        '-lr', '--learningRates',
        type=float,
        required=False,
        default=[0.007, 0.01, 0.02],
        nargs='+',
        help='learning rate values for each resolution (from low to high)',
        dest='lrs',
    )
    parser.add_argument(
        '-sp', '--savePeriods',
        type=int,
        required=False,
        default=[70,100,150],
        nargs='+',
        help='number of iterations between mesh save to disk for each resolution (from low to high)',
        dest='save_periods',
    )
    parser.add_argument(
        '-pp', '--plotPeriod',
        type=int,
        required=False,
        default=10,
        help='number of iterations between each plot update (can be 0 to not plot)',
        dest='plot_period',
    )
    parser.add_argument(
        '-icp', '--initial_icp',
        type=bool,
        required=False,
        default=False,
        help='whether to perform or not an initial icp registration from target to source',
        dest='initial_icp',
    )

    args = parser.parse_args()
    locals().update(args.__dict__)
    Niters = np.array(Niters)

    print(
        "\n"+
        "  _   _   _____    _____     _____   _____  \n"+
        " | \ | | |  __ \  |  __ \   / ____| |  __ \  \n"+
        " |  \| | | |__) | | |__) | | |  __  | |  | |  \n"+
        " | . ` | |  _  /  |  ___/  | | |_ | | |  | |  \n"+
        " | |\  | | | \ \  | |      | |__| | | |__| |  \n"+
        " |_| \_| |_|  \_\ |_|       \_____| |_____/  \n"+
        "\n")






    #### PARAMETERS OF THE REGISTRATION ####
    p = '1'
    device = 'cuda'


    device = torch.device(device)
    if odir == '.': save_periods = [0]*len(Niters)
    if osp.isdir(odir) and odir != '.':
        if input(f'directory {odir} already exists. Continue ?') != 'y':
            sys.exit()
    elif odir != '.':
        os.makedirs(odir)

    srcName = osp.basename(srcPath).split('.')[0]
    trgName = osp.basename(trgPath).split('.')[0]
    print('#'*5, f' Registering {srcName} to {trgName} ', '#'*5)
    reg_mesh = reg_multiscale(
        src_mesh_path   = srcPath,
        trg_mesh_path   = trgPath,
        decimate_values = decimate_values,
        Niters          = Niters,
        lambs           = lambs,
        p               = p,
        lrs             = lrs,
        save_periods    = save_periods,
        plot_period     = plot_period,
        odir            = odir,
        device          = device,
        initial_icp     = args.initial_icp,
    )
    reg_mesh.save(osp.join(odir,f"{srcName}_to_{trgName}.vtp"))















