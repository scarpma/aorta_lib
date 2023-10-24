import torch
import numpy as np
import pyvista as pv
import mesh_ops
import reg
import os
import os.path as osp
from pathlib import Path
import glob
from aorta_lib import utils



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
        nargs='+',
        required=True,
        dest='trgPath',
    )
    parser.add_argument(
        '-o', '--odir',
        type=str,
        required=True,
        dest='odir',
    )
    args = parser.parse_args()

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")


    #### PARAMETERS OF THE REGISTRATION ####
    decimate_values = []
    Niters = np.array([400])
    lambs = [120]
    lambs = [60]
    lrs = [0.007]
    p = '1'
    save_periods = [50]
    plot_period = 50


    #### PATHS ####
    odir = Path(args.odir)
    if odir.exists():
        if input(f'directory {odir} already exists. Continue ?') != 'y':
            sys.exit()
        utils.run_command('rm -r '+str(odir))
    odir.mkdir()
    


    src_mesh_path = Path(args.srcPath)
    #os.symlink(src_mesh_path, odir / src_mesh_path.name )
    m = mesh_ops.remesh(pv.read(args.srcPath).connectivity(largest=True), target_edge_len=1.4) # mm
    new_src_mesh_path = odir / src_mesh_path.name
    m.save(new_src_mesh_path)
    

    for trg_mesh_path in [Path(p) for p in args.trgPath]:
        #os.symlink(trg_mesh_path, odir / trg_mesh_path.name )
        m_t = mesh_ops.remesh(pv.read(trg_mesh_path).connectivity(largest=True), target_edge_len=1.4) # mm
        new_trg_mesh_path = odir / (trg_mesh_path.stem + '_trg' + trg_mesh_path.suffix )
        m_t.save(new_trg_mesh_path)


        print('#'*5, f' Registering {new_src_mesh_path.stem} to {trg_mesh_path.stem} ', '#'*5)

        reg_mesh, _ = reg.reg_multiscale(
            src_mesh_path   = new_src_mesh_path,
            trg_mesh_path   = new_trg_mesh_path,
            decimate_values = decimate_values,
            Niters          = Niters,
            lambs           = lambs,
            p               = p,
            lrs             = lrs,
            save_periods    = save_periods,
            plot_period     = plot_period,
            odir            = odir,
            device          = device,
        )

        reg_mesh.save(odir / (trg_mesh_path.stem + '_reg' + trg_mesh_path.suffix))

