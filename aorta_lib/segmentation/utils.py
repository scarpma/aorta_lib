import numpy as np
import nibabel as nib
import vtk
import pyvista as pv
pv.global_theme.multi_samples = 8
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.pyplot as plt 
import string

from vtk import vtkNIFTIImageReader
class NIFTIReader(pv.BaseReader):
    _class_reader = vtkNIFTIImageReader

plt.style.use(['science','nature'])
plt.rcParams['text.usetex'] = True
rc('text.latex', preamble='\\usepackage{amsmath}')
plt.style.use('science')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times"]})
cmap = mpl.colormaps.get_cmap("cet_coolwarm")


cont_params = dict(linewidth=1.1, alpha=0.6)
cont_params_pred = cont_params.copy(); cont_params_pred['alpha'] = 0.85
pad = 10
length = np.array([200]*2)
img_dims = np.array([200]*2)

def read_CT(path):
    # loads a nifti image in LPS coordinate system
    reader = NIFTIReader(path)
    import nibabel
    nii = nibabel.load(path)
    origin = nii.affine[:3, -1]
    spacing = nii.affine[:3, :3].diagonal()
    mesh = reader.read()
    mesh.SetOrigin(origin * np.sign(spacing))
    return mesh


#def read_CT(path, sign=-1):
#    reader = NIFTIReader(path)
#    import nibabel
#    nii = nibabel.load(path)
#    origin_ = nii.affine[:-1,-1]
#    origin = origin_
#    origin[:-1] = sign*origin[:-1]
#    mesh = reader.read()
#    mesh.origin = origin
#    return mesh

def create_roi(origin,n,u,r,length):
    # create cube to delineate region of interest (roi)
    roi_0 = pv.Cube(center=origin, x_length=length, y_length=length, z_length=length)
    # rotate cube with face in (1,0,0) direction to point as 'normals[-1]'
    n_0 = np.array([1,0,0]) # direction that will be parallel to `n`
    u_0 = np.array([0,0,1]) # direction that will be parallel to `u`
    r_0 = np.array([0,1,0]) # direction that will be parallel to `r`
    rotation_axis1 = np.cross(n_0, n)
    rotation_angle1 = np.arccos(np.dot(n_0, n))
    roi_1 = roi_0.rotate_vector(vector=rotation_axis1, angle=rotation_angle1*180/np.pi, point=origin)
    u_1 = pv.PointSet([u_0]).rotate_vector(vector=rotation_axis1, angle=rotation_angle1*180/np.pi)
    r_1 = pv.PointSet([r_0]).rotate_vector(vector=rotation_axis1, angle=rotation_angle1*180/np.pi)
    # additionally rotate cube with face in (0,0,1) direction to point as `u`
    rotation_axis2 = np.cross(u_1.points[0,:], u)
    rotation_angle2 = np.arccos(np.dot(u_1.points[0,:], u))
    roi_2 = roi_1.rotate_vector(vector=rotation_axis2, angle=rotation_angle2*180/np.pi, point=origin)
    u_2 = u_1.rotate_vector(vector=rotation_axis2, angle=rotation_angle2*180/np.pi).points[0,:]
    r_2 = r_1.rotate_vector(vector=rotation_axis2, angle=rotation_angle2*180/np.pi).points[0,:]
    return roi_2

def world_to_screen(points, origin, n, u, r, length, img_dims):
    # ------ project points to plane ------
    # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    dist = (points - origin).dot(n)     # distance from plane
    ppoints_from_plane_orig = (points-origin) - dist[...,None]*n # 3d vector from origin to ppoints
    plane_coords = np.dot(ppoints_from_plane_orig,np.array([u,-r]).T)
    norm_plane_coords = (plane_coords + length/2) / length * img_dims.T
    return norm_plane_coords

def create_img(vol, origin, length, n, u, r, img_dims, array_name):
    roi = create_roi(origin,n,u,r,length[0])
    plane = pv.Plane(center=origin,
                  direction=[1,0,0],
                  i_size=length[0], j_size=length[1],
                  i_resolution=img_dims[0], j_resolution=img_dims[1])

    n_0 = np.array([1,0,0]) # direction that will be parallel to `n`
    u_0 = np.array([0,0,-1]) # direction that will be parallel to `u`
    #u_0 = np.array([0,0,1]) # direction that will be parallel to `u`
    r_0 = np.array([0,-1,0]) # direction that will be parallel to `r`
    #r_0 = np.array([0,1,0]) # direction that will be parallel to `r`
    rotation_axis1 = np.cross(n_0, n)
    rotation_angle1 = np.arccos(np.dot(n_0, n))
    plane_1 = plane.rotate_vector(vector=rotation_axis1, angle=rotation_angle1*180/np.pi, point=origin)
    u_1 = pv.PointSet([u_0]).rotate_vector(vector=rotation_axis1, angle=rotation_angle1*180/np.pi)
    r_1 = pv.PointSet([r_0]).rotate_vector(vector=rotation_axis1, angle=rotation_angle1*180/np.pi)
    # additionally rotate cube with face in (0,0,1) direction to point as `u`
    rotation_axis2 = np.cross(u_1.points[0,:], u)
    rotation_angle2 = np.arccos(np.dot(u_1.points[0,:], u))
    plane_2 = plane_1.rotate_vector(vector=rotation_axis2, angle=rotation_angle2*180/np.pi, point=origin)
    u_2 = u_1.rotate_vector(vector=rotation_axis2, angle=rotation_angle2*180/np.pi).points[0,:]
    r_2 = r_1.rotate_vector(vector=rotation_axis2, angle=rotation_angle2*180/np.pi).points[0,:]
    
    plane_2 = plane_2.sample(vol).point_data_to_cell_data()
    return plane_2[array_name].reshape(*img_dims), plane_2

def order_line(line):
    assert line.lines.shape[0] > 0, 'empty line'
    assert line.lines[0] == line.lines.shape[0]-1, f'more than 1 polygon in line, {line.lines[0]}, {line.lines.shape}'
    n_segments = line.lines[0]
    ordered_line = line.copy()
    ordered_line.points = line.points[line.lines[1:]]
    ordered_line.lines = np.array([n_segments]+list(range(n_segments)))
    return ordered_line

def get_lines(cont, origin, n, u, r, length, img_dims):
    roi = create_roi(origin,n,u,r,length[0])
    cont_slice = cont.slice(normal=n, origin=origin).clip_box(roi, invert=False)
    contours = [] 
    try:
        slice_splitted = cont_slice.split_bodies()
    except:
        return contours
    for ii, cont in enumerate(slice_splitted):
        line = pv.PolyData(cont.extract_surface()).strip(join=True)
        ordered_line = order_line(line)
        contours.append(world_to_screen(ordered_line.points, origin, n, u, r, length, img_dims))
    return contours

def plot_segmentation_comparison(ct, pred_surf, origin, n, u, r, ax=None, fig=None, label_surf=None):
    img, plane = create_img(ct, origin, length, n, u, r, img_dims, 'NIFTI')
    if label_surf: seg_contours = get_lines(label_surf, origin, n, u, r, length, img_dims)
    pred_contours = get_lines(pred_surf, origin, n, u, r, length, img_dims)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', clim=(-400,600), zorder=0)
    if label_surf:
        for cont in seg_contours:
            ax.plot(*(cont.T), color=f'red', zorder=1, **cont_params)
    for cont in pred_contours:
        ax.plot(*(cont.T), color="C0" if label_surf else f'red', zorder=2, **cont_params_pred, linestyle=(1.5,(4,3)))

    all_points = []
    if label_surf:
        contours_available = [pred_contours, seg_contours]
    else:
        contours_available = [pred_contours]
    for conts in contours_available:
        for cont in conts:
            all_points += cont.tolist()
    all_points = np.array(all_points)
    xlims = np.array([all_points[:,0].min()-pad, all_points[:,0].max()+pad])
    ylims = np.array([all_points[:,1].min()-pad, all_points[:,1].max()+pad])
    dy = ylims.ptp()
    dx = xlims.ptp()
    if dy > dx:
      xlims = xlims.mean() + np.array([-1,1])*dy/2
    else:
      ylims = ylims.mean() + np.array([-1,1])*dx/2
        
        
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.axis('off')
    return

def plot_label_with_distance(pred_surf, label_surf=None, clim=(-1,1)):
    scale = 4 # to scale the image and increase the final resolution
    camera_dist = 550
    shift_from_sa = 130 #mm # set z coordinate as 120 mm from the sa vessels (max z coord found)
    scalar_bar_args = dict(title='Normal dist', font_family='Times',
                           width=0.1, height=0.3, vertical=True, position_x=0.1,
                           position_y=0.02, fmt='%.1f', label_font_size=18*scale, title_font_size=18*scale) 
    
    if label_surf is not None:
        # find points inside cells of pred_surf that are closest to points of label_surf
        closest_cells, closest_points = pred_surf.find_closest_cell(label_surf.points, return_closest_point=True)
        #d_exact = np.linalg.norm(label_surf.points - closest_points, axis=1)
        label_surf = label_surf.compute_normals(cell_normals=False)
        d_exact = np.einsum('ij,ij->i',label_surf.points - closest_points, label_surf['Normals'])
        label_surf["distances"] = d_exact

    camera_center = pred_surf.center
    camera_center[-1] = pred_surf.points[:,-1].max() - shift_from_sa
    pl = pv.Plotter(window_size=(np.array([350,600])*scale).astype(int), off_screen=True)
    #pl = pv.Plotter(window_size=(np.array([350,600])*scale).astype(int))
    pl.reset_camera()
    pv.set_plot_theme('document')
    if label_surf is None:
        pl.add_mesh(pred_surf, color='gray')
    else:
        pl.add_mesh(label_surf, opacity=1, clim=clim, cmap=cmap, scalar_bar_args=scalar_bar_args)
    #pl.add_mesh(pred_surf, color='white', opacity=0.4)
    #pl.add_title('Distance from label to prediction', font_size=6)
    camera_direction = np.array([1,-0.3,0.1])
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    pl.camera.position = camera_center - camera_direction * camera_dist
    pl.camera.focal_point = camera_center + camera_direction * camera_dist
    #pl.show()
    #pl.close()
    return pl.screenshot(None, return_img=True)

def plot_pred_and_label_surfaces(pred_surf, label_surf):
    scale = 4 # to scale the image and increase the final resolution
    camera_dist = 550
    shift_from_sa = 130 #mm # set z coordinate as 120 mm from the sa vessels (max z coord found)
    
    camera_center = pred_surf.center
    camera_center[-1] = pred_surf.points[:,-1].max() - shift_from_sa
    pl = pv.Plotter(window_size=(np.array([350,600])*scale).astype(int), off_screen=True)
    #pl = pv.Plotter(window_size=(np.array([350,600])*scale).astype(int))
    pl.reset_camera()
    pv.set_plot_theme('document')
    pl.add_mesh(pred_surf, color='red', opacity=0.6)
    pl.add_mesh(label_surf, opacity=.8, color='#3385ff')
    #pl.add_mesh(pred_surf, color='white', opacity=0.4)
    #pl.add_title('Distance from label to prediction', font_size=6)
    camera_direction = np.array([1,-0.3,-0.5])
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    pl.camera.position = camera_center - camera_direction * camera_dist
    pl.camera.focal_point = camera_center + camera_direction * camera_dist
    #pl.show()
    #pl.close()
    return pl.screenshot(None, return_img=True)


def plot_seg(image_path, pred_surf, label_surf=None, clim=(-1,1), mode='dist'):
    smoothing_factor = 0.5
    
    ct = read_CT(image_path)
    
    slice_span = pred_surf.points[:,-1].ptp() / 2 - 20
    origins = [pred_surf.center + np.array([0,0,i]) for i in np.linspace(-slice_span,+slice_span,16)]
    n = np.array([0,0,1])
    u = np.array([0,1,0])
    r = np.array([1,0,0])
    
    nrows, ncols = 4, 4
    #labels = ['(\\textbf{{{}}})'.format(c) for c in string.ascii_lowercase]
    labels = ['({})'.format(c) for c in string.ascii_lowercase]
    figure_size = np.array([180,100]) * 0.0393701 # specify in mm and then convert to inches
    fig, axs = plt.subplots(ncols=ncols+1, nrows=nrows, figsize=figure_size, dpi=300, width_ratios=[1]*ncols+[2.])
    gs = axs[0,-1].get_gridspec()
    for ax in axs[:,-1]:
        ax.remove()
    ax = fig.add_subplot(gs[:,-1])
    if mode=='dist':
        ax.imshow(plot_label_with_distance(pred_surf, label_surf, clim=clim))
    elif mode=='comp':
        ax.imshow(plot_pred_and_label_surfaces(pred_surf, label_surf))

    ax.axis('off')
    
    for ii, (origin, ax) in enumerate(zip(origins, axs[:,:-1].ravel())):
        slice_params = dict(origin=origin, n=n, u=u, r=r)
        plot_segmentation_comparison(ct, pred_surf, **slice_params, ax=ax, fig=fig, label_surf=label_surf)
        ax.text(-0.23,0.85, labels[ii], transform=ax.transAxes, fontsize=7)
    
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.0)
    #fig.subplots_adjust(hspace=0.03)
    return fig

if __name__ == "__main__":

    pass
