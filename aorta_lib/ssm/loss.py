import torch
from pytorch3d.loss import chamfer_distance
from myChamfer import chamfer_distance_nonSym

class Loss_fn_bou_match():
    def __init__(self, trg_mesh, src_bou_idxs_=None, trg_bou_idxs_=None, wsa=0.1, wio=0.1):

        self.trg_points = trg_mesh.points_packed().unsqueeze(0)
        self.wsa = wsa
        self.wio = wio

        if src_bou_idxs_ is None:
            self.bou = False
        else:
            self.bou = True
            trg_bou_idxs = [torch.from_numpy(trg_bou_idxs_[i]).to(trg_mesh.device) for i in range(len(trg_bou_idxs_))]
            src_bou_idxs = [torch.from_numpy(src_bou_idxs_[i]).to(trg_mesh.device) for i in range(len(src_bou_idxs_))]

            self.trg_asc_bou_points = self.trg_points[0,trg_bou_idxs[0]].unsqueeze(0)
            self.trg_disc_bou_points = self.trg_points[0,trg_bou_idxs[1]].unsqueeze(0)
            self.trg_sa1_bou_points = self.trg_points[0,trg_bou_idxs[2]].unsqueeze(0)
            self.trg_sa2_bou_points = self.trg_points[0,trg_bou_idxs[3]].unsqueeze(0)
            self.trg_sa3_bou_points = self.trg_points[0,trg_bou_idxs[4]].unsqueeze(0)

            self.src_bou_idxs = src_bou_idxs

    def compute_loss(self, deformed_mesh):

        wsa = self.wsa
        wio = self.wio
        src_points = deformed_mesh.points_packed().unsqueeze(0)
        loss_chamfer, _ = chamfer_distance(self.trg_points, src_points)

        if not self.bou:
            return loss_chamfer

        src_asc_bou_points = src_points[0,self.src_bou_idxs[0]].unsqueeze(0)
        src_disc_bou_points = src_points[0,self.src_bou_idxs[1]].unsqueeze(0)
        src_sa1_bou_points = src_points[0,self.src_bou_idxs[2]].unsqueeze(0)
        src_sa2_bou_points = src_points[0,self.src_bou_idxs[3]].unsqueeze(0)
        src_sa3_bou_points = src_points[0,self.src_bou_idxs[4]].unsqueeze(0)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer_asc, _ = chamfer_distance(self.trg_asc_bou_points, src_asc_bou_points)
        loss_chamfer_disc, _ = chamfer_distance(self.trg_disc_bou_points, src_disc_bou_points)
        loss_chamfer_sa1, _ = chamfer_distance(self.trg_sa1_bou_points, src_sa1_bou_points)
        loss_chamfer_sa2, _ = chamfer_distance(self.trg_sa2_bou_points, src_sa2_bou_points)
        loss_chamfer_sa3, _ = chamfer_distance(self.trg_sa3_bou_points, src_sa3_bou_points)

        # mesh laplacian smoothing
        #loss_laplacian = mesh_ops.discrete_dirichlet_energy(src_mesh)

        # Weighted sum of the losses
        loss = loss_chamfer + wio * loss_chamfer_asc + wio * loss_chamfer_disc + \
                wsa * loss_chamfer_sa1 + wsa * loss_chamfer_sa2 + wsa * loss_chamfer_sa3

        return loss


class Loss_fn_bou_match_nonSym():
    def __init__(self, trg_mesh, src_bou_idxs_=None, trg_bou_idxs_=None, wsa=0.1, wio=0.1):

        self.trg_points = trg_mesh.points_packed().unsqueeze(0)
        self.wsa = wsa
        self.wio = wio

        if src_bou_idxs_ is None:
            self.bou = False
        else:
            self.bou = True
            trg_bou_idxs = [torch.from_numpy(trg_bou_idxs_[i]).to(trg_mesh.device) for i in range(len(trg_bou_idxs_))]
            src_bou_idxs = [torch.from_numpy(src_bou_idxs_[i]).to(trg_mesh.device) for i in range(len(src_bou_idxs_))]

            self.trg_asc_bou_points = self.trg_points[0,trg_bou_idxs[0]].unsqueeze(0)
            self.trg_disc_bou_points = self.trg_points[0,trg_bou_idxs[1]].unsqueeze(0)
            self.trg_sa1_bou_points = self.trg_points[0,trg_bou_idxs[2]].unsqueeze(0)
            self.trg_sa2_bou_points = self.trg_points[0,trg_bou_idxs[3]].unsqueeze(0)
            self.trg_sa3_bou_points = self.trg_points[0,trg_bou_idxs[4]].unsqueeze(0)

            self.src_bou_idxs = src_bou_idxs

    def compute_loss(self, deformed_mesh):

        wsa = self.wsa
        wio = self.wio
        src_points = deformed_mesh.points_packed().unsqueeze(0)
        loss_chamfer, _ = chamfer_distance_nonSym(self.trg_points, src_points)

        if not self.bou:
            return loss_chamfer

        src_asc_bou_points = src_points[0,self.src_bou_idxs[0]].unsqueeze(0)
        src_disc_bou_points = src_points[0,self.src_bou_idxs[1]].unsqueeze(0)
        src_sa1_bou_points = src_points[0,self.src_bou_idxs[2]].unsqueeze(0)
        src_sa2_bou_points = src_points[0,self.src_bou_idxs[3]].unsqueeze(0)
        src_sa3_bou_points = src_points[0,self.src_bou_idxs[4]].unsqueeze(0)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer_asc, _ = chamfer_distance_nonSym(self.trg_asc_bou_points, src_asc_bou_points)
        loss_chamfer_disc, _ = chamfer_distance_nonSym(self.trg_disc_bou_points, src_disc_bou_points)
        loss_chamfer_sa1, _ = chamfer_distance_nonSym(self.trg_sa1_bou_points, src_sa1_bou_points)
        loss_chamfer_sa2, _ = chamfer_distance_nonSym(self.trg_sa2_bou_points, src_sa2_bou_points)
        loss_chamfer_sa3, _ = chamfer_distance_nonSym(self.trg_sa3_bou_points, src_sa3_bou_points)

        # mesh laplacian smoothing
        #loss_laplacian = mesh_ops.discrete_dirichlet_energy(src_mesh)

        # Weighted sum of the losses
        loss = loss_chamfer + wio * loss_chamfer_asc + wio * loss_chamfer_disc + \
                wsa * loss_chamfer_sa1 + wsa * loss_chamfer_sa2 + wsa * loss_chamfer_sa3

        return loss
