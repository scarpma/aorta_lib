import numpy as np
import torch

import scipy.sparse as ssparse

# cpu
import scipy.sparse.linalg as sparse_linalg
#####

import torch
from torch.optim import Optimizer
import math

from aorta_lib.ssm import mesh_ops

class precondition_grad():
    """
    given a gradient "b" and a sparse matrix "A"
    computes the preconditioned gradient "y = A^-1 * b"
    by solving the sparse linear system "A * y = b".

    A = (1 + lambda * L)^p
    where L is the laplacian matrix.

    Since the laplacian matrix is not supposed to change
    during optimization, A is computed only one time
    during initialization.

    """

    def __init__(self, init_src_cloud, pv_src_mesh, lamb=1000, p='1'):
        self.lamb = lamb
        self.p = p

        self.prec_device = init_src_cloud.device

        verts = pv_src_mesh.points
        #edges = pv_src_mesh.extract_all_edges().lines.reshape(-1,3)[:,1:]
        if pv_src_mesh.faces[0]==3: # mesh triangolare
            faces = pv_src_mesh.faces.reshape(-1,4)[:,1:]
            edges = np.c_[faces[:,0], faces[:,1]]
            edges = np.r_[edges, np.c_[faces[:,1], faces[:,0]]]
            edges = np.r_[edges, np.c_[faces[:,0], faces[:,2]]]
            edges = np.r_[edges, np.c_[faces[:,2], faces[:,0]]]
            edges = np.r_[edges, np.c_[faces[:,1], faces[:,2]]]
            edges = np.r_[edges, np.c_[faces[:,2], faces[:,1]]]
            edges = np.unique(edges, axis=0)
        elif pv_src_mesh.faces[0]==4: # mesh quadrilaterale
            # counter clock-wise order
            faces = pv_src_mesh.faces.reshape(-1,5)[:,1:]
            edges = np.c_[faces[:,0], faces[:,1]]
            edges = np.r_[edges, np.c_[faces[:,1], faces[:,0]]]
            edges = np.r_[edges, np.c_[faces[:,1], faces[:,2]]]
            edges = np.r_[edges, np.c_[faces[:,2], faces[:,1]]]
            edges = np.r_[edges, np.c_[faces[:,2], faces[:,3]]]
            edges = np.r_[edges, np.c_[faces[:,3], faces[:,2]]]
            edges = np.r_[edges, np.c_[faces[:,3], faces[:,0]]]
            edges = np.r_[edges, np.c_[faces[:,0], faces[:,3]]]
            edges = np.unique(edges, axis=0)


        L = mesh_ops.laplacian_numpy(verts, edges)
        V = verts.shape[0]

        diag_idx = np.arange(V)
        diag_idx = np.c_[diag_idx, diag_idx].T
        values = np.ones(V, dtype=np.float32)
        identity = ssparse.coo_matrix((values, (diag_idx[0], diag_idx[1])), shape=(V,V))

        temp = (identity + lamb * L).tocoo()#.coalesce()
        indices = np.c_[temp.row, temp.col].T
        values = temp.data

        stemp = ssparse.coo.coo_matrix((values, (indices[0], indices[1])),
            shape=(V, V))
        stemp = stemp.tocsc(
        )  # csc sparse matrix format is more efficient for inverse computation


        if p == '1':
            stemp = stemp
        elif p == '2':
            stemp = stemp @ stemp
        else:
            raise NotImplementedError

        self.stemp = stemp

    # cpu version
    def compute(self, grad):
        grad_ = grad.detach().cpu().numpy()
        prec_grad = sparse_linalg.spsolve(self.stemp, grad_)
        return torch.from_numpy(prec_grad).to(self.prec_device)



def my_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, us,
            *, amsgrad, beta1, beta2, lr, weight_decay, eps, preconditioner):
    """
    Modified Adam algorithm for optimization. Uses preconditioned gradient instead of
    gradient. Moreover, uses infinity norm on second order moment, instead of component
    wise division.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        grad = preconditioner.compute(grad)  ###############
        exp_avg = exp_avgs[i]  # m_{i-1}
        exp_avg_sq = exp_avg_sqs[i]  # v_{i-1}
        step = state_steps[i]
        u = us[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_i
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(),
                                        value=1 - beta2)  # v_i

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i],
                          exp_avg_sq,
                          out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() /
                     math.sqrt(bias_correction2)).add_(eps)
        else:
            #denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            denom = ((exp_avg_sq.abs().max()).sqrt() /
                     math.sqrt(bias_correction2)).add_(
                         eps)       ################ infinity norm

        step_size = lr / bias_correction1

        ###### normal version ###################################
        ## param.addcdiv_(exp_avg, denom, value=-step_size)
        #########################################################


        ###### parametrized version #############################
        u.addcdiv_(exp_avg, denom, value=-step_size)
        # now compute x(u) = (1-lam L)^-p u
        param[:] = preconditioner.compute(u)[:]
        #########################################################



class myAdam(Optimizer):
    """
    Implements modified Adam algorithm.
    with the addition to manage the u array
    (read article for reference)

    """
    def __init__(self,
                 params,
                 lamb,
                 p,
                 init_src_cloud,
                 pv_src_mesh,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(myAdam, self).__init__(params, defaults)

        self.lamb = lamb
        self.p = p
        self.init_src_cloud = init_src_cloud
        self.pv_src_mesh = pv_src_mesh
        self.preconditioner = precondition_grad(
            init_src_cloud=self.init_src_cloud, pv_src_mesh=self.pv_src_mesh, lamb=self.lamb, p=self.p)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            us = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, please consider SparseAdam instead'
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        state['u'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    us.append(state['u'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            my_adam(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=group['amsgrad'],
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    preconditioner=self.preconditioner,
                    us=us,
            )
        return loss
