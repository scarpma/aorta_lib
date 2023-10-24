import numpy as np
import torch

import scipy.sparse as ssparse

import torch
from torch.optim import Optimizer
import math

import mesh_ops

import time



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

        verts = torch.from_numpy(pv_src_mesh.points).to(self.prec_device)
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
        edges = torch.from_numpy(edges).to(self.prec_device)

        L = mesh_ops.laplacian(verts, edges)
        V = verts.shape[0]




        # indici della diagonale
        idx = torch.arange(V, device=self.prec_device)
        idx = torch.stack([idx, idx], dim=0)
        values = torch.ones(V, device=self.prec_device)
        identity = torch.sparse.FloatTensor(idx, values, (V, V))

        temp = (identity + lamb * L).coalesce()
        # csc sparse matrix format is more efficient for inverse computation
        stemp = temp#.tocsc()


        if p == '1':
            stemp = stemp
        elif p == '2':
            stemp = stemp @ stemp
        else:
            raise NotImplementedError

        self.stemp = stemp



    def sparse_multiplier(self, X):
        # non batched adaptation
        Y = [self.stemp @ X[0]]
        return torch.cat(Y, dim=0)



    def compute(self, grad):
        grad_ = grad.detach().unsqueeze(0)
        prec_grad, _ = cg_batch(
            self.sparse_multiplier,
            grad_,
            M_bmm=None,
            X0=None,
            rtol=1.e-3,
            atol=0.,
            maxiter=None,
            verbose=False,
        )
        return prec_grad[0]



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




def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if not optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


