import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function as F


class StiefelParameter(nn.Parameter):
    pass


def init_bimap_parameter(W):
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            v = th.empty(ni, ni, dtype=W.dtype, device=W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :no]
            W.data[i, j] = vv


def init_bimap_parameter_identity(W):
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            W.data[i, j] = th.eye(ni, no)


class SPDParameter(nn.Parameter):
    pass


def bimap(X, W):
    return W.t().matmul(X).matmul(W)


def bimap_channels(X, W):
    batch_size, channels_in, n_in, _ = X.shape
    channels_out, _, _, n_out = W.shape
    P = th.zeros(batch_size, channels_out, n_out, n_out, dtype=X.dtype, device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([bimap(X[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)])
    return P


def modeig_forward(P, op, eig_mode='svd', param=None):
    batch_size, channels, n, n = P.shape
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if eig_mode == 'eig':
                s, U[i, j] = th.eig(P[i, j], True); S[i, j] = s[:, 0]
            elif eig_mode == 'svd':
                U[i, j], S[i, j], _ = th.svd(P[i, j])
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0; L[L == np.inf] = 0; L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class LogEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Log_op)


class ReEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Re_op)


class ExpEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class ReCov(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Re_op)


class SqmEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


class PowerEig(F):
    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Power_op), None


class InvEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Inv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Inv_op)


def geodesic(A, B, t):
    M = CongrG(PowerEig.apply(CongrG(B, A, 'neg'), t), A, 'pos')[0, 0]
    return M


def cov_pool(f, reg_mode='mle'):
    bs, n, T = f.shape
    X = f.matmul(f.transpose(-1, -2)) / (T - 1)
    if reg_mode == 'mle':
        ret = X
    elif reg_mode == 'add_id':
        ret = add_id(X, 1e-6)
    elif reg_mode == 'adjust_eig':
        ret = adjust_eig(X, 0.75)
    if len(ret.shape) == 3:
        return ret[:, None, :, :]
    return ret


def cov_pool_mu(f, reg_mode):
    alpha = 1
    bs, n, T = f.shape
    mu = f.mean(-1, True); f = f - mu
    X = f.matmul(f.transpose(-1, -2)) / (T - 1)
    if reg_mode == 'mle':
        ret = X
    elif reg_mode == 'add_id':
        ret = add_id(X, 1e-6)
    elif reg_mode == 'adjust_eig':
        ret = adjust_eig(0.75)(X)
    if len(ret.shape) == 3:
        return ret[:, None, :, :]
    return ret


def add_id(P, alpha):
    for i in range(P.shape[0]):
        P[i] = P[i] + alpha * P[i].trace() * th.eye(P[i].shape[-1], dtype=P.dtype, device=P.device)
    return P


def dist_riemann(x, y):
    return LogEig.apply(CongrG(x, y, 'neg')).view(x.shape[0], x.shape[1], -1).norm(p=2, dim=-1)


def CongrG(P, G, mode):
    if mode == 'pos':
        GG = SqmEig.apply(G[None, None, :, :])
    elif mode == 'neg':
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P).matmul(GG)
    return PP


def LogG(x, X):
    return CongrG(LogEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def ExpG(x, X):
    return CongrG(ExpEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def BatchDiag(P):
    batch_size, channels, n = P.shape
    Q = th.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q


def karcher_step(x, G, alpha):
    x_log = LogG(x, G)
    G_tan = x_log.mean(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def BaryGeom(x):
    k = 1
    alpha = 1
    with th.no_grad():
        G = th.mean(x, dim=0)[0, :, :]
        for _ in range(k):
            G = karcher_step(x, G, alpha)
        return G


def karcher_step_weighted(x, G, alpha, weights):
    x_log = LogG(x, G)
    G_tan = x_log.mul(weights[:, None, None, None]).sum(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def bary_geom_weighted(x, weights):
    k = 1
    alpha = 1
    G = x.mul(weights[:, None, None, None]).sum(dim=0)[0, :, :]
    for _ in range(k):
        G = karcher_step_weighted(x, G, alpha, weights)
    return G[None, None, :, :]


class Log_op():
    @staticmethod
    def fn(S, param=None):
        return th.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class ReCov_op():
    _threshold = -1e-3

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class Sqm_op():
    @staticmethod
    def fn(S, param=None):
        return th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / th.sqrt(S)


class Sqminv_op():
    @staticmethod
    def fn(S, param=None):
        return 1 / th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / th.sqrt(S)**3


class Power_op():
    _power = 0.5

    @classmethod
    def fn(cls, S, param=None):
        return S**cls._power

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (cls._power) * S**(cls._power - 1)


class Inv_op():
    @classmethod
    def fn(cls, S, param=None):
        return 1 / S

    @classmethod
    def fn_deriv(cls, S, param=None):
        return log(S)


class Exp_op():
    @staticmethod
    def fn(S, param=None):
        return th.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return th.exp(S)
