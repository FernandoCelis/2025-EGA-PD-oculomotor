import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional


class BiMap(nn.Module):
    def __init__(self, ho, hi, ni, no, dtype, device):
        super(BiMap, self).__init__()
        self._W = functional.StiefelParameter(th.empty(ho, hi, ni, no, dtype=dtype, device=device))
        self._ho = ho; self._hi = hi; self._ni = ni; self._no = no
        functional.init_bimap_parameter(self._W)

    def forward(self, X):
        return functional.bimap_channels(X, self._W)


class LogEig(nn.Module):
    def forward(self, P):
        return functional.LogEig.apply(P)


class ExpEig(nn.Module):
    def forward(self, P):
        return functional.ExpEig.apply(P)


class SqmEig(nn.Module):
    def forward(self, P):
        return functional.SqmEig.apply(P)


class ReEig(nn.Module):
    def forward(self, P):
        return functional.ReEig.apply(P)


class BaryGeom(nn.Module):
    def forward(self, x):
        return functional.BaryGeom(x)


class BatchNormSPD(nn.Module):
    def __init__(self, n, dtype):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = th.eye(n, dtype=dtype)
        self.weight = functional.SPDParameter(th.eye(n, dtype=dtype))

    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N*h, 1).permute(2, 3, 0, 1).contiguous()
        if self.training:
            mean = functional.BaryGeom(X_batched)
            with th.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)
            X_centered = functional.CongrG(X_batched, mean, 'neg')
        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, 'neg')
        X_normalized = functional.CongrG(X_centered, self.weight, 'pos')
        return X_normalized.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CovPool(nn.Module):
    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool(f, self._reg_mode)


class CovPoolBlock(nn.Module):
    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        ff = [functional.cov_pool(f[:, i, :, :], self._reg_mode)[:, None, :, :, :] for i in range(f.shape[1])]
        return th.cat(ff, 1)


class CovPoolMean(nn.Module):
    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool_mu(f, self._reg_mode)


class SqrtSPD(nn.Module):
    def forward(self, P):
        return functional.SqmEig.apply(P)


class NegativeSqrtSPD(nn.Module):
    def forward(self, P):
        return functional.SqminvEig.apply(P)


class SPDSoftmax(nn.Module):
    def __init__(self, norm=False):
        super(SPDSoftmax, self).__init__()
        self.norm = norm

    def forward(self, input):
        S, U = th.linalg.eigh(input)
        if self.norm == True:
            S_sum = S.sum(dim=-1, keepdim=True)
            S = S / S_sum
        S = th.nn.functional.softmax(S, dim=-1)
        S = S.diag_embed()
        return U @ S @ U.transpose(-2, -1)


class EGA(nn.Module):
    def __init__(self, dim, output_dim=None, dtype=None, device=None):
        super(EGA, self).__init__()

        if output_dim is None:
            output_dim = dim // 2

        self.dim = dim
        self.output_dim = output_dim
        self._dtype = dtype
        self._device = device

        self.bire_q = nn.Sequential(BiMap(1, 1, dim, output_dim, dtype=dtype, device=device), ReEig())
        self.bire_k = nn.Sequential(BiMap(1, 1, dim, output_dim, dtype=dtype, device=device), ReEig())
        self.bire_v = nn.Sequential(BiMap(1, 1, dim, output_dim, dtype=dtype, device=device), ReEig())

        self.spd_softmax = SPDSoftmax()
        self.log_eig = LogEig()
        self.sqrt = SqrtSPD()
        self.invsqrt = NegativeSqrtSPD()

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        Q = self.bire_q(x1)
        K = self.bire_k(x2)
        V = self.bire_v(x2)

        QK = self.sqrt(K) @ self.sqrt(self.invsqrt(K) @ Q @ self.invsqrt(K)) @ self.sqrt(K)
        att = self.spd_softmax(QK)
        out = self.log_eig(att) + self.log_eig(V)

        return out
