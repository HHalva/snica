"""IIA-TCL model"""


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from subfunc.showdata import *


# =============================================================
# =============================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        # m, _ = torch.max(torch.reshape(x, (*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:])), dim=2)
        m, _ = torch.max(torch.reshape(x, (*x.shape[:1], self._pool_size, x.shape[1] // self._pool_size, *x.shape[2:])), dim=1)
        return m

class smooth_leaky_relu(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    def forward(self, x):
        return self._alpha * x + (1 - self._alpha) * torch.logaddexp(x, torch.tensor(0))


# =============================================================
# =============================================================
class Net(nn.Module):
    def __init__(self, h_sizes, num_class, h_sizes_z=None, pool_size=1, alpha=0.1):
        """ Network model for segment-wise stationary model
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_class: number of classes
             h_sizes_z: number of channels for each layer of MLP_z [num_layer+1] (first size is input-dim)
             pool_size: pool size of max-out nonlinearity
         """
        super(Net, self).__init__()
        # h
        layer = [nn.Linear(h_sizes[k-1],h_sizes[k]*pool_size) for k in range(1,len(h_sizes)-1)]
        layer.append(nn.Linear(h_sizes[-2],h_sizes[-1]))
        self.layer = nn.ModuleList(layer)
        # hz
        if h_sizes_z is None:
            h_sizes_z = h_sizes.copy()
            h_sizes_z[0] = np.int(h_sizes_z[0]/2)
        layerz = [nn.Linear(h_sizes_z[k-1],h_sizes_z[k]*pool_size) for k in range(1,len(h_sizes_z)-1)]
        layerz.append(nn.Linear(h_sizes_z[-2],h_sizes_z[-1]))
        self.layerz = nn.ModuleList(layerz)
        # self.activation = Maxout(pool_size)
        self.activation = smooth_leaky_relu(alpha)
        self.mlr = nn.Linear((h_sizes[-1] + h_sizes_z[-1])*2, num_class)

        # initialize
        for k in range(len(self.layer)):
            torch.nn.init.xavier_uniform_(self.layer[k].weight)
        for k in range(len(self.layerz)):
            torch.nn.init.xavier_uniform_(self.layerz[k].weight)
        torch.nn.init.xavier_uniform_(self.mlr.weight)


    def forward(self, x):
        """ forward
         Args:
             x: input [batch, dim]
         """
        batch_size, in_dim = x.size()
        num_comp = in_dim // 2
        xz = x[:, num_comp:]

        # h
        h = x
        for k in range(len(self.layer)):
            h = self.layer[k](h)
            if k != len(self.layer)-1:
                h = self.activation(h)
        h_nonlin = torch.cat((h**2, h), 1)

        # hz
        hz = xz
        for k in range(len(self.layerz)):
            hz = self.layerz[k](hz)
            if k != len(self.layerz)-1:
                hz = self.activation(hz)
        hz_nonlin = torch.cat((hz**2, hz), 1)

        # concatenate
        hhz = torch.cat((h_nonlin, hz_nonlin), 1)
        # MLR
        y = self.mlr(hhz)

        return y, h, hz


# =============================================================
# =============================================================
def sconcat_shifted_data(x, label=None, shift=1):
    """ Spatially concatenate temporally shifted signals to original ones.
    Args:
        x: signals. 2D ndarray [num_comp, num_data]
        label: labels. 1D ndarray [num_data]
        shift: amount of temporal shift
    Returns:
        y: signals concatenated with their temporal shifts, y(t) = [x(t); x(t-1)]. 2D ndarray [2*num_comp, num_data-1]
        label: labels. 1D ndarray [num_data-1]
    """
    if not isinstance(shift, list):
        shift = [shift]

    xdim = x.shape[0]
    y = copy.copy(x)
    for sn in shift:
        if sn >= 1:
            xshift = np.concatenate([np.zeros([xdim,sn]), copy.copy(x[:,0:-sn])], axis=1)
        elif sn <=-1:
            xshift = np.concatenate([copy.copy(x[:,-sn:]), np.zeros([xdim,-sn])], axis=1)
        else:
            raise ValueError

        y = np.concatenate([y, xshift], axis=0)

    if np.max(shift) >= 1:
        y = y[:, np.max(shift):]
        if label is not None:
            label = label[0:-np.max(shift)]
    if np.min(shift) <= -1:
        y = y[:, :np.min(shift)]
        if label is not None:
            label = label[:np.min(shift)]

    return y, label

