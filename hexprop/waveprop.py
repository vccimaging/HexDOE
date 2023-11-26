import numpy as np
import torch.nn as nn

from .hffts import *
from .basics import *

class RSIR(nn.Module):
    def __init__(self, n, m, s, wl, z):
        super(RSIR, self).__init__()
        k = 2 * np.pi / wl
        x, y = hex_xy_grid(n, m, s)
        r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        h = z / r ** 2 / (1j * wl) * torch.exp(1j * k * r)
        self.H = hfft(h) * s ** 2 * 3 ** 0.5 / 2

    def to(self, device):
        self.H = self.H.to(device)
        return self

    def forward(self, u1):
        U1 = hfft(u1)
        U2 = self.H * U1
        u2 = ihfft(U2)
        return u2


class RSTF(nn.Module):
    def __init__(self, n, m, s, wl, z):
        super(RSTF, self).__init__()
        k = 2 * np.pi / wl
        fx, fy = hex_fxfy_grid(n, m, s)
        self.H = torch.exp(1j * k * z * (1 - wl ** 2 * (fx ** 2 + fy ** 2)) ** 0.5)

    def to(self, device):
        self.H = self.H.to(device)
        return self

    def forward(self, u1):
        U1 = hfft(u1)
        U2 = self.H * U1
        u2 = ihfft(U2)
        return u2


class FresnelIR(nn.Module):
    def __init__(self, n, m, s, wl, z):
        super(FresnelIR, self).__init__()
        k = 2 * np.pi / wl
        x, y = hex_xy_grid(n, m, s)
        h = torch.exp(1j * k / (2 * z) * (x ** 2 + y ** 2))
        self.H = torch.exp(1j * k * z * torch.ones_like(h)) / (1j * wl * z) * hfft(h) * s ** 2 * 3 ** 0.5 / 2

    def to(self, device):
        self.H = self.H.to(device)
        return self

    def forward(self, u1):
        U1 = hfft(u1)
        U2 = self.H * U1
        u2 = ihfft(U2)
        return u2


class FresnelTF(nn.Module):
    def __init__(self, n, m, s, wl, z):
        super(FresnelTF, self).__init__()
        k = 2 * np.pi / wl
        fx, fy = hex_fxfy_grid(n, m, s)
        self.H = torch.exp(- 1j * np.pi * wl * z * (fx ** 2 + fy ** 2))
        self.c = torch.exp(1j * k * z * torch.ones_like(fx))

    def to(self, device):
        self.H = self.H.to(device)
        self.c = self.c.to(device)
        return self

    def forward(self, u1):
        U1 = hfft(u1)
        U2 = self.H * U1
        u2 = ihfft(U2)
        return u2 * self.c


class FF(nn.Module):
    def __init__(self, n, m, s, wl, z):
        super(FF, self).__init__()
        k = 2 * np.pi / wl
        x, y = hex_fxfy_grid(n, m, s)
        x = wl * z * x
        y = wl * z * y
        self.c = 1 / 1j / wl / z * torch.exp(1j * k * z + 1j * k / 2 / z * (x ** 2 + y ** 2))
        self.c = self.c * s ** 2 * 3 ** 0.5 / 2

    def to(self, device):
        self.c = self.c.to(device)
        return self

    def forward(self, u1):
        u2 = self.c * hfft(u1)
        return u2


class Prop(nn.Module):
    def __init__(self, n, m, s, wl, z, method='RSIR', need_update=False, phase=None, normalization=False, p_mask=torch.Tensor([1])):
        super(Prop, self).__init__()
        self.need_update = need_update
        self.normalization = normalization
        self.p_mask = p_mask
        if method == 'RSIR':
            self.prop = RSIR(n, m, s, wl, z)
        elif method == 'RSTF':
            self.prop = RSTF(n, m, s, wl, z)
        elif method == 'FresnelTF':
            self.prop = FresnelTF(n, m, s, wl, z)
        elif method == 'FresnelIR':
            self.prop = FresnelIR(n, m, s, wl, z)
        elif method == 'FF':
            self.prop = FF(n, m, s, wl, z)
        else:
            raise Exception('Method not supported.')

        if need_update:
            if phase is None:
                self.phase = nn.Parameter(p_mask * torch.rand(2, n, m))
            else:
                self.phase = nn.Parameter(p_mask * phase)
            self.register_parameter("Phase", self.phase)
            if normalization:
                self.norm = nn.Parameter(torch.Tensor([1]))
                self.register_parameter("Normalization", self.norm)
            else:
                self.norm = torch.Tensor([1])
        else:
            self.phase = torch.zeros(2, n, m)
            self.norm = torch.Tensor([1])

    def to(self, device):
        if self.need_update:
            self.phase = nn.Parameter(self.phase.data.to(device))
            if self.normalization:
                self.norm = nn.Parameter(self.norm.data.to(device))
                # print(self.norm)
            else:
                self.norm = self.norm.to(device)
        else:
            self.phase = self.phase.to(device)
            self.norm = self.norm.to(device)
        self.p_mask = self.p_mask.to(device)
        self.prop = self.prop.to(device)
        return self

    def forward(self, a):
        u2 = self.prop(a * self.norm * torch.exp(2j * np.pi * self.p_mask * self.phase))
        return u2