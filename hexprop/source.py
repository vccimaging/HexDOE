import numpy as np
import scipy.special

from .basics import *

def rectangular(n, m, s, wx, wy=None, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    if wy is None:
        wy = wx
    u = (torch.abs((x - x0) / wx) <= 1) * (torch.abs((y - y0) / wy) <= 1)
    u = u.float()
    return u


def rect_result_fn(n, m, s, wx, wl, z, wy=None, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    if wy is None:
        wy = wx
    k = 2 * np.pi / wl
    a1 = -(2 / wl / z) ** 0.5 * (wx + (x - x0))
    a2 = (2 / wl / z) ** 0.5 * (wx - (x - x0))
    b1 = -(2 / wl / z) ** 0.5 * (wy + (y - y0))
    b2 = (2 / wl / z) ** 0.5 * (wy - (y - y0))
    sa1, ca1 = scipy.special.fresnel(a1)
    sa2, ca2 = scipy.special.fresnel(a2)
    sb1, cb1 = scipy.special.fresnel(b1)
    sb2, cb2 = scipy.special.fresnel(b2)
    # Ix = 1 / 2 ** 0.5 * (ca2 - ca1 + 1j * (sa2 - sa1))
    # Iy = 1 / 2 ** 0.5 * (cb2 - cb1 + 1j * (sb2 - sb1))
    # u = np.exp(1j * k * z) / 2 / 1j * Ix * Iy
    u = np.exp(1j * k * z) / 2 / 1j * (ca2 - ca1 + 1j * (sa2 - sa1)) * (cb2 - cb1 + 1j * (sb2 - sb1))
    return u


def rect_result_ff(n, m, s, w, wl, z, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    k = 2 * np.pi / wl
    u = 4 * w ** 2 / 1j / wl / z * torch.exp(1j * k * z + 1j * k / 2 / z * ((x - x0) ** 2 + (y - y0) ** 2)) *\
        torch.sinc(2 * w * (x - x0) / wl / z) * torch.sinc(2 * w * (y - y0) / wl / z)
    return u


# circular beam
def circular(n, m, s, rx, ry=None, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    if ry is None:
        ry = rx
    u = ((x - x0) ** 2 / rx ** 2 + (y - y0) ** 2 / ry ** 2 <= 1)
    u = u.float()
    return u


# def circular_result_ff_old(n, m, s, r, wl, z, x0=0, y0=0):
#     x, y = hex_xy_grid(n, m, s)
#     k = 2 * np.pi / wl
#     jinc1 = scipy.special.jv(1, 2*math.pi*r/lbd/z*((X-x0)**2+(Y-y0)**2)**0.5)
#     jinc2 = r/lbd/z*((X-x0)**2+(Y-y0)**2)**0.5
#     jinc3 = jinc1/jinc2
#     # jinc3 = torch.where(jinc2==0, math.pi, jinc1/jinc2)
#     N, M = X.shape
#     jinc3[N//2][M//2] = math.pi
#     u = 1/1j/lbd/z*r**2*torch.exp(1j*k*z+1j*k/2/z*((X-x0)**2+(Y-y0)**2))*jinc3
#     return u


def circular_result_ff(n, m, s, r, wl, z, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    k = 2 * np.pi / wl
    jinc1 = scipy.special.jv(1, 2 * np.pi * r / wl / z * ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5)
    jinc2 = r / wl / z * ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
    flag = ((x == 0) * (y == 0)).float()
    jinc2 += flag
    # jinc3 = jinc1 / jinc2
    jinc3 = (jinc1 / jinc2) * (1 - flag) + np.pi * flag
    # jinc3 = torch.where(jinc2==0, math.pi, jinc1/jinc2)
    # _, N, M = X.shape
    # jinc3[:, N//2, M//2] = math.pi
    u = -1j / wl / z * r ** 2 * torch.exp(1j * k * z + 1j * k / 2 / z * ((x - x0) ** 2 + (y - y0) ** 2)) * jinc3
    return u


# Gaussian beam
def gaussian(n, m, s, s1, s2=None, r=0, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    if s2 is None:
        s2 = s1
    u = torch.exp(-1 / (1 - r ** 2) * ((x - x0) ** 2 / s1 ** 2 - 2 * r * (x - x0) * (y - y0) / s1 / s2 + (y - y0) ** 2 / s2 ** 2))
    return u


def gaussian_result(n, m, s, wl, z, w0, x0=0, y0=0):
    x, y = hex_xy_grid(n, m, s)
    k = 2 * np.pi / wl
    zr = np.pi * w0 ** 2 / wl
    wz = w0 * (1 + (z / zr) ** 2) ** 0.5
    # rz = z + zr ** 2 / z
    rzr = z / (z ** 2 + zr ** 2)
    # qz = z + 1j * zr
    phaiz = np.atan(z / zr)
    # a2 = math.pi * w0 ** 2 + 1j * math.pi * lbd * z
    # a = (math.pi * w0 ** 2 + 1j * math.pi * lbd * z) ** 0.5
    u = w0 / wz * torch.exp(-(x ** 2 + y ** 2) / wz ** 2) * torch.exp(1j * (k * z + k * rzr * (x ** 2 + y ** 2) / 2 - phaiz))
    # u = math.pi * w0 ** 2 / a2 * torch.exp(1j * k * z - math.pi / a2 * (X ** 2 + Y ** 2))
    # u = 1/qz*torch.exp(-1j*k*(X**2+Y**2)/2/qz)
    return u


def spiral(n, m, s, l=4e-3, w=2e-3, k=12, t=2500):
    x, y = hex_xy_grid(n, m, s)
    r = (x ** 2 + y ** 2) ** 0.5
    theta = torch.atan2(y, x)
    out = (r * t - theta) % (np.pi / k)
    out = out > (np.pi / k / 2)
    out = out * (x ** 2 + y ** 2 <= (l / 2) ** 2)
    return out.float()