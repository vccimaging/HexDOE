import numpy as np
from torch.fft import *

from .basics import *

def hfft(x):
    n = x.shape[1]
    m = x.shape[2]
    device = x.device
    _, r, c = hex_grid(n, m)
    r = r.to(device)
    c = c.to(device)
    r = r[0]
    c = c[0]

    x00 = fft(torch.cat((x[0], torch.zeros(n, m).to(device)), 1), dim=1)
    x01 = fft(torch.cat((x[1], torch.zeros(n, m).to(device)), 1), dim=1)

    g00 = x00[:, ::2]
    g10 = x01[:, ::2]
    g01 = x00[:, 1::2]
    g11 = x01[:, 1::2]

    g00 = g00[:int(n // 2), :] + g00[int(n // 2):, :]
    g10 = g10[:int(n // 2), :] + g10[int(n // 2):, :]
    g01 = g01[:int(n // 2), :] - g01[int(n // 2):, :]
    g11 = g11[:int(n // 2), :] - g11[int(n // 2):, :]

    g01 = g01 * torch.exp(-1j * np.pi / n * r[::2, :])
    g11 = g11 * torch.exp(-1j * np.pi / n * r[::2, :])

    x10 = fft(g00, dim=0)
    x11 = fft(g10, dim=0)
    c00 = fft(g01, dim=0)
    c01 = fft(g11, dim=0)

    f00 = torch.cat((x10, x10), 0)
    f01 = torch.cat((c00, c00), 0)
    f10 = torch.cat((x11, x11), 0)
    f11 = torch.cat((c01, c01), 0)

    tf0 = torch.exp(-1j * np.pi / n * (2 * r)) * torch.exp(-1j * np.pi / m * c)
    tf1 = torch.exp(-1j * np.pi / n * (2 * r + 1)) * torch.exp(-1j * np.pi / m * (2 * c + 1) / 2)

    y0 = f00 + tf0 * f10
    y1 = f01 + tf1 * f11

    y = torch.stack([y0, y1], dim=0)

    return y


def ihfft(x):
    n = x.shape[1]
    m = x.shape[2]

    device = x.device
    _, r, c = hex_grid(n, m)
    r = r.to(device)
    c = c.to(device)
    r = r[0]
    c = c[0]

    x00 = ifft(torch.cat((x[0], torch.zeros(n, m).to(device)), 1), dim=1)
    x01 = ifft(torch.cat((x[1], torch.zeros(n, m).to(device)), 1), dim=1)

    g00 = x00[:, ::2]
    g10 = x01[:, ::2]
    g01 = x00[:, 1::2]
    g11 = x01[:, 1::2]

    g00 = g00[:int(n // 2), :] + g00[int(n // 2):, :]
    g10 = g10[:int(n // 2), :] + g10[int(n // 2):, :]
    g01 = g01[:int(n // 2), :] - g01[int(n // 2):, :]
    g11 = g11[:int(n // 2), :] - g11[int(n // 2):, :]

    g01 = g01 * torch.exp(1j * np.pi / n * r[::2, :])
    g11 = g11 * torch.exp(1j * np.pi / n * r[::2, :])

    x10 = ifft(g00, dim=0)
    x11 = ifft(g10, dim=0)
    c00 = ifft(g01, dim=0)
    c01 = ifft(g11, dim=0)

    f00 = torch.cat((x10, x10), 0)
    f01 = torch.cat((c00, c00), 0)
    f10 = torch.cat((x11, x11), 0)
    f11 = torch.cat((c01, c01), 0)

    tf0 = torch.exp(1j * np.pi / n * (2 * r)) * torch.exp(1j * np.pi / m * c)
    tf1 = torch.exp(1j * np.pi / n * (2 * r + 1)) * torch.exp(1j * np.pi / m * (2 * c + 1) / 2)

    y0 = f00 + tf0 * f10
    y1 = f01 + tf1 * f11

    y = torch.stack([y0, y1], dim=0) / 2

    return y