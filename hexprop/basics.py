import torch
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib as mpl

def hex_grid(n, m):
    a = torch.stack([torch.zeros(n, m), torch.ones(n, m)], dim=0)
    r0, c0 = torch.meshgrid(torch.arange(n), torch.arange(m), indexing='ij')
    r = torch.stack([r0, r0], dim=0)
    c = torch.stack([c0, c0], dim=0)
    return a, r, c

def hex2car(a, r, c):
    x = 1 / 2 * a + c
    y = 3 ** 0.5 / 2 * a + 3 ** 0.5 * r
    return x, y

def hex_xy_grid(n, m, s):
    a, r, c =hex_grid(n, m)
    x0, y0 = hex2car(a, r, c)

    x = torch.where((y0 < 3 ** 0.5 * (n - 0.1) / 2) * ((y0 - 3 ** 0.5 * (n - 0.1) / 4) < 3 ** 0.5 * (m / 2 - x0)), x0,
                x0 - m)
    y = torch.where((y0 < 3 ** 0.5 * (n - 0.1) / 2) * ((y0 - 3 ** 0.5 * (n - 0.1) / 4) < 3 ** 0.5 * (m / 2 - x0)), y0,
                y0 - 3 ** 0.5 / 2 * n)
    x = torch.where((y0 > 3 ** 0.5 * (n - 0.1) / 2) * ((y0 - 3 * 3 ** 0.5 * (n + 0.1) / 4) > 3 ** 0.5 * (x0 - m / 2)), x0,
                x)
    y = torch.where((y0 > 3 ** 0.5 * (n - 0.1) / 2) * ((y0 - 3 * 3 ** 0.5 * (n + 0.1) / 4) > 3 ** 0.5 * (x0 - m / 2)),
                y0 - 3 ** 0.5 * n, y)

    return x * s, y * s

def hex_xy_grid_noshift(n, m, s):
    a, r, c =hex_grid(n, m)
    x, y = hex2car(a, r, c)
    return x * s, y * s

def hex_fxfy_grid(n, m, s):
    x, y = hex_xy_grid(n, m, 1. / s)
    return x / m, y / 1.5 / n


def img2hex(n, m, s, lx, file, kind='linear', need_mask=False):
    I = plt.imread(file)
    I = torch.Tensor(I).flip(0)
    I = I / I.max()
    if len(I.shape) == 3:
        if I.shape[2] == 4:
            I = I[:, :, :3]
        I = torch.sum(I ** 2, dim=2)
        I = I ** 0.5
    I = I / I.max()
    u = I ** 0.5
    ny = u.shape[0]
    nx = u.shape[1]

    dx = s / lx * nx

    xc1 = torch.arange(-nx / 2, nx / 2)
    yc1 = torch.arange(-ny / 2, ny / 2)
    xc2 = xc1
    yc2 = yc1 + 3 ** 0.5 * n * dx
    xc3 = xc1 + m * dx
    yc3 = yc1 + 3 ** 0.5 * n * dx / 2
    func1 = interpolate.interp2d(xc1, yc1, u, kind=kind, fill_value=0)
    func2 = interpolate.interp2d(xc2, yc2, u, kind=kind, fill_value=0)
    func3 = interpolate.interp2d(xc3, yc3, u, kind=kind, fill_value=0)

    xh0 = torch.arange(m)
    xh1 = xh0 + 0.5
    yh0 = 3 ** 0.5 * torch.arange(n)
    yh1 = yh0 + 3 ** 0.5 / 2
    xh0 = xh0 * dx
    yh0 = yh0 * dx
    xh1 = xh1 * dx
    yh1 = yh1 * dx

    x, y = hex_xy_grid(n, m, dx)
    mask1 = ((x > 0 - 0.1 * dx) * (y > 0 - 0.1 * dx)).float()
    mask2 = ((x > 0 - 0.1 * dx) * (y < 0)).float()
    mask3 = (x < 0).float()

    u_hex0 = torch.Tensor(func1(xh0, yh0)) * mask1[0] + torch.Tensor(func2(xh0, yh0)) * mask2[0] + torch.Tensor(func3(xh0, yh0)) * \
             mask3[0]
    u_hex1 = torch.Tensor(func1(xh1, yh1)) * mask1[1] + torch.Tensor(func2(xh1, yh1)) * mask2[1] + torch.Tensor(func3(xh1, yh1)) * \
             mask3[1]
    u_hex = torch.stack([u_hex0, u_hex1], dim=0)

    if need_mask:
        mask = (x.abs() <= (nx / 2)) * (y.abs() <= (ny / 2))
        return u_hex, mask.float()
    else:
        return u_hex


def img2hex_noshift(n, m, dx, file, kind='linear', need_mask=False):
    I = plt.imread(file)
    I = torch.Tensor(I).flip(0)
    I = I / I.max()
    if len(I.shape) == 3:
        if I.shape[2] == 4:
            I = I[:, :, :3]
        I = torch.sum(I ** 2, dim=2)
        I = I ** 0.5
    I = I / I.max()
    u = I ** 0.5
    ny = u.shape[0]
    nx = u.shape[1]

    xc = torch.arange(-nx/2, nx/2)
    yc = torch.arange(-ny/2, ny/2)
    # xc2 = xc1
    # yc2 = yc1 + 3 ** 0.5 * n * dx
    # xc3 = xc1 + m * dx
    # yc3 = yc1 + 3 ** 0.5 * n * dx / 2
    func = interpolate.interp2d(xc, yc, u, kind=kind, fill_value=0)
    # func2 = interpolate.interp2d(xc2, yc2, u, kind=kind, fill_value=0)
    # func3 = interpolate.interp2d(xc3, yc3, u, kind=kind, fill_value=0)

    xh0 = torch.arange(m) - m / 2
    xh1 = xh0 + 0.5
    yh0 = 3 ** 0.5 * (torch.arange(n) - n / 2)
    yh1 = yh0 + 3 ** 0.5 / 2
    xh0 = xh0 * dx
    yh0 = yh0 * dx
    xh1 = xh1 * dx
    yh1 = yh1 * dx
    a, r, c = hex_grid(n, m)
    x, y = hex2car(a, r, c)
    x, y = dx * (x - m / 2), dx * (y - n * 3 ** 0.5 / 2)
    # x, y = hex_xy_grid(n, m, dx)
    # mask1 = ((x > 0 - 0.1 * dx) * (y > 0 - 0.1 * dx)).float()
    # mask2 = ((x > 0 - 0.1 * dx) * (y < 0)).float()
    # mask3 = (x < 0).float()

    u_hex0 = torch.Tensor(func(xh0, yh0))
    u_hex1 = torch.Tensor(func(xh1, yh1))
    # u_hex1 = torch.Tensor(func1(xh1, yh1)) * mask1[1] + torch.Tensor(func2(xh1, yh1)) * mask2[1] + torch.Tensor(func3(xh1, yh1)) * mask3[1]
    u_hex = torch.stack([u_hex0, u_hex1], dim=0)

    if need_mask:
        mask = (x.abs() <= (nx / 2)) * (y.abs() <= (ny / 2))
        return u_hex, mask.float()
    else:
        return u_hex


class Hex(object):
    def __init__(self, tensor):
        super(Hex, self).__init__()
        self.array = tensor
        self.shape = tensor.shape
        self.side = 1

    def __str__(self):
        super(Hex, self).__init__()
        stack_array = self.stack()
        stack_str = str(stack_array)
        space = int((len(stack_str) / stack_array.shape[0] - 10) / stack_array.shape[1] // 2)
        stack_str = stack_str[7:-1]
        out_str = ''
        last_pos = 0
        flag = 1
        for i in range(len(stack_str)):
            if stack_str[i] == '\n':
                out_str += stack_str[last_pos:i+1]
                last_pos = i + 8
                if flag:
                    out_str += ' ' * space
                flag = 1 - flag
        out_str += stack_str[last_pos:]
        return 'hexagonal array (\n' + out_str + ')'

    def shape(self):
        return self.array.shape

    def stack(self):
        stack_array = torch.stack([self.array[0], self.array[1]], dim=1)
        stack_array = stack_array.view(2 * self.shape[1], self.shape[2])
        return stack_array

    def set_side(self, a):
        self.side = a
        return self

    def plot(self, type='real', gamma=1):
        array1 = self.array[0]
        array1 = array1.view(array1.shape[0], array1.shape[1], 1)
        array1 = array1.repeat(1, 1, 2)
        array1 = array1.view(array1.shape[0], array1.shape[1] * 2)
        array1 = torch.cat((array1, torch.zeros(array1.shape[0], 1)), 1)

        array2 = self.array[1]
        array2 = array2.view(array2.shape[0], array2.shape[1], 1)
        array2 = array2.repeat(1, 1, 2)
        array2 = array2.view(array2.shape[0], array2.shape[1] * 2)
        array2 = torch.cat((torch.zeros(array2.shape[0], 1), array2), 1)

        show_array = torch.stack([array1, array2], dim=1)
        show_array = show_array.view(array1.shape[0] * 2, array1.shape[1])

        lx = self.side * (self.shape[2] + 0.5)
        ly = self.side * self.shape[1] * 3 ** 0.5

        if type == 'real':
            show_array = show_array.real
        if type == 'imag':
            show_array = show_array.imag
        plt.figure()
        plt.imshow(show_array, extent=(-lx / 2, lx / 2, -ly / 2, ly / 2), aspect=1, norm=mpl.colors.PowerNorm(gamma=gamma))
        plt.colorbar()

    def plot(self, datatype='real', gamma=1, cmap='rainbow', figsize=None, vmax=None, vmin=None, colorbar=True):
        if datatype == 'amplitude':
            plot_data = self.array.abs()
        elif datatype == 'phase':
            plot_data = self.array.angle()
        elif datatype == 'real':
            plot_data = self.array.real
        elif datatype == 'imag':
            plot_data = self.array.imag
        elif datatype == 'intensity':
            plot_data = self.array.abs() ** 2
        else:
            raise Exception('Type not supported.')

        x_grid, y_grid = self.grid_xy()
#         x_grid, y_grid = hex_xy_grid(self.array.shape[1], self.array.shape[2], self.side)
        fig, ax = plt.subplots(figsize=figsize, dpi=72)
        r = self.side / 3 ** 0.5
        plt.scatter(x_grid, y_grid, c=torch.zeros_like(plot_data), marker='h',
                    cmap=cmap, norm=mpl.colors.PowerNorm(gamma=gamma), linewidths=0)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        if colorbar:
            cb = plt.colorbar()
        plt.axis('equal')
        r_x = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        r_y = ax.transData.transform([0, r])[1] - ax.transData.transform([0, 0])[1]
        # print(r_x, r_y)
        r_ = min(r_x, r_y)
        if colorbar:
            cb.remove()
        plt.clf()
        if gamma == 1:
            plt.scatter(x_grid, y_grid, s=4 * (r_ + 0.3) ** 2, c=plot_data, marker='h',
                        cmap=cmap, vmax=vmax, vmin=vmin, linewidths=0)
        else:
            plt.scatter(x_grid, y_grid, s=4 * (r_ + 0.3) ** 2, c=plot_data, marker='h',
                        cmap=cmap, norm=mpl.colors.PowerNorm(gamma=gamma), linewidths=0)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        if colorbar:
            plt.colorbar()
        else:
            plt.axis('off')
        plt.axis('equal')
        # plt.grid()

    def plot_noshift(self, datatype='real', gamma=1, cmap='rainbow', figsize=None, vmax=None, vmin=None, colorbar=True):
        if datatype == 'amplitude':
            plot_data = self.array.abs()
        elif datatype == 'phase':
            plot_data = self.array.angle()
        elif datatype == 'real':
            plot_data = self.array.real
        elif datatype == 'imag':
            plot_data = self.array.imag
        elif datatype == 'intensity':
            plot_data = self.array.abs() ** 2
        else:
            raise Exception('Type not supported.')

        a, r, c = Hex(self.array).grid()
        x_grid, y_grid = hex2car(a, r, c)
        x_grid, y_grid = self.side * x_grid, self.side * y_grid
        fig, ax = plt.subplots(figsize=figsize, dpi=72)
        r = self.side / 3 ** 0.5
        plt.scatter(x_grid, y_grid, c=plot_data, marker='h',
                    cmap=cmap, norm=mpl.colors.PowerNorm(gamma=gamma), linewidths=0)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        if colorbar:
            cb = plt.colorbar()
        plt.axis('equal')
        r_x = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        r_y = ax.transData.transform([0, r])[1] - ax.transData.transform([0, 0])[1]
        # print(r_x, r_y)
        r_ = min(r_x, r_y)
        if colorbar:
            cb.remove()
        plt.clf()
        if gamma == 1:
            plt.scatter(x_grid, y_grid, s=4 * (r_ + 0.3) ** 2, c=plot_data, marker='h',
                        cmap=cmap, vmax=vmax, vmin=vmin, linewidths=0)
        elif gamma == 'log':
            plt.scatter(x_grid, y_grid, s=4 * (r_ + 0.3) ** 2, c=plot_data, marker='h',
                        cmap=cmap, norm=mpl.colors.LogNorm(), linewidths=0)
        else:
            plt.scatter(x_grid, y_grid, s=4 * (r_ + 0.3) ** 2, c=plot_data, marker='h',
                        cmap=cmap, norm=mpl.colors.PowerNorm(gamma=gamma), linewidths=0)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        if colorbar:
            plt.colorbar()
        else:
            plt.axis('off')
        plt.axis('equal')

    def plot_band(self, datatype='phase', gamma=1, cmap='rainbow', figsize=None, colorbar=True):
        if datatype == 'amplitude':
            plot_data = self.array.abs()
        elif datatype == 'phase':
            plot_data = self.array.angle()
        elif datatype == 'real':
            plot_data = self.array.real
        elif datatype == 'imag':
            plot_data = self.array.imag
        else:
            raise Exception('Type not supported.')

        fx_grid, fy_grid = self.grid_fxfy()
        # fig, ax = plt.subplots(figsize=figsize, dpi=72)
        plt.figure(figsize=figsize, dpi=72)
        r = min(1 / self.side / self.shape[2] / 3 ** 0.5, 2 / 3 / self.side / self.shape[1] / 3 ** 0.5)
        plt.scatter(fx_grid, fy_grid, c=torch.zeros_like(plot_data), marker='h',
                    cmap=cmap, norm=mpl.colors.PowerNorm(gamma=gamma), linewidth=0)
        # plt.scatter(fx_grid, fy_grid, c=torch.zeros_like(plot_data), marker='h',
        #             cmap=cmap, norm=mpl.colors.LogNorm(vmin=plot_data.min(), vmax=plot_data.max()), linewidth=0)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.xlabel('fx')
        plt.ylabel('fy')
        if colorbar:
            cb = plt.colorbar()
        plt.axis('equal')
        # r_ = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        ax = plt.gca()
        r_x = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        r_y = ax.transData.transform([0, r])[1] - ax.transData.transform([0, 0])[1]
        r_ = max(r_x, r_y)
        if colorbar:
            cb.remove()
        plt.clf()
        plt.scatter(fx_grid, fy_grid, s=4 * (r_ + 0.1) ** 2, c=plot_data, marker='h',
                    cmap=cmap, norm=mpl.colors.LogNorm(vmin=plot_data.min(), vmax=plot_data.max()), linewidth=0)
        # plt.scatter(fx_grid, fy_grid, s=4 * (r_ + 0.1) ** 2, c=plot_data, marker='h',
        #             cmap=cmap, norm=mpl.colors.LogNorm(vmin=plot_data.min(), vmax=plot_data.max()), linewidth=0)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.xlabel('fx')
        plt.ylabel('fy')
        if colorbar:
            plt.colorbar()
        plt.axis('equal')
        # plt.show()
        # plt.grid()

    def grid(self):
        a, r, c = hex_grid(self.shape[1], self.shape[2])
        return self.side * a, self.side * r, self.side * c

    def grid_xy(self):
        x, y = hex_xy_grid(self.shape[1], self.shape[2], self.side)
        return x, y

    def grid_fxfy(self):
        fx, fy = hex_fxfy_grid(self.shape[1], self.shape[2], self.side)
        return fx, fy

