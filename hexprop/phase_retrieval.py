import numpy as np
import torch.nn as nn

from .waveprop import *

def phase_retrieval_bp(a0, a1_target, s, wl, z,
                    lr=0.01, optim='Adam', momentum=0.9,
                    normalization=False,
                    method='FresnelTF',
                    p_initial=None,
                    loss_max=1e-8, dloss=0, max_iter=100,
                    need_plot=False,
                    mask=torch.Tensor([1]),
                    p_mask=torch.Tensor([1]),
                    use_intensity=True,
                    regularization=None):
    n = a0.shape[1]
    m = a0.shape[2]

    device = a1_target.device
    if type(p_mask) is torch.Tensor:
        p_mask = p_mask.to(device)
    if type(a0) is torch.Tensor:
        a0 = a0.to(device)
    if type(p_initial) is torch.Tensor:
        p_initial = p_initial.to(device)
    if type(mask) is torch.Tensor:
        mask = mask.to(device)

    prop = Prop(n, m, s, wl, z, need_update=True, method=method, phase=p_initial, normalization=normalization, p_mask=p_mask).to(device)
    if optim == 'SGD':
        if normalization:
            optimizer = torch.optim.SGD([{'params': prop.norm, 'lr': 0.01},
                                     {'params': prop.phase}],
                                    lr=lr, momentum=momentum)
        else:
            optimizer = torch.optim.SGD(prop.parameters(), lr=lr, momentum=momentum)
    else:
        if normalization:
            optimizer = torch.optim.Adam([{'params': prop.norm, 'lr': 0.01},
                                     {'params': prop.phase}], lr=lr)
        else:
            optimizer = torch.optim.Adam(prop.parameters(), lr=lr)
    loss_fun = nn.MSELoss()
    loss_list = []
    for i in range(max_iter):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            a1 = prop(a0).abs()
            if need_plot:
                Hex(a1.detach()).plot(datatype='real', cmap='gray')
            if use_intensity:
                # loss = loss_fun(mask * a1_target ** 2, mask * a1 ** 2) * mask.size()[0] * mask.size()[1] * mask.size()[2] / mask.sum()
                loss = loss_fun(mask * a1_target ** 2, mask * a1 ** 2)
            else:
                loss = loss_fun(mask * a1_target, mask * a1)
                # loss = loss_fun(mask * a1_target, mask * a1) * mask.size()[0] * mask.size()[1] * mask.size()[
                #     2] / mask.sum()
            if regularization is not None:
                loss += regularization * (((1 - mask) * a1) ** 2).mean()
            loss_list.append(loss.item())
            print(i, ',   ', loss.item())
            if loss <= loss_max:
                break
            if i >= 2 and abs(loss - loss_list[-2]) <= dloss:
                break
            loss.backward()
            optimizer.step()
    p0 = prop.phase.data % 1
    p0 = p0 * a0
    if normalization:
        norm = prop.norm.item()
        return 2 * np.pi * p0, norm, loss_list
    else:
        return 2 * np.pi * p0, loss_list


def phase_retrieval_gs(a0, a1_target, s, wl, z,  method='FresnelTF', p_initial=None,
            loss_max=1e-8, dloss=-1, max_iter=100, need_plot=False, use_intensity=True):
    n = a0.shape[1]
    m = a0.shape[2]

    if p_initial is None:
        p0 = torch.rand_like(a0) * np.pi * 2
    else:
        p0 = p_initial

    device = a1_target.device
    if type(a0) is torch.Tensor:
        a0 = a0.to(device)
    if type(p0) is torch.Tensor:
        p0 = p0.to(device)
    forward = Prop(n, m, s, wl, z, method=method).to(device)
    backward = Prop(n, m, s, wl, -z, method=method).to(device)
    loss_fun = nn.MSELoss()
    loss_list = []
    for i in range(max_iter):
        u1 = forward(a0 * torch.exp(1j * p0))
        a1 = u1.abs()
        p1 = u1.angle()
        if need_plot:
            Hex(a1.cpu()).plot(datatype='real', cmap='gray')
        if use_intensity:
            loss = loss_fun(a1_target ** 2, a1 ** 2)
        else:
            loss = loss_fun(a1_target, a1)
        loss_list.append(loss.item())
        print(i, ',   ', loss.item())
        if loss <= loss_max:
            break
        if i >= 2 and abs(loss - loss_list[-2]) <= dloss:
            break
        p0 = backward(a1_target * torch.exp(1j * p1)).angle() #/ 2 / np.pi

    return p0 * a0 % (2 * np.pi), loss_list