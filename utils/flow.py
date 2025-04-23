# This code is modified from https://github.com/haofeixu/gmflow and https://github.com/liuziyang123/LDRFlow.


import torch
import torch.nn.functional as F
from einops import rearrange


def backward_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    # mask = torch.ones(x.size()).to(DEVICE)
    # mask = F.grid_sample(mask, vgrid)

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    return output


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=10
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = backward_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = backward_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta
    # threshold = 0

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def calculate_imf_map(x, y):
    imf_map = torch.zeros(256).cuda()
    r = 0
    for i in range(256):
        if x[i] == 0:
            imf_map[i] = -1
        else:
            p, v, j = x[i], 0, r
            while True:
                if y[j] < p:
                    p = p - y[j]
                    v = v + y[j] * j
                    j += 1
                else:
                    r = j
                    y[j] = y[j] - p
                    v = v + p * j
                    imf_map[i] = (v / x[i]).round()
                    break
    imf_map = imf_map.unsqueeze(dim=0)
    return imf_map


def IMF(ue, oe):
    B, C, H, W = ue.shape
    ue = (ue * 255).round()
    oe = (oe * 255).round()

    imf_map = []
    ue_rgb = torch.split(ue, 1, dim=1)
    oe_rgb = torch.split(oe, 1, dim=1)
    imf_map = [
        calculate_imf_map(torch.histc(x, bins=256, min=0, max=255), torch.histc(y, bins=256, min=0, max=255)) for x, y in zip(ue_rgb, oe_rgb)
    ]
    imf_map = torch.concat(imf_map, dim=0)

    zeros = torch.zeros([C, 1], dtype=torch.float32).cuda()
    imf_map = torch.concat((imf_map, zeros), 1)

    ue_imf = rearrange(ue.squeeze(), 'c h w -> c (h w)')
    ue_imf_floor = ue_imf.floor()
    for c in range(C):
        ind = ue_imf_floor[c].long()
        ue_imf[c, :] = (ue_imf[c, :] - ue_imf_floor[c, :]) * (imf_map[c, :][ind + 1] - imf_map[c, :][ind]) + imf_map[c, :][ind]
    ue_imf = rearrange(ue_imf, 'c (h w) -> c h w', h=H, w=W).unsqueeze(dim=0)
    ue_imf = (ue_imf / 255.).clamp(0, 1)
    return ue_imf