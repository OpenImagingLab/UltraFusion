import torch
from einops import rearrange


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


def IMF2(ue, oe):
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