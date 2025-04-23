import os, tqdm, math
import torch
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf
from accelerate.utils import set_seed

from dataset.test_dataset import TestDataset, get_color_and_struct
from model.raft.raft import RAFT
from utils.common import instantiate_from_config
from utils.flow import backward_warp, forward_backward_consistency_check, IMF


def pad_imgv3(x, crop_size, crop_step):
    _, _, h, w = x.size()
    n_h = max(math.ceil((h - crop_size) / crop_step), 0)
    n_w = max(math.ceil((w - crop_size) / crop_step), 0)
    h_target = crop_size + n_h * crop_step
    w_target = crop_size + n_w * crop_step
    mod_pad_h = h_target - h
    mod_pad_w = w_target - w
    x_np = x.cpu().numpy()
    x_np = np.pad(x_np, pad_width=((0,0),(0,0),(0,mod_pad_h),(0,mod_pad_w)), mode='reflect')
    res = torch.from_numpy(x_np).cuda()
    return res


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def crop_parallel(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=torch.Tensor().to(img.device)
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list = torch.cat([lr_list, crop_img])
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w


def combine_parallel_wo_artifact(sr_list, num_h, num_w, new_h, new_w, patch_size, step):
    p_size = patch_size
    pred_lr_list = sr_list

    pred_full_w_list = [] # rectangle
    for i in range(num_h):
        pred_full_w = torch.zeros([1, 3, p_size, new_w]).cuda()
        pred_full_w[:, :, :, 0 : step] = pred_lr_list[i * num_w][:, :, 0 : step]
        # pred_full_w[:, :, :, 0 : patch_size] = pred_lr_list[i * num_w][:, :, 0 : patch_size]
        pred_full_w[:, :, :, new_w - step :] = pred_lr_list[i * num_w + num_w - 1][:, :, -step:]
        for j in range(1, num_w):
            repeat_l = j * step
            repeat_r = repeat_l + (p_size - step)
            ind = i * num_w + j - 1

            pred_full_w[:, :, :, repeat_r : repeat_r + 2 * step - patch_size] = pred_lr_list[ind + 1][:, :, patch_size - step : step]

            for k in range(repeat_l, repeat_r):
                alpha = (k - repeat_l) / (repeat_r - repeat_l)
                pred_full_w[:, :, :, k] = pred_lr_list[ind][:, :, step + k - repeat_l] * (1 - alpha) + pred_lr_list[ind + 1][:, :, k - repeat_l] * alpha
                # pred_full_w[:, :, :, k] = pred_full_w[:, :, :, k] * (1 - alpha) + pred_lr_list[ind + 1][:, :, k - repeat_l] * alpha
        pred_full_w_list.append(pred_full_w)
    
    pred = torch.zeros([1, 3, new_h, new_w], device=sr_list[0].device)
    pred[:, :, 0 : step, :] = pred_full_w_list[0][:, :, 0 : step, :]
    # pred[:, :, 0 : patch_size, :] = pred_full_w_list[0][:, :, 0 : patch_size, :]
    pred[:, :, -step :, :] = pred_full_w_list[-1][:, :, -step :, :]
    for i in range(1, num_h):
        repeat_u = i * step
        repeat_d = repeat_u + (p_size - step)
        for k in range(repeat_u, repeat_d):
            alpha = (k - repeat_u) / (repeat_d - repeat_u)
            pred[:, :, k, :] = pred_full_w_list[i - 1][:, :, step + k - repeat_u, :] * (1 - alpha) + pred_full_w_list[i][:, :, k - repeat_u, :] * alpha
            # pred[:, :, k, :] = pred[:, :, k, :] * (1 - alpha) + pred_full_w_list[i][:, :, k - repeat_u, :] * alpha
    return pred, pred_full_w_list


def mef(img1, img2, img_name, flow_model, pipe, args, consistent_start=None):
    _, _, H, W = img2.shape
    img1 = pad_img(img1, 16)
    img2 = pad_img(img2, 16)
    img1_light = IMF(img1, img2)
    with torch.no_grad():
        _, img12_flow = flow_model(img1_light * 2 - 1, img2 * 2 - 1, iters=20, test_mode=True)
        _, img21_flow = flow_model(img2 * 2 - 1, img1_light * 2 - 1, iters=20, test_mode=True)

    img12 = backward_warp(img1, img21_flow)
    _, occ_mask = forward_backward_consistency_check(img12_flow, img21_flow)
    occ_mask = occ_mask.unsqueeze(dim=1)
    img12_mask = img12 * (1. - occ_mask)

    img1 = img1[:, :, :H, :W]
    img2 = img2[:, :, :H, :W]
    img12 = img12[:, :, :H, :W]
    img12_mask = img12_mask[:, :, :H, :W]
    occ_mask = occ_mask[:, :, :H, :W]

    if not args.prealign:
        img12_mask = img1 # cancel pre-align


    img2 = pad_imgv3(img2, args.tile_size, args.tile_stride)
    img12_mask = pad_imgv3(img12_mask, args.tile_size, args.tile_stride)

    img1_mscn_norm, img1_color = get_color_and_struct(isrgb=True, input_img=img12_mask, ksize=7, sigmaX=0, c=0.0000001) 
    img1_mscn_norm, img1_color = img1_mscn_norm.unsqueeze(dim=0), img1_color.unsqueeze(dim=0)
    img1_mscn_norm, img1_color = img1_mscn_norm.cuda(), img1_color.cuda()
    fidelity_input = torch.cat([img2, img1_mscn_norm, img1_color], dim=1)

    img2_patches, num_h, num_w, new_h, new_w = crop_parallel(img2, args.tile_size, args.tile_stride)
    img1_struct_patches, num_h, num_w, new_h, new_w = crop_parallel(img1_mscn_norm, args.tile_size, args.tile_stride)
    img1_color_patches, num_h, num_w, new_h, new_w = crop_parallel(img1_color, args.tile_size, args.tile_stride)
    img2_patches_list = torch.split(img2_patches, 1, dim=0)
    img1_struct_patches_list = torch.split(img1_struct_patches, 1, dim=0)
    img1_color_patches_list = torch.split(img1_color_patches, 1, dim=0)
    fidelity_input_patches, num_h, num_w, new_h, new_w = crop_parallel(fidelity_input, args.tile_size, args.tile_stride)
    fidelity_input_patches_list = torch.split(fidelity_input_patches, 1, dim=0)
    out_list = []
    for ind, (img2_, img1_struct_, img1_color_, fidelity_input_) in enumerate(zip(img2_patches_list, img1_struct_patches_list, img1_color_patches_list, fidelity_input_patches_list)):
        set_seed(args.seed)
        out = pipe.run(lq2=img2_, lq1_mscn_norm=img1_struct_, lq1_color=img1_color_, tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride, cond_fn=cond_fn, fidelity_input=fidelity_input_, consistent_start=consistent_start) # [-1, 1]
        out_list.append(out)
    
    out_list = torch.cat(out_list, dim=0)
    out, _ = combine_parallel_wo_artifact(out_list, num_h, num_w, new_h, new_w, args.tile_size, args.tile_stride)

    out = out[:, :, :H, :W]
    img1 = img1[:, :, :H, :W]
    img1_light = img1_light[:, :, :H, :W]
    img12 = img12[:, :, :H, :W]
    img2 = img2[:, :, :H, :W]
    occ_mask = occ_mask[:, :H, :W]
    img12_mask = img12_mask[:, :, :H, :W]
    img1_mscn_norm = img1_mscn_norm[:, :, :H, :W] 
    img1_color = img1_color[:, :, :H, :W]

    u = torch.zeros_like(out)
    v = torch.zeros_like(out)
    u[:, 1:, :, :] = img1_color
    v[:, :2, :, :] = img1_color

    save_image((out + 1) / 2, '{}/{}_out_{}.png'.format(args.output, img_name, 'align' if args.prealign else 'noalign'))
    if args.save_all:
        save_image(img1, '{}/{}_ue.png'.format(args.output, img_name))
        save_image(img1_light, '{}/{}_ue_imf.png'.format(args.output, img_name))
        save_image(img12_mask, '{}/{}_ue2oe_mask_{}.png'.format(args.output, img_name, 'align' if args.prealign else 'noalign'))
        save_image(img12, '{}/{}_ue2oe_{}.png'.format(args.output, img_name, 'align' if args.prealign else 'noalign'))
        save_image(img1_mscn_norm, '{}/{}_ue2oe_mask_mscn_{}.png'.format(args.output, img_name, 'align' if args.prealign else 'noalign'))
        save_image(img2, '{}/{}_oe.png'.format(args.output, img_name))
        save_image(occ_mask, '{}/{}_occmask_{}.png'.format(args.output, img_name, 'align' if args.prealign else 'noalign'))
        save_image(u, '{}/{}_u.png'.format(args.output, img_name))
        save_image(v, '{}/{}_v.png'.format(args.output, img_name))

    return out

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default='MEFB')
parser.add_argument("--output", default='results', type=str)
parser.add_argument("--tiled", action='store_true', default=False)
parser.add_argument("--tile_size", type=int, default=512)
parser.add_argument("--tile_stride", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
parser.add_argument("--seed", type=int, default=231, choices=["cpu", "cuda", "mps"])
parser.add_argument("--prealign", action='store_true', default=False)
parser.add_argument("--save_all", action='store_true', default=False)
args = parser.parse_args()

from model.V4_CA.cldm import ControlLDM
from model.V4_CA.gaussian_diffusion import Diffusion
from pipeline.V4_CA.pipeline import UltraFusionPipeline
### load uent, vae, clip
cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/ultrafusion.yaml").model.cldm)
sd = torch.load('ckpts/v2-1_512-ema-pruned.ckpt', map_location="cpu")["state_dict"]
unused = cldm.load_pretrained_sd(sd)
print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
### load controlnet
control_sd = torch.load('ckpts/ultrafusion.pt', map_location="cpu")
cldm.load_controlnet_from_ckpt(control_sd)
print(f"strictly load controlnet weight")
cldm.eval().to(args.device)
### load fidelity encoder
fidelity_encoder = instantiate_from_config(OmegaConf.load("configs/fcb.yaml").model.fidelity_encoder)
fidelity_encoder_sd = torch.load('ckpts/fcb.pt')
fidelity_encoder.load_state_dict(fidelity_encoder_sd, strict=True)
fidelity_encoder = fidelity_encoder.cuda()
fidelity_encoder.eval()
### load diffusion
diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/ultrafusion.yaml").model.diffusion)
diffusion.to(args.device)
### load flow model
flow_state_dict = torch.load('ckpts/raft-sintel.pth', map_location='cpu')
flow_model = RAFT(args).cuda()
new_flow_state_dict = OrderedDict()
for k, v in flow_state_dict.items():
    new_flow_state_dict[k.replace("module.", "")] = v
flow_model.load_state_dict(new_flow_state_dict)
flow_model = flow_model.eval()
cond_fn = None

pipe = UltraFusionPipeline(cldm=cldm, diffusion=diffusion, fidelity_encoder=fidelity_encoder, device=args.device)

to_tensor = ToTensor()

dataset = TestDataset(args.dataset)
dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    num_workers=0
)

if not os.path.exists(args.output):
    os.mkdir(args.output)
args.output = os.path.join(args.output, args.dataset)
if not os.path.exists(args.output):
    os.mkdir(args.output)

for batch in dataloader:
    ue = batch['ue'].cuda()
    oe = batch['oe'].cuda()
    img_name = batch['file_name'][0]

    _ = mef(img1=ue, img2=oe, img_name=img_name, flow_model=flow_model, pipe=pipe, args=args, consistent_start=None)