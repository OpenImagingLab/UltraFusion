from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from model.V4_CA.controlnet import (
    ControlledUnetModel, ControlNet,
)
from model.V4_CA.vae import AutoencoderKL
from model.clip import FrozenOpenCLIPEmbedder

from utils.common import sliding_windows, count_vram_usage, gaussian_weights
from torchvision.utils import save_image


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self,
        unet_cfg,
        vae_cfg,
        clip_cfg,
        controlnet_cfg,
        latent_scale_factor
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_sd(self, sd: Dict[str, torch.Tensor]) -> Set[str]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=True)
        unused = set(sd.keys()) - used
        # NOTE: this is slightly different from previous version, which haven't switched
        # the UNet to eval mode and disabled the requires_grad flag.
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused
    
    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                elif this.size(1) > target.size(1):
                    print(this.size(1), target.size(1))
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
                else:
                    print(this.size(1), target.size(1))
                    d_ic = this.size(1)
                    print(target.shape)
                    init_sd[key] = target[:, :d_ic, :, :]
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    def vae_encode(self, image: torch.Tensor, sample: bool=True) -> torch.Tensor:
        if sample:
            return self.vae.encode(image).sample() * self.scale_factor
        else:
            return self.vae.encode(image).mode() * self.scale_factor
    
    def vae_encode_tiled(self, image: torch.Tensor, tile_size: int, tile_stride: int, sample: bool=True, fidelity_encoder=None, fidelity_input: torch.Tensor=None) -> torch.Tensor:
        bs, _, h, w = image.shape
        z = torch.zeros((bs, 4, h // 8, w // 8), dtype=torch.float32, device=image.device)
        count = torch.zeros_like(z, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=image.device)
        tiles = sliding_windows(h // 8, w // 8, tile_size // 8, tile_stride // 8)
        skip_feats = []
        for hi, hi_end, wi, wi_end in tiles:
            tile_image = image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8]
            if fidelity_input is not None:
                tile_fidelity_input = fidelity_input[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8]
            z[:, :, hi:hi_end, wi:wi_end] += self.vae_encode(tile_image, sample=sample) * weights
            with torch.no_grad():
                # tile_images = torch.cat([tile_image_ue, tile_image], dim=1)
                # skip_feat = fidelity_encoder(tile_images)
                if fidelity_input is not None:
                    skip_feat = fidelity_encoder(tile_fidelity_input)
                    skip_feats.append(skip_feat)
                else:
                    skip_feats.append(None)
            count[:, :, hi:hi_end, wi:wi_end] += weights
        z.div_(count)
        return z, skip_feats
    
    def vae_decode(self, z: torch.Tensor, skip_feat=None) -> torch.Tensor:
        return self.vae.decode(z / self.scale_factor, skip_feat)
    

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        # style_mean = torch.tensor([0.6906, 0.6766, 0.6749]).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
        # style_std = torch.tensor([0.1955, 0.2096, 0.2236]).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

        k = (style_std) / (content_std)
        b = style_mean - content_mean * style_std / content_std
        res = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        return res

    
    @count_vram_usage
    def vae_decode_tiled(self, z: torch.Tensor, tile_size: int, tile_stride: int, skip_feats=None, consistent_start=None) -> torch.Tensor:
        bs, _, h, w = z.shape
        image = torch.zeros((bs, 3, h * 8, w * 8), dtype=torch.float32, device=z.device)
        count = torch.zeros_like(image, dtype=torch.float32)
        weights = gaussian_weights(tile_size * 8, tile_size * 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=z.device)
        tiles = sliding_windows(h, w, tile_size, tile_stride)
        for ind, ((hi, hi_end, wi, wi_end), skip_feat) in enumerate(zip(tiles, skip_feats)):
            tile_z = z[:, :, hi:hi_end, wi:wi_end]
            tile_z_decoded = self.vae_decode(tile_z, skip_feat)
            if consistent_start is not None:
                tile_z_decoded = self.adaptive_instance_normalization(tile_z_decoded, consistent_start[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8])
            image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += tile_z_decoded * weights
            count[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += weights
        image.div_(count)
        return image


    def prepare_condition(self, lq2: torch.Tensor, lq1_struct: torch.Tensor, lq1_color: torch.Tensor, txt: List[str]) -> Dict[str, torch.Tensor]:
        # Note the lq_vis should be normalized to [-1, 1] and lq_ifr normalized to [0, 1]!!!
        return dict(
            c_txt=self.clip.encode(txt),
            c_lq2=self.vae_encode(lq2, sample=False),
            c_lq1_mscn_norm=lq1_struct,
            c_lq1_color=lq1_color
        )
    

    @count_vram_usage
    def prepare_condition_tiled(self, lq2: torch.Tensor, lq1_mscn_norm:torch.Tensor, lq1_color: torch.Tensor, txt: List[str], tile_size: int, tile_stride: int, fidelity_encoder=None, fidelity_input: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        c_lq2, skip_feats = self.vae_encode_tiled(lq2, tile_size, tile_stride, sample=False, fidelity_encoder=fidelity_encoder, fidelity_input=fidelity_input) # why smaple = False ?
        return dict(
            c_txt=self.clip.encode(txt),
            c_lq2=c_lq2,
            c_lq1_mscn_norm = lq1_mscn_norm,
            c_lq1_color = lq1_color
        ), skip_feats

    def forward(self, x_noisy, t, cond):
        c_txt = cond["c_txt"]

        c_lq2 = cond["c_lq2"]
        c_lq1_mscn_norm = cond["c_lq1_mscn_norm"]
        c_lq1_color = cond["c_lq1_color"]
        c_img = [c_lq2, c_lq1_mscn_norm, c_lq1_color]

        control = self.controlnet(
            x=x_noisy, hint=c_img,
            timesteps=t, context=c_txt
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy, timesteps=t,
            context=c_txt, control=control, only_mid_control=False
        )
        return eps
