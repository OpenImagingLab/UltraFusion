import torch
import torchvision.transforms.functional as tvtf

from typing import overload, Tuple
from torch.nn import functional as F
from torchvision.utils import save_image


class Guidance:

    def __init__(self, scale: float, t_start: int, t_stop: int, space: str, repeat: int) -> "Guidance":
        """
        Initialize restoration guidance.

        Args:
            scale (float): Gradient scale (denoted as `s` in our paper). The larger the gradient scale, 
                the closer the final result will be to the output of the first stage model.
            t_start (int), t_stop (int): The timestep to start or stop guidance. Note that the sampling 
                process starts from t=1000 to t=0, the `t_start` should be larger than `t_stop`.
            space (str): The data space for computing loss function (rgb or latent).

        Our restoration guidance is based on [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior).
        Thanks for their work!
        """
        self.scale = scale * 3000
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat
    
    def load_target(self, target: torch.Tensor) -> None:
        self.target = target

    def __call__(self, target_x0, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # avoid propagating gradient out of this scope
        pred_x0 = pred_x0.detach().clone()
        tmp1, tmp2 = target_x0
        tmp1 = tmp1.detach().clone()
        tmp2 = tmp2.detach().clone()
        return self._forward([tmp1, tmp2], pred_x0, t)
        # return self._forward([tmp1, tmp2], pred_x0, t)
    
    @overload
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        ...


class MSEGuidance(Guidance):

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        target_y = 0.299 * target_x0[:, 0, :, :] + 0.587 * target_x0[:, 1, :, :] + 0.114 * target_x0[:, 2, :, :]
        weight_map = (target_y > 0.99).type(torch.float32)
        save_image(weight_map, 'weight_map.png')
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * (1. - weight_map)).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()


class WeightedMSEGuidance(Guidance):

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        # convert RGB to G
        rgb_to_gray_kernel = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        target = torch.sum(target * rgb_to_gray_kernel.to(target.device), dim=1, keepdim=True)
        # initialize sobel kernel in x and y axis
        G_x = [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]
        G_y = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]
        G_x = torch.tensor(G_x, dtype=target.dtype, device=target.device)[None]
        G_y = torch.tensor(G_y, dtype=target.dtype, device=target.device)[None]
        G = torch.stack((G_x, G_y))

        target = F.pad(target, (1, 1, 1, 1), mode='replicate') # padding = 1
        grad = F.conv2d(target, G, stride=1)
        mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()

        n, c, h, w = mag.size()
        block_size = 2
        blocks = mag.view(n, c, h // block_size, block_size, w // block_size, block_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        block_mean = blocks.sum(dim=(-2, -1), keepdim=True).tanh().repeat(1, 1, 1, 1, block_size, block_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        block_mean = block_mean.view(n, c, h, w)
        weight_map = 1 - block_mean

        return weight_map

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        with torch.no_grad():
            w = self._get_weight((target_x0 + 1) / 2)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()



class L_spa(torch.nn.Module):

    def __init__(self, patch_size):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)

        # kernel_left_up = torch.FloatTensor( [[-1,0,0],[0,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_right_up = torch.FloatTensor( [[0,0,-1],[0,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_left_down = torch.FloatTensor( [[0,0,0],[0,1, 0 ],[-1,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_right_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,0,-1]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_left = torch.nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = torch.nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = torch.nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = torch.nn.Parameter(data=kernel_down, requires_grad=False)
        # self.weight_left_up = nn.Parameter(data=kernel_left_up, requires_grad=False)
        # self.weight_right_up = nn.Parameter(data=kernel_right_up, requires_grad=False)
        # self.weight_left_down = nn.Parameter(data=kernel_left_down, requires_grad=False)
        # self.weight_right_down = nn.Parameter(data=kernel_right_down, requires_grad=False)
        self.pool = torch.nn.AvgPool2d(patch_size)

    def forward(self, org , enhance, weight_map):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
        weight_map_pool = self.pool(weight_map)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        # D_org_left_up = F.conv2d(org_pool , self.weight_left_up, padding=1)
        # D_org_right_up = F.conv2d(org_pool , self.weight_right_up, padding=1)
        # D_org_left_down = F.conv2d(org_pool , self.weight_left_down, padding=1)
        # D_org_right_down = F.conv2d(org_pool , self.weight_right_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        # D_enhance_left_up = F.conv2d(enhance_pool , self.weight_left_up, padding=1)
        # D_enhance_right_up = F.conv2d(enhance_pool , self.weight_right_up, padding=1)
        # D_enhance_left_down = F.conv2d(enhance_pool , self.weight_left_down, padding=1)
        # D_enhance_right_down = F.conv2d(enhance_pool , self.weight_right_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)

        # D_left_up = torch.pow(D_org_left_up - D_enhance_left_up, 2)
        # D_right_up = torch.pow(D_org_right_up - D_enhance_right_up, 2)
        # D_left_down = torch.pow(D_org_left_down - D_enhance_left_down, 2)
        # D_right_down = torch.pow(D_org_right_down - D_enhance_right_down, 2)

        E = (D_left + D_right + D_up +D_down) 

        return torch.mean(E * weight_map_pool)


class L_mscn(torch.nn.Module):

    def __init__(self):
        super(L_mscn, self).__init__()
        self.l1 = torch.nn.L1Loss()
    def forward(self, img1, img2, weight_map):
        y1 = 0.299 * img1[:, 0, :, :] + 0.587 * img1[:, 1, :, :] + 0.114 * img1[:, 2, :, :]
        y2 = 0.299 * img2[:, 0, :, :] + 0.587 * img2[:, 1, :, :] + 0.114 * img2[:, 2, :, :]
        y1 = y1.type(torch.float64)
        y2 = y2.type(torch.float64)
        mu1 = tvtf.gaussian_blur(y1, (7, 7))
        mu2 = tvtf.gaussian_blur(y2, (7, 7))
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        sigma1 = torch.sqrt(torch.abs(tvtf.gaussian_blur(y1 * y1, (7, 7)) - mu1_sq))
        sigma2 = torch.sqrt(torch.abs(tvtf.gaussian_blur(y2 * y2, (7, 7)) - mu2_sq))
        dividend1 = y1 - mu1
        dividend2 = y2 - mu2
        divisor1 = sigma1 + 1e-7
        divisor2 = sigma2 + 1e-7
        struct1 = (dividend1 / divisor1)
        struct2 = (dividend2 / divisor2)
        struct1_norm = (struct1 - struct1.min()) / (struct1.max() - struct1.min())
        struct2_norm = (struct2 - struct2.min()) / (struct2.max() - struct2.min())
        
        return ((struct1 - struct2).pow(2) * weight_map).mean()
        # return self.l1(struct1_norm, struct2_norm)
        # return (struct1_norm - struct2_norm).mean()


class StructureGuidance(Guidance):

    def __init__(self, scale: float, t_start: int, t_stop: int, space: str, repeat: int) -> Guidance:
        super(StructureGuidance, self).__init__(scale, t_start, t_stop, space, repeat)
        # self.spa1_loss = L_spa(1)
        # self.spa2_loss = L_spa(2)
        self.spa4_loss = L_spa(1)
        # self.loss = L_mscn()
    
    def mscn_torch(self, input_img: torch.Tensor, ksize, c):  #input an RGB image
        y = 0.299 * input_img[:, 0, :, :] + 0.587 * input_img[:, 1, :, :] + 0.114 * input_img[:, 2, :, :]
        y = y.type(torch.float64)
        mu = tvtf.gaussian_blur(y, (ksize,ksize))
        mu_sq = mu * mu
        sigma = torch.sqrt(torch.abs(tvtf.gaussian_blur(y*y, (ksize,ksize)) - mu_sq))
        dividend = y - mu
        divisor = sigma + c
        struct = (dividend / divisor).type(torch.float32)
        struct_norm = (struct - struct.min()) / (struct.max() - struct.min() + 1e-6)
        return struct_norm

    def _forward(self, target_x0, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # pred_x0: [-1, 1], nchw, rgb
        target1, target2 = target_x0
        pred_x0 = (pred_x0 + 1) / 2 # [0, 1]
        target1_struct = self.mscn_torch(input_img=target1, ksize=7, c=1e-7)
        target2_struct = self.mscn_torch(input_img=target2, ksize=7, c=1e-7)
        target2_y = 0.299 * target2[:, 0, :, :] + 0.587 * target2[:, 1, :, :] + 0.114 * target2[:, 2, :, :]
        weight_map = (target2_y > 0.99).type(torch.float32)
        
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            # loss = (pred_x0_struct - target_x0_struct).pow(2).mean((1, 2)).sum()
            # loss = self.spa_loss(target2, pred_x0, 1. - weight_map)
            loss = self.spa4_loss(target2, pred_x0, 1. - weight_map)
            # loss = self.loss(target1, pred_x0, 1. - weight_map)
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()