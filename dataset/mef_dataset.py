import os, random, glob, time
import tqdm
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, CenterCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip

from torch.utils.data import DataLoader
from torchvision.utils import save_image


def get_color_and_struct(isrgb, input_img: torch.Tensor, ksize, sigmaX, c):  #input an RGB image

    input_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)

    if isrgb==True:
        yuv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2YUV).astype(np.float32)
        y = np.expand_dims(yuv_img[:,:,0], axis=-1).astype(np.float64)
        u = np.expand_dims(yuv_img[:,:,1], axis=-1).astype(np.float32)
        v = np.expand_dims(yuv_img[:,:,2], axis=-1).astype(np.float32)
    else:
        y = input_img.astype(np.float64)
    #mu = gaussian_filter(y, ksize, ksize/6)
    mu = cv2.GaussianBlur(y, (ksize,ksize), sigmaX).astype(np.float64)
    mu_sq = mu * mu
    sigma = np.sqrt(np.absolute(cv2.GaussianBlur(y*y, (ksize,ksize), sigmaX) - mu_sq)).astype(np.float64)
    mu = np.expand_dims(mu, axis=-1)
    sigma = np.expand_dims(sigma, axis=-1)
    dividend = y.astype(np.float64) - mu
    divisor = sigma + c
    struct = dividend / divisor
    struct = struct.astype(np.float32)
    struct_norm = (struct - struct.min()) / (struct.max() - struct.min() + 1e-6)
    struct_norm = torch.from_numpy(struct_norm).permute(2, 0, 1)
    u = torch.from_numpy(u).permute(2, 0, 1)
    v = torch.from_numpy(v).permute(2, 0, 1)
    img_uv = torch.cat([u, v], dim=0)
    return struct_norm, img_uv


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


class MEFDataset(data.Dataset):
    def __init__(self, img_dir, motion_img_dir, random_crop=True, random_resize=True, rotate=True, flip=True):
        super(MEFDataset, self).__init__()
        self.img_dir = img_dir
        self.motion_img_dir = motion_img_dir
        self.random_crop = random_crop
        self.random_resize = random_resize
        self.rotate = rotate
        self.flip = flip
        self.to_tensor = ToTensor()

        self.ldr_list1 = []
        self.ldr_list2 = []
        self.gt_list = []
        self.file_name_list = []
        self.img_dir_subset_list = os.listdir(self.img_dir)
        for img_dir_subset in self.img_dir_subset_list:
            if img_dir_subset != 'Label':
                self.ldr_list1.append(os.path.join(self.img_dir, img_dir_subset, 'se.JPG'))
                self.ldr_list2.append(os.path.join(self.img_dir, img_dir_subset, 'le.JPG'))
                gt_path1 = glob.glob(os.path.join(self.img_dir, 'Label', '{}.*'.format(img_dir_subset)))
                gt_path2 = glob.glob(os.path.join(self.img_dir, 'Label', '{}_align.*'.format(img_dir_subset)))
                if len(gt_path2) > 0:
                    gt_path = gt_path2
                else:
                    gt_path = gt_path1
                self.gt_list.append(gt_path[0])
                self.file_name_list.append(img_dir_subset)
        
        self.motion_mask_list = os.listdir(self.motion_img_dir)

    def __getitem__(self, index):
        gt_path = self.gt_list[index % len(self.gt_list)]
        lq1_path = self.ldr_list1[index % len(self.gt_list)]
        lq2_path = self.ldr_list2[index % len(self.gt_list)]
        file_name = self.file_name_list[index % len(self.gt_list)]
        lq1 = Image.open(lq1_path).convert('RGB')
        lq2 = Image.open(lq2_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if 'align' in gt_path:
            # crop black region caused by warp
            W, H = gt.size
            cc = CenterCrop([H - 100, W - 100])
            lq1 = cc(lq1)
            lq2 = cc(lq2)
            gt = cc(gt)
        if self.random_resize:
            W, H = gt.size
            min_size = 512
            max_size = min(H, W)
            tgt_size = torch.randint(min_size, max_size + 1, (1, )).item()
            lq1 = Resize(tgt_size)(lq1)
            lq2 = Resize(tgt_size)(lq2)
            gt = Resize(tgt_size)(gt)
        if self.random_crop:
            crop_params = RandomCrop.get_params(gt, [512, 512])
            lq1 = crop(lq1, *crop_params)
            lq2 = crop(lq2, *crop_params)
            gt = crop(gt, *crop_params)
        else:
            lq1 = CenterCrop(512)(lq1)
            lq2 = CenterCrop(512)(lq2)
            gt = CenterCrop(512)(gt)
        if self.rotate:
            rotate_params = random.randint(0, 3) * 90
            lq1 = rotate(lq1, rotate_params)
            lq2 = rotate(lq2, rotate_params)
            gt = rotate(gt, rotate_params)
        if self.flip:
            if torch.rand(1) > 0.5:
                lq1 = RandomHorizontalFlip(1)(lq1)
                lq2 = RandomHorizontalFlip(1)(lq2)
                gt = RandomHorizontalFlip(1)(gt)
            if torch.rand(1) > 0.5:
                lq1 = RandomVerticalFlip(1)(lq1)
                lq2 = RandomVerticalFlip(1)(lq2)
                gt = RandomVerticalFlip(1)(gt)

        lq1 = self.to_tensor(lq1)
        lq2 = self.to_tensor(lq2)
        gt = self.to_tensor(gt) 

        motion_type = torch.rand(1)
        if motion_type < 0.75:
            # local motion
            motion_ind = torch.randint(0, len(self.motion_mask_list), (1, )).item()
            mask = Image.open(os.path.join(self.motion_img_dir, self.motion_mask_list[motion_ind])).convert('RGB').resize([512, 512])
            mask = self.to_tensor(mask)
            mask = mask[:1, :, :]
            lq1_motion = lq1 * (1. - mask)
        else:
            mask = torch.zeros_like(gt)[:1, :, :]
            lq1_motion = lq1

        lq1_struct, lq1_color = get_color_and_struct(isrgb=True, input_img=lq1_motion, ksize=7, sigmaX=0, c=0.0000001)
        
        # Normalize to [-1, 1]
        gt = gt * 2 - 1
        lq1 = lq1 * 2 - 1
        lq1_motion = lq1_motion * 2 - 1
        lq2 = lq2 * 2 - 1

        return {
            'gt': gt,
            'lq1_struct': lq1_struct,
            'lq1_color': lq1_color,
            'lq2': lq2,
            'mask': mask,
            # 'fm': fuse_map,
            'prompt': '',
            'file_name': file_name
        }

    def __len__(self):
        return len(self.gt_list) * 10
        # return 1