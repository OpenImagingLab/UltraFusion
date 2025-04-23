import os, random, glob, time
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor


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


class TestDataset(data.Dataset):
    def __init__(self, dataset):
        super(TestDataset, self).__init__()
        self.dataset = dataset
        self.img_dir_dict = {
            'UltraFusion': './data/UltraFusionBenchmark',
            'MEFB': './data/MEFB',
            'RealHDRV': './data/Real-HDRV-Deghosting-sRGB-Testing',
        }
        self.ldr_list1 = []
        self.ldr_list2 = []
        self.file_name_list = []
        self.to_tensor = ToTensor()

        self.scene_list = os.listdir(self.img_dir_dict[dataset])
        self.scene_list.sort()
        for scene in self.scene_list:
            if len(os.listdir(os.path.join(self.img_dir_dict[dataset], scene))) > 0:
                self.ldr_list1.append(glob.glob(os.path.join(self.img_dir_dict[dataset], scene, '*ue.*'))[0])
                self.ldr_list2.append(glob.glob(os.path.join(self.img_dir_dict[dataset], scene, '*oe.*'))[0])
                self.file_name_list.append('{}_{}'.format(dataset, scene))
        

    def __getitem__(self, index):
        ldr1_path = self.ldr_list1[index]
        ldr2_path = self.ldr_list2[index]
        file_name = self.file_name_list[index]

        ldr1 = Image.open(ldr1_path).convert('RGB')
        ldr2 = Image.open(ldr2_path).convert('RGB')

        W, H = ldr1.size

        if W * H >= 6000 * 4000:
            ldr1 = ldr1.resize([W // 4, H // 4])
            ldr2 = ldr2.resize([W // 4, H // 4])
        elif W * H >= 2000 *1500:
            ldr1 = ldr1.resize([W * 2 // 5, H * 2 // 5])
            ldr2 = ldr2.resize([W * 2 // 5, H * 2 // 5])

        ldr1 = self.to_tensor(ldr1)
        ldr2 = self.to_tensor(ldr2)

        return {
            'ue': ldr1,
            'oe': ldr2,
            'file_name': file_name
        }

    def __len__(self):
        return len(self.ldr_list1)