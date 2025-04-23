import os,glob
import torch
import pyiqa
from tqdm import tqdm


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1
        self.min = 10000

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val
    
    def get_max(self):
         return self.max


metric_list = {
    'musiq': pyiqa.create_metric('musiq', as_loss=False).cuda(),
    'paq2piq': pyiqa.create_metric('paq2piq', as_loss=False).cuda(),
    'hyperiqa': pyiqa.create_metric('hyperiqa', as_loss=False).cuda(),
}
res_list = {}

for k in metric_list:
    print('{} lower better: {}'.format(k, metric_list[k].lower_better))
    res_list[k] = AverageMeter()

img_list = glob.glob('/ailab/user/chenzixuan/Research/Diff-HDR/cvpr2025_release/MEFB/*out*.png')

for img_path in tqdm(img_list):
    for k in metric_list:
        tmp = metric_list[k](img_path)
        res_list[k].update(tmp)

for k in res_list:
    print('{}: {}'.format(k, res_list[k].avg))