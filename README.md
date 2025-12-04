<p align="center">

  <h2 align="center">
  UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion

  (CVPR 2025 Highlight)
  </h2>
  <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=pwixOhcAAAAJ&hl=zh-CN"><strong>Zixuan Chen</strong></a><sup>1,3</sup>
    ·
    <a><strong>Yujin Wang</strong></a><sup>1</sup>
    ·
    <a href="https://caixin98.github.io/"><strong>Xin Cai</strong></a><sup>2</sup>
    ·
    <a href="https://zhiyuanyou.github.io/"><strong>Zhiyuan You</strong></a><sup>2</sup>
    <br>
    <a href="https://person.zju.edu.cn/lzmhome"><strong>Zheming Lu</strong></a><sup>3</sup>
    ·
    <a><strong>Fan Zhang</strong></a><sup>1</sup>
    ·
    <a href="https://guoshi28.github.io/"><strong>Shi Guo</strong></a><sup>1</sup>
    ·
    <a href="https://tianfan.info/"><strong>Tianfan Xue</strong></a><sup>2,1</sup>
    <!-- <br> -->
    <br>
    <sup>1</sup>Shanghai AI Laboratory, <sup>2</sup>The Chinese University of Hong Kong, 
    <sup>3</sup>Zhejiang University  
    <br>
    <div align="center">
    <a href="https://arxiv.org/abs/2501.11515"><img src='https://img.shields.io/badge/arXiv-UltraFusion-red' alt='Paper PDF'></a>
    <a href='https://openimaginglab.github.io/UltraFusion/'><img src='https://img.shields.io/badge/Project_Page-UltraFusion-blue' alt='Project Page'></a>
    <a href='https://huggingface.co/spaces/iimmortall/UltraFusion'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
    </div>
  </p>
</p>
  
![teaser_img](assets/teaser.png)

## :mega: News
- **2024.4.23**: Inference codes, benchmark and results are released.
- **2024.4.5**: Our UltraFusion is selected to be presented as a :sparkles:***highlight***:sparkles: in CVPR 2025.
- **2025.2.27**: Accepeted by ***CVPR 2025*** :tada::tada::tada:.
- **2025.1.21**: Feel free to try online demos at <a href="https://huggingface.co/spaces/iimmortall/UltraFusion">Hugging Face</a> and <a href="https://openxlab.org.cn/apps/detail/OpenImagingLab/UltraFusion">OpenXLab</a> :blush:.


## :memo: ToDo List
- [ ] Release training codes.
- [x] Release inference codes and pre-trained model. 
- [x] Release UltraFusion benchmark and visual results.
- [x] Release more visual comparison in our [project page](https://openimaginglab.github.io/UltraFusion/)

## :bridge_at_night: UltraFusion Benchmark (UltraFusion100)
We capture 100 challenging real-world HDR scenes for performance evaluation. 
Our benchmark (UltraFusion100) and results (include competing methods) are availble at [Google Drive](https://drive.google.com/drive/folders/18icr4A_0qGvwqehPhxH29hqJYO8HS6bi?usp=sharing) and [Baidu Disk](). 
Moreover, we also provide results of our method and the comparison methods on [RealHDV](https://github.com/yungsyu99/Real-HDRV) and [MEFB](https://github.com/xingchenzhang/MEFB).

> *<sub>Note: The HDR reconstruction methods perform poorly in some scenes because we follow their setup to retrain 2-exposure version, while the training set they used only provide ground truth for the middle exposure, limiting the dynamic range. We believe that using training data with higher dynamic range can improve performance.</sub>*

## Quick Start
**Installation**
```shell
# clone this repo
git clone https://github.com/OpenImagingLab/UltraFusion.git
cd UltraFusion

# create environment
conda create -n UltraFusion python=3.10
conda activate UltraFusion
pip install -r requirements.txt
```
**Prepare Data and Pre-trained Model**

Download [raft-sintel.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing), [v2-1_512-ema-pruned.ckpt](https://drive.google.com/file/d/1IFFqrIoRVVtSE9UxWDeMB2rfOa2ox-4h/view?usp=drive_link), [fcb.pt](https://huggingface.co/zxchen00/UltraFusion/blob/main/fcb.pt) and [ultrafusion.pt](https://huggingface.co/zxchen00/UltraFusion/blob/main/ultrafusion.pt), and put them in ```ckpts``` folder. Download three benchmarks ([Google Drive](https://drive.google.com/drive/folders/18icr4A_0qGvwqehPhxH29hqJYO8HS6bi?usp=sharing) or [Baidu Disk]()) and put them in ```data``` folder.

**Inference**

Run the following scripts for inference.
```shell
# UltraFusion Benchmark
python inference.py --dataset UltraFusion --output results --tiled --tile_size 512 --tile_stride 256 --prealign --save_all
# RealHDRV
python inference.py --dataset RealHDRV --output results --tiled --tile_size 512 --tile_stride 256 --prealign --save_all
# MEFB (cancel pre-alignment for static scenes)
python inference.py --dataset MEFB --output results --tiled --tile_size 512 --tile_stride 256 --save_all
```
You can also use ```val_nriqa.py``` for evaluation.




## Acknowledgements
This project is developped on the codebase of [DiffBIR](https://github.com/XPixelGroup/DiffBIR). We appreciate their great work! 

## :love_you_gesture: Citation
If you find our paper and repo are helpful for your research, please consider citing:
```BibTeX
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Zixuan and Wang, Yujin and Cai, Xin and You, Zhiyuan and Lu, Zheming and Zhang, Fan and Guo, Shi and Xue, Tianfan},
    title     = {UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {16111-16121}
}
```
