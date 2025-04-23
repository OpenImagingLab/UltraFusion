# Final results
CUDA_VISIBLE_DEVICES=0 python inference.py --dataset UltraFusion --output results --tiled --tile_size 512 --tile_stride 256 --save_all