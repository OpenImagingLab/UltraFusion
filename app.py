# -*- coding: utf-8 -*-
import os
import sys
import gradio as gr
import numpy as np
import random
# import spaces #[uncomment to use ZeroGPU]
# from diffusers import DiffusionPipeline
import torch
from torchvision.transforms import ToTensor, ToPILImage
# import logging
# logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)


# model_name = "iimmortall/UltraFusion"
# auth_token = os.getenv("ModelAccessToken")
# from huggingface_hub import hf_hub_download, snapshot_download
# model_folder = snapshot_download(repo_id=model_name, token=auth_token, local_dir="/home/user/app")
base_path = '/home/xlab-app-center'
# download repo to the base_path directory using git
print(os.system('pwd'))
# os.system('apt install git')
# os.system('apt install git-lfs')
auth_token = os.getenv("ModelAccessToken")
# please replace "your_git_token" with real token
os.system(f'git clone https://OpenImagingLab:{auth_token}@code.openxlab.org.cn/OpenImagingLab/UltraFusionModel.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

# print(sys.path)

from ultrafusion_utils import load_model, run_ultrafusion, check_input

to_tensor = ToTensor()
to_pil = ToPILImage()
ultrafusion_pipe, flow_model = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# @spaces.GPU(duration=60) #[uncomment to use ZeroGPU]
def infer(
    under_expo_img,
    over_expo_img,
    num_inference_steps
):
    print(under_expo_img.size)
    print("reciving image")

    # under_expo_img = under_expo_img.resize([1500, 1000])
    # over_expo_img = over_expo_img.resize([1500, 1000])
    under_expo_img, over_expo_img = check_input(under_expo_img, over_expo_img, max_l=1500)

    ue = to_tensor(under_expo_img).unsqueeze(dim=0).to("cuda")
    oe = to_tensor(over_expo_img).unsqueeze(dim=0).to("cuda")

    out = run_ultrafusion(ue, oe, 'test', flow_model=flow_model, pipe=ultrafusion_pipe, steps=num_inference_steps, consistent_start=None)

    out = out.clamp(0, 1).squeeze()
    out_pil = to_pil(out)

    return out_pil


examples= [
    [os.path.join("examples", img_name, "ue.jpg"), 
     os.path.join("examples", img_name, "oe.jpg")] for img_name in sorted(os.listdir("examples"))
]
IMG_W = 320
IMG_H = 240
css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""
# max-heigh: 1500px;

_HEADER_ = r"""
<h1 style="text-align: center;"><b>UltraFusion</b></h1>

- This is an HDR algorithm that fuses two images with different exposures.

- This can fuse two images with a very large exposure difference, even up to 9 stops.

- The maximum resolution we support is 1500 x 1500. If the images you upload are larger than this, they will be downscaled while maintaining the original aspect ratio.

- The two input images should have the same resolution; otherwise, an error will be reported.

- This is only for internal testing. Do not share it publicly.
"""

_CITE_ = r"""
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xxx,
  title={xxx},
  author={xxx},
  journal={arXiv preprint arXiv:xx.xx},
  year={2024}
}
```

📋 **License**

CC BY-NC 4.0. LICENSE.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>xxx@gmail.com</b>.
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(_HEADER_)
        with gr.Row():
            under_expo_img = gr.Image(label="UnderExposureImage", show_label=True,
                image_mode="RGB",
                sources=["upload", ],
                width=IMG_W,
                height=IMG_H,
                type="pil"
            )
            over_expo_img = gr.Image(label="OverExposureImage", show_label=True, 
                image_mode="RGB",
                sources=["upload", ],
                width=IMG_W,
                height=IMG_H,
                type="pil"
            )
        with gr.Row():
            run_button = gr.Button("Run", variant="primary") # scale=0, 
        
        result = gr.Image(label="Result", show_label=True, 
                          type='pil', 
                          image_mode='RGB', 
                          format="png",
                          width=IMG_W*2,
                          height=IMG_H*2,
                        )
        with gr.Accordion("Advanced Settings", open=True):
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=2,
                maximum=50,
                step=1,
                value=20,  # Replace with defaults that work for your model
                interactive=True
            )
       
        gr.Examples(
            examples=examples, 
            inputs=[under_expo_img, over_expo_img, num_inference_steps], 
            label="Examples",
            # examples_per_page=10,
            fn=infer,
            cache_examples=True,
            outputs=[result,],
            )
        # gr.Markdown(_CITE_)
    run_button.click(fn=infer,
                     inputs=[under_expo_img, over_expo_img, num_inference_steps],
                     outputs=[result,],
                    )

if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(share=True)
    # demo.launch(server_name="0.0.0.0", debug=True, show_api=True, show_error=True, share=False)
