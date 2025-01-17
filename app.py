# -*- coding: utf-8 -*-
import os
import sys
import datetime
import gradio as gr
import numpy as np
import random
# import spaces #[uncomment to use ZeroGPU]
import torch
from torchvision.transforms import ToTensor, ToPILImage

base_path = '/home/xlab-app-center/UltraFusionModel'
# download repo to the base_path directory using git
print(os.system('pwd'))
auth_token = os.getenv("ModelAccessToken")
# please replace "your_git_token" with real token
os.system(f'git clone https://OpenImagingLab:{auth_token}@code.openxlab.org.cn/OpenImagingLab/UltraFusionModel.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
os.system(f'mv {base_path}/* ./')
print(os.system('pwd'))
print(os.system('ls'))

from ultrafusion_utils import load_model, run_ultrafusion, check_input

RUN_TIMES = 0

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
    
    under_expo_img_lr, over_expo_img_lr, under_expo_img, over_expo_img, use_bgu = check_input(under_expo_img, over_expo_img, max_l=1500)

    ue = to_tensor(under_expo_img_lr).unsqueeze(dim=0).to("cuda")
    oe = to_tensor(over_expo_img_lr).unsqueeze(dim=0).to("cuda")
    ue_hr = to_tensor(under_expo_img).unsqueeze(dim=0).to("cuda")
    oe_hr = to_tensor(over_expo_img).unsqueeze(dim=0).to("cuda")

    print("num_inference_steps:", num_inference_steps)
    try:
        if num_inference_steps is None:
            num_inference_steps = 20
        num_inference_steps = int(num_inference_steps)
    except Exception as e:
        num_inference_steps = 20

    out = run_ultrafusion(ue, oe, ue_hr, oe_hr, use_bgu, 'test', flow_model=flow_model, pipe=ultrafusion_pipe, steps=num_inference_steps, consistent_start=None)
    
    out = out.clamp(0, 1).squeeze()
    out_pil = to_pil(out)

    global RUN_TIMES
    RUN_TIMES = RUN_TIMES + 1
    print("---------------------------- Using Times---------------------------------------")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Using times: {RUN_TIMES}")

    return out_pil


def build_demo():
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

    _README_ = r"""

    - 这是一个可以融合两张不同曝光图像的 HDR 算法。

    - 它能够融合曝光差异非常大的两张图像，甚至曝光差异高达9档。
    
    - 两张输入图像应具有相同的分辨率；否则，将得到异常的结果。

    - 我们不会存储您上传的任何数据及其处理结果。

    """
    # - The maximum resolution we support is 1500 x 1500. If the images you upload are larger than this, they will be downscaled while maintaining the original aspect ratio.
    # - This is only for internal testing. Do not share it publicly.
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
            gr.Markdown("""<h1 style="text-align: center; font-size: 32px;"><b>浦像•超级HDR📸✨</b></h1>""")
            gr.Markdown("""<h1 style="text-align: center; font-size: 24px;"><b>该如何使用它呢？</b></h1>""")
            with gr.Row():
                gr.Image("ui/ch-short.png", width=IMG_W//3, show_label=False, interactive=False, show_download_button=False)
                gr.Image("ui/ch-long.png", width=IMG_W//3, show_label=False, interactive=False, show_download_button=False)
                gr.Image("ui/ch-run.png", width=IMG_W//3, show_label=False, interactive=False, show_download_button=False)
            
            with gr.Row():
                gr.Markdown("""<h1 style="text-align: center; font-size: 12px;"><b>➀ 点击拍照界面，向下拖动☀︎图标，拍摄短曝光照片。</b></h1>""")
                gr.Markdown("""<h1 style="text-align: center; font-size: 12px;"><b>➁ 点击拍照界面，向上拖动☀︎图标，拍摄长曝光照片。</b></h1>""")
                gr.Markdown("""<h1 style="text-align: center; font-size: 12px;"><b>➂ 上传拍摄的短曝光和长曝光照片，随后点击“运行”按钮，获取处理后的结果。</b></h1>""")

            gr.Markdown("""<h1 style="text-align: center; font-size: 24px;"><b>开始体验它吧!</b></h1>""")
            with gr.Row():
                under_expo_img = gr.Image(label="短爆光图", show_label=True,
                    image_mode="RGB",
                    sources=["upload", ],
                    width=IMG_W,
                    height=IMG_H,
                    type="pil"
                )
                over_expo_img = gr.Image(label="长曝光图", show_label=True, 
                    image_mode="RGB",
                    sources=["upload", ],
                    width=IMG_W,
                    height=IMG_H,
                    type="pil"
                )
            with gr.Row():
                run_button = gr.Button("运行", variant="primary") # scale=0, 
            
            result = gr.Image(label="结果", show_label=True, 
                            type='pil', 
                            image_mode='RGB', 
                            format="png",
                            width=IMG_W*2,
                            height=IMG_H*2,
                            )
            gr.Markdown(r"""<h1 style="text-align: center; font-size: 18px;"><b>喜欢上面的结果吗？点击图像上的下载按钮📥即可下载。</b></h1>""") # width="100" height="100"  <img src="ui/download.svg" alt="download"> 
            with gr.Accordion("高级设置", open=True):
                num_inference_steps = gr.Slider(
                    label="扩散模型的推理步长",
                    minimum=2,
                    maximum=50,
                    step=1,
                    value=20,  # Replace with defaults that work for your model
                    interactive=True
                )
        
            gr.Examples(
                examples=examples, 
                inputs=[under_expo_img, over_expo_img, num_inference_steps], 
                label="示例",
                # examples_per_page=10,
                fn=infer,
                cache_examples=True,
                outputs=[result,],
                )
            gr.Markdown(_README_)
            # gr.Markdown(_CITE_)
        run_button.click(fn=infer,
                        inputs=[under_expo_img, over_expo_img, num_inference_steps],
                        outputs=[result,],
                        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=10)
    demo.launch(share=True)
    # demo.launch(server_name="0.0.0.0", debug=True, show_api=True, show_error=True, share=False)
