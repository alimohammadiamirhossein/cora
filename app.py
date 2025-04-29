import os
import argparse
import torch
from PIL import Image
import jsonc as json
import numpy as np
import torch.nn.functional as F

from utils.pipeline_utils import *
from utils import get_args, extract_mask, remove_foreground
from src import get_ddpm_inversion_scheduler
from visualization import save_results

import gradio as gr
import spaces

from main import run

pipeline = None
def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = load_pipeline(fp16=False, cache_dir=None)
    return pipeline

def process_masks(masks):
    #masks: list of file paths
    processed_masks = []
    mask_composit = torch.zeros((512, 512), dtype=torch.float32, device='cuda')
    for mask_path in masks:
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = torch.tensor(np.array(mask), dtype=torch.float32, device='cuda')
        mask[mask > 0] = 255.0
        mask = mask / 255.0
        processed_masks.append(mask)
        mask_composit += mask
    mask_composit = torch.clamp(mask_composit, 0, 1)
    mask_composit = F.interpolate(mask_composit[None, None, :, :], size=(64, 64), mode="nearest")[0, 0]
    if mask_composit.sum() == 0:
        mask_composit = None

    return mask_composit
@spaces.GPU
def main_pipeline(
    input_image: str,
    src_prompt: str,
    tgt_prompt: str,
    alpha: float,
    beta: float,
    w1: float,
    seed: int,
    object_insertion: bool = False,
    dift_correction: bool = True,
):
    args = get_args()
    pipeline = get_pipeline()

    args.theta = alpha
    args.alpha = beta
    args.w1 = w1
    args.seed = seed
    args.structural_alignment = True
    args.support_new_object = object_insertion
    args.apply_dift_correction = dift_correction
    print(args.theta, args.alpha, args.w1, args.seed, args.support_new_object)
    torch.cuda.empty_cache()
    res_image = run(input_image['background'], src_prompt, tgt_prompt, masks=process_masks(input_image['layers']), pipeline=pipeline, args=args)[2]

    return res_image



DESCRIPTION = """# Cora
    """


with gr.Blocks(css="app/style.css") as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML(
        """<a href="https://huggingface.co/spaces/turboedit/turbo_edit?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue"""
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.ImageMask(
                label="Input image", type="filepath", height=512, width=512,brush=gr.Brush(color_mode='fixed', colors=['#555555', '#000000'])
            )
            src_prompt = gr.Text(
                label="Source Prompt",
                max_lines=1,
                placeholder="Source Prompt",
            )
            tgt_prompt = gr.Text(
                label="Target Prompt",
                max_lines=1,
                placeholder="Target Prompt",
            )
            with gr.Accordion("Advanced Options", open=False):
                seed = gr.Slider(
                    label="seed", minimum=0, maximum=16 * 1024, value=200, step=1
                )
                w1 = gr.Slider(
                    label="w", minimum=1.0, maximum=3.0, value=1.9, step=0.05
                )
                alpha = gr.Slider(
                    label="alpha", minimum=0, maximum=1, value=0, step=0.01
                )
                beta = gr.Slider(
                    label="beta", minimum=0, maximum=1, value=0, step=0.01
                )
                with gr.Row():
                    object_insertion = gr.Checkbox(
                        label="Enable object insertion",
                        value=False
                    )
                    dift_correction = gr.Checkbox(
                        label="Apply correspondence correction",
                        value=True)
            run_button = gr.Button("Edit")
        with gr.Column():
            result = gr.Image(label="Result", type="pil", height=512, width=512)

            # examples = [
            #     [
            #         "examples_demo/1.jpeg",  # input_image
            #         "a dreamy cat sleeping on a floating leaf",  # src_prompt
            #         "a dreamy bear sleeping on a floating leaf",  # tgt_prompt
            #         7,  # seed
            #         1.3,  # w1
            #     ],
            #     [
            #         "examples_demo/2.jpeg",  # input_image
            #         "A painting of a cat and a bunny surrounded by flowers",  # src_prompt
            #         "a polygonal illustration of a cat and a bunny",  # tgt_prompt
            #         2,  # seed
            #         1.5,  # w1
            #     ],
            #     [
            #         "examples_demo/3.jpg",  # input_image
            #         "a chess pawn wearing a crown",  # src_prompt
            #         "a chess pawn wearing a hat",  # tgt_prompt
            #         2,  # seed
            #         1.3,  # w1
            #     ],
            # ]
            #
            # gr.Examples(
            #     examples=examples,
            #     inputs=[
            #         input_image,
            #         src_prompt,
            #         tgt_prompt,
            #         seed,
            #         w1,
            #     ],
            #     outputs=[result],
            #     fn=main_pipeline,
            #     cache_examples=True,
            # )

    inputs = [
        input_image,
        src_prompt,
        tgt_prompt,
        alpha,
        beta,
        w1,
        seed,
        object_insertion,
        dift_correction
    ]
    outputs = [result]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)
demo.launch(share=False)

