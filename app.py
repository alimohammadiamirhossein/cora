import gradio as gr
import spaces

from PIL import Image

import numpy as np
import torch.nn.functional as F

from utils.pipeline_utils import *
from utils import get_args

from main import run

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = load_pipeline(fp16=False, cache_dir=None)
    return pipeline


def process_masks(masks):
    # masks: list of file paths
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

    args.alpha = alpha
    args.beta = beta
    args.w1 = w1
    args.seed = seed
    args.structural_alignment = True
    args.support_new_object = object_insertion
    args.apply_dift_correction = dift_correction
    torch.cuda.empty_cache()
    res_image = run(input_image['background'], src_prompt, tgt_prompt, masks=process_masks(input_image['layers']),
                    pipeline=pipeline, args=args)[2]

    return res_image


DESCRIPTION = """# Cora 🖼️🐱🦅
        ## Fast & Controllable Image Editing

        ### 🛠️ Quick start  
        1. **Upload** or drag-and-drop the image you’d like to edit.  
        2. **Source prompt** – describe what’s in the original image.  
        3. **Target prompt** – describe the result you want.  
        4. Adjust the parameters as needed.  
        5. *(Optional)* Paint a mask to specify the area to edit.  
        6. Click **Edit** and wait a few seconds for the output.

        ### ⚙️ Parameter cheat-sheet  

        | Parameter | What it does | `0` (minimum) | `1` (maximum) |
        |-----------|--------------|---------------|---------------|
        | **alpha** | Appearance transfer control | preserve source appearance | target prompt affects appearance |
        | **beta**  | Structural change control | preserve original structure | full layout change |
        | **w**     | Prompt strength    | subtle tweaks | strong changes |
        | **Seed**  | Fixes randomness for reproducibility | – | – |
        | **Enable object insertion** | Turn on when adding new objects | – | – |
        | **Apply correspondence correction** | Uses correspondence-aware latent fix | – | – |

        ### 📜 Tips  
        - To replicate **TurboEdit**, set **alpha = 1**, **beta = 1**, and turn **off** *Enable object insertion* and *Apply correspondence correction*.  
        - To test reconstruction quality of the inversion, use identical source & target prompts with **alpha = 1** and **beta = 1**.
        #### 🙏 Acknowledgements  
        The demo template is largely adapted from **[TurboEdit on Hugging Face Spaces](https://huggingface.co/spaces/turboedit/turbo_edit)**.  
    """

with gr.Blocks(css="app/style.css") as demo:
    gr.HTML(
        """<a href="https://huggingface.co/spaces/armikaeili/cora?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue"""
    )
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            input_image = gr.ImageMask(
                label="Input image", type="filepath", height=512, width=512, brush=gr.Brush(color_mode='defaults')
            )
            result = gr.Image(label="Result", type="pil", height=512, width=512)
        with gr.Column():
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
                    label="beta", minimum=0, maximum=1, value=0.04, step=0.01
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

            examples = [
                [
                    "dataset/white_cat.png",  # input_image
                    "a cat",  # src_prompt
                    "a cat wearing a suit",  # tgt_prompt
                    0.1,  # alpha
                    0.04,  # beta
                    1.9,  # w1
                    7,  # seed
                    True,  # object_insertion
                    True  # dift_correction
                ],
                [
                    "dataset/bear.png",  # input_image
                    "a sitting brown bear",  # src_prompt
                    "a roaring blue bear",  # tgt_prompt
                    0.7,  # alpha
                    0.04,  # beta
                    1.9,  # w1
                    7,  # seed
                    False,  # object_insertion
                    True  # dift_correction
                ],
                [
                    "dataset/gcat.jpg",  # input_image
                    "a cat",  # src_prompt
                    "an eagle",  # tgt_prompt
                    0.7,  # alpha
                    0.3,  # beta
                    1.9,  # w1
                    7,  # seed
                    False,  # object_insertion
                    True  # dift_correction
                ],
                [
                    "dataset/dog.png",  # input_image
                    "a photo of a dog",  # src_prompt
                    "a photo of a dog lying",  # tgt_prompt
                    0.0,  # alpha
                    1,  # beta
                    1.9,  # w1
                    7,  # seed
                    False,  # object_insertion
                    True  # dift_correction
                ],

            ]

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
            #
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                fn=main_pipeline,
                cache_examples=False,
            )

    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)
demo.queue(max_size=50).launch(share=False)