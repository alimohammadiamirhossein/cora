import os
import torch
from PIL import Image
import jsonc as json

from model import (
    DirectionalAttentionControl,
    StableDiffusionXLImg2ImgPipeline,
    register_attention_editor_diffusers,
)

from utils.pipeline_utils import *
from utils import get_args, extract_mask
from src import get_ddpm_inversion_scheduler
from visualization import save_results


def run(
    image_path,
    src_prompt,
    tgt_prompt,
    masks,
    pipeline: StableDiffusionXLImg2ImgPipeline,
    args,
):
    seed = args.seed
    num_timesteps = args.timesteps
    torch.manual_seed(seed)
    generator = torch.Generator(device=SAMPLING_DEVICE).manual_seed(seed)

    timesteps, config = set_pipeline(pipeline, num_timesteps, generator, args)

    x_0_image = Image.open(image_path).convert("RGB").resize((512, 512), RESIZE_TYPE)
    x_0 = encode_image(x_0_image, pipeline, generator)
    x_ts = create_xts(
        config.noise_shift_delta,
        config.noise_timesteps,
        generator,
        pipeline.scheduler,
        timesteps,
        x_0,
    )
    x_ts = [xt.to(dtype=x_0.dtype) for xt in x_ts]
    latents = [x_ts[0]]

    if not isinstance(masks, torch.Tensor):
        mask = extract_mask(masks, 512, 512)
    else:
        mask = masks

    pipeline.scheduler = get_ddpm_inversion_scheduler(
        pipeline.scheduler,
        config,
        timesteps,
        latents,
        x_ts,
        w1=args.w1,
        dift_timestep=args.dift_timestep,
        movement_intensifier=args.movement_intensifier,
        apply_dift_correction=args.apply_dift_correction,
        mask=mask,
    )

    
    step, layer = 0, 44
    editor = DirectionalAttentionControl(
                                        step, layer, total_steps=11, 
                                        model_type="SDXL", 
                                        alpha=args.alpha, mode=args.mode, beta=1-args.beta,
                                        structural_alignment=args.structural_alignment,
                                        support_new_object=args.support_new_object
                                        )
    register_attention_editor_diffusers(pipeline, editor)

    latent = latents[0].expand(3, -1, -1, -1)
    prompt = [src_prompt, src_prompt, tgt_prompt]
    pipeline.unet.latent_store.reset()
    image = pipeline.__call__(image=latent, prompt=prompt).images
    return [x_0_image, image[0], image[2]]


if __name__ == "__main__":
    args = get_args()

    img_paths_to_prompts = json.load(open(args.prompts_file, "r"))
    eval_dataset_folder = args.eval_dataset_folder

    img_paths = [
        f"{eval_dataset_folder}/{img_name}" for img_name in img_paths_to_prompts.keys()
    ]
    pipeline = load_pipeline(args.fp16, args.cache_dir)

    sim_scores_total = 0
    os.makedirs(args.output_dir, exist_ok=True)

    images_to_plot = []
    output_dir = args.output_dir

    for i, img_path in enumerate(img_paths):
        args.img_path = img_path
        img_name = img_path.split("/")[-1]
        prompt = img_paths_to_prompts[img_name]["src_prompt"]
        edit_prompts = img_paths_to_prompts[img_name]["tgt_prompt"]
        args.alpha = img_paths_to_prompts[img_name].get("alpha", 0.7)
        args.beta = img_paths_to_prompts[img_name].get("beta", 1)
        masks = img_paths_to_prompts[img_name].get("masks", None)
        args.mask = masks
        args.source_prompt = prompt
        args.target_prompt = edit_prompts[0]

        res = run(
            img_path,
            prompt,
            edit_prompts[0],
            masks,
            pipeline=pipeline,
            args=args,
        )
        
        torch.cuda.empty_cache()
        save_results(
            args=args,
            source_prompt=prompt,
            target_prompt=edit_prompts[0],
            images=res
            )
