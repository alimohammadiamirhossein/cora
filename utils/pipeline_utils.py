
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_timesteps,
    retrieve_latents,
)

import torch
from functools import partial
from diffusers import DDPMScheduler
from model.pipeline_sdxl import StableDiffusionXLImg2ImgPipeline

SAMPLING_DEVICE = "cpu"  # "cuda"
VAE_SAMPLE = "argmax"  # "argmax" or "sample"
RESIZE_TYPE = None  # Image.LANCZOS

device = "cuda" if torch.cuda.is_available() else "cpu"

def encode_image(image, pipe, generator):
    pipe_dtype = pipe.dtype
    image = pipe.image_processor.preprocess(image)
    image = image.to(device=device, dtype=pipe.dtype)

    if pipe.vae.config.force_upcast:
        image = image.float()
        pipe.vae.to(dtype=torch.float32)

    init_latents = retrieve_latents(
        pipe.vae.encode(image), generator=generator, sample_mode=VAE_SAMPLE
    )

    if pipe.vae.config.force_upcast:
        pipe.vae.to(pipe_dtype)

    init_latents = init_latents.to(pipe_dtype)
    init_latents = pipe.vae.config.scaling_factor * init_latents

    return init_latents

def create_xts(
    noise_shift_delta,
    noise_timesteps,
    generator,
    scheduler,
    timesteps,
    x_0,
):
    if noise_timesteps is None:
        noising_delta = noise_shift_delta * (timesteps[0] - timesteps[1])
        noise_timesteps = [timestep - int(noising_delta) for timestep in timesteps]
        # noise_timesteps = [timestep for timestep in timesteps]
    
    # print(noise_timesteps, timesteps)
    first_x_0_idx = len(noise_timesteps)
    for i in range(len(noise_timesteps)):
        if noise_timesteps[i] <= 0:
            first_x_0_idx = i
            break

    noise_timesteps = noise_timesteps[:first_x_0_idx]

    x_0_expanded = x_0.expand(len(noise_timesteps), -1, -1, -1)
    noise = torch.randn(
        x_0_expanded.size(), generator=generator, device=SAMPLING_DEVICE
    ).to(x_0.device)

    x_ts = scheduler.add_noise(
        x_0_expanded,
        noise,
        torch.IntTensor(noise_timesteps),
    )
    x_ts = [t.unsqueeze(dim=0) for t in list(x_ts)]
    x_ts += [x_0] * (len(timesteps) - first_x_0_idx)
    x_ts += [x_0]
    return x_ts

def load_pipeline(fp16, cache_dir):
    kwargs = (
        {
            "torch_dtype": torch.float16,
            "variant": "fp16",
        }
        if fp16
        else {}
    )
    from model.unet_sdxl import OursUNet2DConditionModel
    unet = OursUNet2DConditionModel.from_pretrained(
        "stabilityai/sdxl-turbo", 
        subfolder="unet", 
        cache_dir=cache_dir,
        safety_checker=None,
        **kwargs,
    )
    
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        unet=unet,
        cache_dir=cache_dir,
        safety_checker=None,
        **kwargs,
    )
    
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDPMScheduler.from_pretrained(  # type: ignore
        "stabilityai/sdxl-turbo",
        subfolder="scheduler",
    )

    return pipeline

def set_pipeline(pipeline: StableDiffusionXLImg2ImgPipeline, num_timesteps, generator, config):
    if config.timesteps is None:
        denoising_start = config.step_start / config.num_steps_inversion
        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, config.num_steps_inversion, device, None
        )
        timesteps, num_inference_steps = pipeline.get_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            denoising_start=denoising_start,
            strength=0,
        )
        timesteps = timesteps.type(torch.int64)
        pipeline.__call__ = partial(
            pipeline.__call__,
            num_inference_steps=config.num_steps_inversion,
            guidance_scale=config.guidance_scale,
            generator=generator,
            denoising_start=denoising_start,
            strength=0,
        )
        pipeline.scheduler.set_timesteps(
            timesteps=timesteps.cpu(),
        )
    else:
        timesteps = torch.tensor(config.timesteps, dtype=torch.int64)
        pipeline.__call__ = partial(
            pipeline.__call__,
            timesteps=timesteps,
            guidance_scale=config.guidance_scale,
            denoising_start=0,
            strength=1,
        )
        pipeline.scheduler.set_timesteps(
            timesteps=config.timesteps,  # device=pipeline.device
        )
    timesteps = [torch.tensor(t) for t in timesteps.tolist()]
    return timesteps, config

