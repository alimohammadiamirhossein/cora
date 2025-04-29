import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from utils import normalize
from model import freq_exp, gen_nn_map
from src.ddpm_step import deterministic_ddpm_step
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput


# Kernel sizes for the DIFT correction at successive time-ranges
DIFT_KERNELS: Tuple[int, int, int, int] = (12, 7, 5, 3)

def _get_kernel_for_timestep(timestep: int) -> Tuple[int, int]:
    if timestep >= 799:
        return DIFT_KERNELS[0], 1
    if timestep >= 599:
        return DIFT_KERNELS[1], 1
    if timestep >= 299:
        return DIFT_KERNELS[2], 1
    return DIFT_KERNELS[3], 1

def step_save_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
    noise_pred_uncond: Optional[torch.FloatTensor] = None,
    **kwargs,
):
    timestep_index = self._timesteps.index(timestep)
    next_timestep_index = timestep_index + 1
    
    u_hat_t, beta_coef = deterministic_ddpm_step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        scheduler=self,
    )

    x_t_minus_1 = self.x_ts[next_timestep_index]
    self.x_ts_c_predicted.append(u_hat_t)

    z_t = x_t_minus_1 - u_hat_t
    self.latents.append(z_t)

    z_t, _ = normalize(z_t, timestep_index, self._config.max_norm_zs)

    x_t_minus_1_predicted = u_hat_t + z_t

    if not return_dict:
        return (x_t_minus_1_predicted,)

    return DDIMSchedulerOutput(prev_sample=x_t_minus_1, pred_original_sample=None)


def step_use_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
    noise_pred_uncond: Optional[torch.FloatTensor] = None,
    **kwargs,
):
    timestep_index = self._timesteps.index(timestep)
    next_timestep_index = timestep_index + 1

    z_t = self.latents[next_timestep_index]
    _, normalize_coefficient = normalize(
        z_t,
        timestep_index,
        self._config.max_norm_zs,
    )

    x_t_hat_c_hat, beta_coef = deterministic_ddpm_step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        scheduler=self,
    )

    x_t_minus_1_exact = self.x_ts[next_timestep_index]
    x_t_minus_1_exact = x_t_minus_1_exact.expand_as(x_t_hat_c_hat)

    x_t_c_predicted: torch.Tensor = self.x_ts_c_predicted[next_timestep_index]

    x_t_c = x_t_c_predicted[0].expand_as(x_t_hat_c_hat)
    
    mask: Optional[Tensor] = kwargs.get("mask", None)
    if mask is not None and timestep > 300:
        mask = mask.to(x_t_hat_c_hat.device)
        movement_intensifier = kwargs.get("movement_intensifier", 0.0)

        if timestep > 900 and movement_intensifier > 0.0:
            latent_mask_h, *_  = freq_exp(
                x_t_hat_c_hat[1:], 
                "auto_mask", 
                None, 
                mask.unsqueeze(0),
                movement_intensifier
                )
            x_t_hat_c_hat[1:] = latent_mask_h

        x_t_hat_c_hat[-1] = x_t_hat_c_hat[-1] * mask + (1-mask) * x_t_c[-1]

    edit_prompts_num = model_output.size(0) // 2
    x_t_hat_c_indices = (
        0,
        edit_prompts_num,
    )
    edit_images_indices = (
        edit_prompts_num,
        (model_output.size(0)),
    )

    x_t_hat_c = torch.zeros_like(x_t_hat_c_hat)
    x_t_hat_c[edit_images_indices[0] : edit_images_indices[1]] = x_t_hat_c_hat[
        x_t_hat_c_indices[0] : x_t_hat_c_indices[1]
    ]

    w1 = kwargs.get("w1", 1.9)
    cross_prompt_term = x_t_hat_c_hat - x_t_hat_c
    cross_trajectory_term = x_t_hat_c - normalize_coefficient * x_t_c

    x_t_minus_1_hat_ = (
        normalize_coefficient * x_t_minus_1_exact
        + cross_trajectory_term
        + w1 * cross_prompt_term
    )

    x_t_minus_1_hat_[x_t_hat_c_indices[0] : x_t_hat_c_indices[1]] = x_t_minus_1_hat_[
        edit_images_indices[0] : edit_images_indices[1]
    ]
    
    dift_timestep = kwargs.get("dift_timestep", 700)
    
    if timestep < dift_timestep and kwargs.get("apply_dift_correction", False):
        z_t = torch.cat([z_t]*x_t_hat_c_hat.shape[0], dim=0)

        dift_features: Optional[Tensor] = kwargs.get("dift_features", None)
        dift_s, _, dift_t = dift_features.chunk(3)

        resized_src_features = F.interpolate(dift_s[0].unsqueeze(0), size=z_t.shape[-1], mode='bilinear', align_corners=False).squeeze(0)
        resized_tgt_features = F.interpolate(dift_t[0].unsqueeze(0), size=z_t.shape[-1], mode='bilinear', align_corners=False).squeeze(0)
        
        kernel_size, stride = _get_kernel_for_timestep(timestep)
        torch.cuda.empty_cache()
        
        updated_z_t = gen_nn_map(z_t[1], resized_src_features, resized_tgt_features, 
                                 kernel_size=kernel_size, stride=stride,
                                 device=z_t.device, timestep=timestep)

        alpha = 1.0
        z_t[1] = alpha * updated_z_t + (1 - alpha) * z_t[1]

        x_t_minus_1_hat = x_t_hat_c_hat + z_t * normalize_coefficient
    else:
        x_t_minus_1_hat = x_t_minus_1_hat_

    if not return_dict:
        return (x_t_minus_1_hat,)

    return DDIMSchedulerOutput(
        prev_sample=x_t_minus_1_hat,
        pred_original_sample=None,
    )


def get_ddpm_inversion_scheduler(
    scheduler,
    config,
    timesteps,
    latents,
    x_ts,
    **kwargs,
):
    def step(
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        noise_pred_uncond: Optional[torch.FloatTensor] = None,
        dift_features: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ):
        # predict and save x_t_c
        res_inv = step_save_latents(
            scheduler,
            model_output[:1, :, :, :],
            timestep,
            sample[:1, :, :, :],
            return_dict,
            noise_pred_uncond[:1, :, :, :],
            **kwargs,
        )

        res_inf = step_use_latents(
            scheduler,
            model_output[1:, :, :, :],
            timestep,
            sample[1:, :, :, :],
            return_dict,
            noise_pred_uncond[1:, :, :, :],
            dift_features=dift_features,
            **kwargs,
        )
        res = (torch.cat((res_inv[0], res_inf[0]), dim=0),)
        return res

    scheduler._timesteps = timesteps
    scheduler._config = config
    scheduler.latents = latents
    scheduler.x_ts = x_ts
    scheduler.x_ts_c_predicted = [None]
    scheduler.step = step
    return scheduler

