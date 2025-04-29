# Adapted from: https://github.com/kookie12/FlexiEdit/blob/main/flexiedit/frequency_utils.py

import torch
import torch.fft as fft
import math
import torch.nn.functional as F

''' define hyperparameters '''
# low-pass filter settings
filter_type= "gaussian" #"butterworth" 
n= 4 # gaussian parameter
# Sampling process settings
global alpha, reinversion_step, d_s, d_t, refined_step, masa_step_original, masa_step_target_branch, masa_step_retarget_branch
alpha = 0.7
d_t= 0.3
d_s= 0.3
refined_step = 0 
masa_step_original = 4
masa_step_target_branch = 51
masa_step_retarget_branch = 0

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    #mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0
    mask[..., crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask

def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def freq_2d(x, LPF, alpha):
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF

    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    x_freq_sum = x_freq

    # IFFT
    _x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(_x_freq_low, dim=(-3, -2, -1)).real
    x_low_alpha = fft.ifftn(_x_freq_low*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(_x_freq_high, dim=(-3, -2, -1)).real
    x_high_alpha = fft.ifftn(_x_freq_high*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
    
    _x_freq_low_alpha_high = fft.ifftshift(x_freq_low + x_freq_high*alpha, dim=(-3, -2, -1))
    x_low_alpha_high = fft.ifftn(_x_freq_low_alpha_high, dim=(-3, -2, -1)).real
    
    _x_freq_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high, dim=(-3, -2, -1))
    x_high_alpha_low = fft.ifftn(_x_freq_high_alpha_low, dim=(-3, -2, -1)).real

    _x_freq_alpha_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high*alpha, dim=(-3, -2, -1))
    x_alpha_high_alpha_low = fft.ifftn(_x_freq_alpha_high_alpha_low, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum, x_low_alpha, x_high_alpha, x_low_alpha_high, x_high_alpha_low, x_alpha_high_alpha_low


def freq_exp(feat, mode, user_mask, auto_mask, movement_intensifier):
    movement_intensifier = 1 - movement_intensifier
    """ Frequency manipulation for latent space. """
    feat = feat.view(4,1,64,64)
    f_shape = feat.shape # 1, 4, 64, 64
    LPF = get_freq_filter(f_shape, feat.device, filter_type, n, d_s, d_t) # d_s, d_t
    f_dtype = feat.dtype
    feat_low, feat_high, feat_sum, feat_low_alpha, feat_high_alpha, feat_low_alpha_high, feat_high_alpha_low, x_alpha_high_alpha_low = freq_2d(feat.to(torch.float64), LPF, movement_intensifier)
    feat_low = feat_low.to(f_dtype)
    feat_high = feat_high.to(f_dtype)
    feat_sum = feat_sum.to(f_dtype)
    feat_low_alpha = feat_low_alpha.to(f_dtype)
    feat_high_alpha = feat_high_alpha.to(f_dtype)
    feat_low_alpha_high = feat_low_alpha_high.to(f_dtype)
    feat_high_alpha_low = feat_high_alpha_low.to(f_dtype)

    latent_low = feat_low.view(1,4,64,64)
    
    latent_high = feat_high.view(1,4,64,64)
    
    latent_sum = feat_sum.view(1,4,64,64)
    
    latent_low_alpha_high = feat_low_alpha_high.view(1,4,64,64)
    latent_high_alpha_low = feat_high_alpha_low.view(1,4,64,64)
    
    mask = torch.zeros_like(latent_sum)
    if mode == "auto_mask":
        auto_mask = auto_mask.unsqueeze(1) # [1,64,64] => [1,1,64,64]
        mask = auto_mask.expand_as(latent_sum) # [1,1,64,64] => [1,4,64,64]
        
    elif mode == "user_mask":
        bbx_start_point, bbx_end_point = user_mask
        mask[:, :, bbx_start_point[1]//8:bbx_end_point[1]//8, bbx_start_point[0]//8:bbx_end_point[0]//8] = 1
        
    latents_shape = latent_sum.shape
    random_gaussian = torch.randn(latents_shape, device=latent_sum.device)
    
    # Apply gaussian scaling
    g_range = random_gaussian.max() - random_gaussian.min()
    l_range = latent_low_alpha_high.max() - latent_low_alpha_high.min()
    random_gaussian = random_gaussian * (l_range/g_range)

    # No scaling applied. If you wish to apply scaling to the mask, replace the following lines accordingly.
    s_range, r_range, s_range2, r_range2 = 1, 1, 1, 1
        
    latent_mask_h = latent_sum * (1 - mask) + (latent_low_alpha_high + (1-movement_intensifier)*random_gaussian) * (s_range/r_range) *mask # edit 할 부분에 high frequency가 줄어들고 가우시안 더하기
    latent_mask_l = latent_sum * (1 - mask) + (latent_high_alpha_low + (1-movement_intensifier)*random_gaussian) * (s_range2/r_range2) *mask # edit 할 부분에 low frequency가 줄어들고 가우시안 더하기
    
    return latent_mask_h, latent_mask_l, latent_sum # latent_low, latent_high, latent_sum