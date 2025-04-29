import torch 
import torch.nn.functional as F
import io
import cv2
import numpy as np
from PIL import Image


def normalize(
    z_t,
    i,
    max_norm_zs,
):
    max_norm = max_norm_zs[i]
    if max_norm < 0:
        return z_t, 1

    norm = torch.norm(z_t)
    if norm < max_norm:
        return z_t, 1

    coeff = max_norm / norm
    z_t = z_t * coeff
    return z_t, coeff

def normalize2(x, dim):
    x_mean = x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    x_normalized = (x - x_mean) / x_std
    return x_normalized

def find_lambda_via_newton_batched(Qp, K_source, K_target, max_iter=50, tol=1e-7):
    dot_QpK_source = torch.einsum("bcd,bmd->bcm", Qp, K_source) # shape [B]
    dot_QpK_target = torch.einsum("bcd,bmd->bcm", Qp, K_target) # shape [B]
    X = torch.exp(dot_QpK_source)

    lmbd = torch.zeros([1], device=Qp.device, dtype=Qp.dtype) + 0.7
    for it in range(max_iter):
        y = torch.exp(lmbd * dot_QpK_target)
        Z = (X + y).sum(dim=(2), keepdim=True)
        x = X / Z
        y = y / Z
        val = (x.sum(dim=(1,2)) - y.sum(dim=(1,2))).sum()

        grad = - (dot_QpK_target * y).sum()

        if not (val.abs() > tol and grad.abs() > 1e-12):
            break

        lmbd = lmbd - val / grad
        if lmbd.item() < 0.4:
            return 0.1
        elif lmbd.item() > 0.9:
            return 0.65
        
    return lmbd.item()

def find_lambda_via_super_halley(Qp, K_source, K_target, max_iter=50, tol=1e-7):
    dot_QpK_source = torch.einsum("bcd,bmd->bcm", Qp, K_source)
    dot_QpK_target = torch.einsum("bcd,bmd->bcm", Qp, K_target)
    X = torch.exp(dot_QpK_source)

    lmbd = torch.zeros([], device=Qp.device, dtype=Qp.dtype) + 0.8

    for it in range(max_iter):
        y = torch.exp(lmbd * dot_QpK_target)

        Z = (X + y).sum(dim=2, keepdim=True)
        x = X / Z
        y = y / Z
        
        val = (x.sum(dim=(1,2)) - y.sum(dim=(1,2))).sum()

        grad = - (dot_QpK_target * y).sum()

        f2 = - (dot_QpK_target**2 * y).sum()

        if not (val.abs() > tol and grad.abs() > 1e-12):
            break

        denom = grad**2 - val * f2
        if denom.abs() < 1e-20:
            break

        update = (val * grad) / denom
        lmbd = lmbd - update

        print(f"iter={it}, λ={lmbd.item():.6f}, val={val.item():.6e}, grad={grad.item():.6e}")

    return lmbd

def find_smallest_key_with_suffix(features_dict: dict, suffix: str = "_1") -> str:
        smallest_key = None
        smallest_number = float('inf')
        for key in features_dict.keys():
            if key.endswith(suffix):
                try:
                    number = int(key.split('_')[0])
                    if number < smallest_number:
                        smallest_number = number
                        smallest_key = key
                except ValueError:
                    continue
        return smallest_key

def extract_mask(masks, original_width, original_height):
    if not masks:
        return None

    combined_mask = torch.zeros(512, 512)
    scale_x = 512 / original_width
    scale_y = 512 / original_height

    for mask in masks:
        start_x, start_y = mask["start_point"]
        end_x, end_y = mask["end_point"]

        start_x, end_x = min(start_x, end_x), max(start_x, end_x)
        start_y, end_y = min(start_y, end_y), max(start_y, end_y)

        scaled_start_x, scaled_start_y = int(start_x * scale_x), int(start_y * scale_y)
        scaled_end_x, scaled_end_y = int(end_x * scale_x), int(end_y * scale_y)
        combined_mask[scaled_start_y:scaled_end_y, scaled_start_x:scaled_end_x] += 1

    binary_mask = (combined_mask > 0).float()
    resized_mask = F.interpolate(binary_mask[None, None, :, :], size=(64, 64), mode="nearest")[0, 0]

    return resized_mask

