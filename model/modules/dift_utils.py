import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.decomposition import PCA
from typing import Optional, Tuple
from PIL import Image
from model.modules.new_object_detection import *


class DIFTLatentStore:
    def __init__(self, steps: List[int], up_ft_indices: List[int]):
        self.steps = steps
        self.up_ft_indices = up_ft_indices
        self.dift_features = {}
        self.smoothed_dift_features = {}

    def __call__(self, features: torch.Tensor, t: int, layer_index: int):
        if t in self.steps and layer_index in self.up_ft_indices:
            self.dift_features[f'{int(t)}_{layer_index}'] = features
    
    def smooth(self, kernel_size=3, sigma=1):
        for key, value in self.dift_features.items():
            if key not in self.smoothed_dift_features:
                self.smoothed_dift_features[key] = torch.stack([gaussian_smooth(x, kernel_size=kernel_size, sigma=sigma) for x in value], dim=0)
                
    def copy(self):
        copy_dift = DIFTLatentStore(self.steps, self.up_ft_indices)

        for key, value in self.dift_features.items():
            copy_dift.dift_features[key] = value.clone()

        return copy_dift

    def reset(self):
        self.dift_features = {}
        self.smoothed_dift_features = {}

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                      np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = torch.Tensor(kernel / kernel.sum()).to(input_tensor.dtype).to(input_tensor.device)
    
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    smoothed_slices = []
    for i in range(input_tensor.size(0)):
        slice_tensor = input_tensor[i, :, :]
        slice_tensor = F.conv2d(slice_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)[0, 0]
        smoothed_slices.append(slice_tensor)
    
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)

    return smoothed_tensor

def cos_dist(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    res = a_norm @ b_norm.T
    return 1 - res

def extract_patches(feature_map: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    # feature_map is (C, H, W). Unfold requires (B, C, H, W).
    feature_map = feature_map.unsqueeze(0)  # (1, C, H, W)

    # Unfold: output shape will be (B, C * patch_size^2, num_patches)
    patches = F.unfold(
        feature_map,
        kernel_size=patch_size,
        stride=stride
    )
    # Now patches is (1, C*patch_size^2, num_patches)

    # Transpose to get shape (num_patches, C*patch_size^2)
    patches = patches.squeeze(0).transpose(0, 1)  # (num_patches, C*patch_size^2)
    return patches

def reassemble_patches(
    patches: torch.Tensor,
    out_shape: Tuple[int, int, int],
    patch_size: int,
    stride: int
) -> torch.Tensor:
    C, H, W = out_shape
    
    # 1) Convert from (num_patches, C*patch_size^2) to (B=1, C*patch_size^2, num_patches)
    patches_4d = patches.transpose(0, 1).unsqueeze(0)  # (1, C*patch_size^2, num_patches)
    
    # 2) fold: reassemble patches to (1, C, H, W)
    reassembled = F.fold(
        patches_4d,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride
    )
    
    # 3) Create a divisor mask to account for overlapping regions.
    #    We do this by folding a "ones" tensor of the same shape as patches_4d.
    ones_input = torch.ones_like(patches_4d)
    overlap_count = F.fold(
        ones_input,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride
    )
    
    # 4) Divide to normalize overlapping areas
    reassembled = reassembled / overlap_count.clamp_min(1e-8)
    
    # 5) Remove the batch dimension -> (C, H, W)
    reassembled = reassembled.squeeze(0)
    
    return reassembled

def calculate_patch_distance(index1: int, index2: int, grid_size: int, stride: int, patch_size: int) -> float:
    row1, col1 = index1 // grid_size, index1 % grid_size
    row2, col2 = index2 // grid_size, index2 % grid_size
    # print('row1, col1:', row1, col1)
    x_center1, y_center1 = (row1 * stride) + (patch_size / 2), (col1 * stride) + (patch_size / 2)
    x_center2, y_center2 = (row2 * stride) + (patch_size / 2), (col2 * stride) + (patch_size / 2)
    return math.sqrt((x_center2 - x_center1)**2 + (y_center2 - y_center1)**2)

def gen_nn_map(
    latent, 
    src_features, 
    tgt_features, 
    device, 
    kernel_size=3, 
    stride=1, 
    return_newness=False,
    **kwargs
    ):
    batch_size = kwargs.get("batch_size", None)
    timestep = kwargs.get("timestep", None)
    
    if kwargs.get("visualize", False):
        dift_visualization(src_features, tgt_features, filename_out=f"output/feat_colors_{timestep}.png")
        
    src_patches = extract_patches(src_features, kernel_size, stride)
    tgt_patches = extract_patches(tgt_features, kernel_size, stride)
    
    if isinstance(latent, list):
        latent_patches = [extract_patches(l, kernel_size, stride) for l in latent]
    else:
        latent_patches = extract_patches(latent, kernel_size, stride)

    num_tgt = src_patches.size(0)
    batch = batch_size or num_tgt
    nearest_neighbor_indices = torch.empty(num_tgt, dtype=torch.long, device=device)
    nearest_neighbor_distances = torch.empty(num_tgt, dtype=torch.long, device=device)
    dist_chunks = []

    for start in range(0, num_tgt, batch):
        sims = cos_dist(src_patches, tgt_patches[start : start + batch])
        dist_chunks.append(sims)
        min_distances, best_idx = sims.min(0)
        nearest_neighbor_indices[start : start + batch] = best_idx
        nearest_neighbor_distances[start : start + batch] = min_distances

    if not isinstance(latent, list):
        aligned_latent = latent_patches[nearest_neighbor_indices]
        aligned_latent = reassemble_patches(aligned_latent, latent.shape, kernel_size, stride)
    else:
        aligned_latent = [latent_patches[i][nearest_neighbor_indices] for i in range(len(latent_patches))]
        aligned_latent = [reassemble_patches(l, latent[0].shape, kernel_size, stride) for l in aligned_latent]
    
    if return_newness:
        dist_matrix = torch.cat(dist_chunks, dim=0)
        newness_method = 'two_sided'
        # newness_method = 'distance'
        if newness_method.lower() == "distance":
            newness = detect_newness_distance(nearest_neighbor_distances, quantile=0.97)
        
        elif newness_method.lower() == "two_sided":
            newness = detect_newness_two_sided(dist_matrix, k=4)

        out_shape = latent[0].shape if isinstance(latent, list) else latent.shape
        out_shape = (1, out_shape[1], out_shape[2])

        newness = reassemble_patches(newness.unsqueeze(-1), out_shape, kernel_size, stride)
    

    del src_patches, tgt_patches, latent_patches, nearest_neighbor_indices, nearest_neighbor_distances
    
    ################## visualization of changing source features to match target   ##################
    if False:
        updated_src_patches = src_patches[nearest_neighbor_indices]
        updated_src_patches = reassemble_patches(updated_src_patches, src_features.shape, kernel_size, stride)
        dift_visualization(
            updated_src_patches, tgt_features,
            filename_out=f"output/updated_feat_colors_{timestep}.png",
        )
    
    if return_newness:
        if isinstance(aligned_latent, list):
            aligned_latent.append(newness)
        else:
            return aligned_latent, newness
    return aligned_latent

def dift_visualization(
    src_feature: torch.Tensor,
    tgt_feature: torch.Tensor,
    filename_out: str,
    resize_to: Optional[Tuple[int, int]] = (512, 512)
):
    """
    Flatten features, apply PCA for 3D embedding, normalize for RGB, then reshape and save as image
    """

    C, H_s, W_s = src_feature.shape
    _, H_t, W_t = tgt_feature.shape

    src_flat = src_feature.permute(1, 2, 0).reshape(-1, C)  # (H_s*W_s, C)
    tgt_flat = tgt_feature.permute(1, 2, 0).reshape(-1, C)  # (H_t*W_t, C)

    all_features = torch.cat([src_flat, tgt_flat], dim=0)  # shape: (N_total, C)

    all_features_np = all_features.detach().cpu().numpy()

    num_components = 3
    pca = PCA(n_components=num_components)
    all_features_3d = pca.fit_transform(all_features_np)  # shape: (N_total, 3)

    # 6) Normalize each dimension to [0,1]
    def normalize_to_01(array_2d):
        min_vals = array_2d.min(axis=0)
        max_vals = array_2d.max(axis=0)
        denom = (max_vals - min_vals) + 1e-8
        return (array_2d - min_vals) / denom

    all_features_rgb = normalize_to_01(all_features_3d)

    N_src = H_s * W_s
    src_rgb_flat = all_features_rgb[:N_src]      # (N_src, 3)
    tgt_rgb_flat = all_features_rgb[N_src:]      # (N_tgt, 3)

    src_color_map = src_rgb_flat.reshape(H_s, W_s, 3)
    tgt_color_map = tgt_rgb_flat.reshape(H_t, W_t, 3)

    src_img = Image.fromarray((src_color_map * 255).astype(np.uint8))
    tgt_img = Image.fromarray((tgt_color_map * 255).astype(np.uint8))

    src_img_resized = src_img.resize(resize_to, Image.Resampling.LANCZOS)
    tgt_img_resized = tgt_img.resize(resize_to, Image.Resampling.LANCZOS)

    combined_width = resize_to[0] * 2
    combined_height = resize_to[1]
    combined_img = Image.new("RGB", (combined_width, combined_height))
    combined_img.paste(src_img_resized, (0, 0))
    combined_img.paste(tgt_img_resized, (resize_to[0], 0))

    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    combined_img.save(filename_out)

    print(f"Saved visualization to {filename_out}")

