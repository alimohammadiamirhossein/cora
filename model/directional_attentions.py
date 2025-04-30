import torch
import torch.nn.functional as F
import ctypes
import numpy as np

from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from typing import Optional, Union, Tuple, List, Callable, Dict


from model.modules.dift_utils import gen_nn_map


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sim: torch.Tensor,
        attn: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()

        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sim: torch.Tensor,
        attn: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

class DirectionalAttentionControl(AttentionBase):
    MODEL_TYPE = {"SD": 16, "SDXL": 70}

    def __init__(
        self,
        start_step: int = 4,
        start_layer: int = 10,
        layer_idx: Optional[List[int]] = None,
        step_idx: Optional[List[int]] = None,
        total_steps: int = 50,
        model_type: str = "SD",
        **kwargs
    ):
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))

        self.w = 1.0
        self.structural_alignment = kwargs.get("structural_alignment", False)
        self.style_transfer_only = kwargs.get("style_transfer_only", False)
        self.alpha = kwargs.get("alpha", 0.5)
        self.beta = kwargs.get("beta", 0.5)
        self.newness = kwargs.get("support_new_object", True)
        self.mode = kwargs.get("mode", "normal")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sim: torch.Tensor,
        attn: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        q_s, q_middle, q_t = q.chunk(3)
        k_s, k_middle, k_t = k.chunk(3)
        v_s, v_middle, v_t = v.chunk(3)
        attn_s, attn_middle, attn_t = attn.chunk(3)

        out_s = self.attn_batch(q_s, k_s, v_s, sim, attn_s, is_cross, place_in_unet, num_heads, **kwargs)
        out_middle = self.attn_batch(q_middle, k_middle, v_middle, sim, attn_middle, is_cross, place_in_unet, num_heads, **kwargs)
        
        if self.cur_step <= 0 and self.beta > 0 and \
                self.structural_alignment:
            q_t = self.align_queries_via_matching(q_s, q_t, beta=self.beta)
        
        out_t = self.apply_mode(q_t, k_s, k_t, v_s, v_t, attn_t, sim, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_s, out_middle, out_t], dim=0)
        return out

    def apply_mode(
        self,
        q_t: torch.Tensor,
        k_s: torch.Tensor,
        k_t: torch.Tensor,
        v_s: torch.Tensor,
        v_t: torch.Tensor,
        attn_t: torch.Tensor,
        sim: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        mode = self.mode

        if 'dift' in mode and self.cur_step <= 0:
            mode = 'normal'

        if mode == "concat":
            out_t = self.attn_batch(
                q_t, torch.cat([k_s, 0.85 * k_t]), torch.cat([v_s, v_t]),
                sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs
            )

        elif mode == "concat_dift":
            updated_k_s, updated_v_s, _ = self.process_dift_features(kwargs.get("dift_features"), k_s, k_t, v_s, v_t)
            out_t = self.attn_batch(
                q_t, torch.cat([updated_k_s, k_t]), torch.cat([updated_v_s, v_t]),
                sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs
            )

        elif mode == "masa":
            out_t = self.attn_batch(q_t, k_s, v_s, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        elif mode == "normal":
            out_t = self.attn_batch(q_t, k_t, v_t, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        elif mode == "lerp":
            time = self.alpha
            k_lerp = k_s + time * (k_t - k_s)
            v_lerp = v_s + time * (v_t - v_s)
            out_t = self.attn_batch(q_t, k_lerp, v_lerp, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        elif mode == "lerp_dift":
            updated_k_s, updated_v_s, newness = self.process_dift_features(
                kwargs.get("dift_features"), k_s, k_t, v_s, v_t, return_newness=self.newness
            )
            out_t = self.apply_lerp_dift(q_t, k_s, k_t, v_s, v_t, updated_k_s, updated_v_s, newness, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        elif mode in ("slerp", "log_slerp"):
            time = self.alpha
            k_slerp = self.slerp_fixed_length_batch(k_s, k_t, t=time)
            v_slerp = self.slerp_batch(v_s, v_t, t=time, log_slerp="log" in mode)
            out_t = self.attn_batch(q_t, k_slerp, v_slerp, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        elif mode in ("slerp_dift", "log_slerp_dift"):
            out_t = self.apply_slerp_dift(q_t, k_s, k_t, v_s, v_t, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        else:
            out_t = self.attn_batch(q_t, k_t, v_t, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)

        return out_t

    def attn_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sim: torch.Tensor,
        attn: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        scale = kwargs.get("scale", 1.0)
        sim_batched = torch.einsum("h i d, h j d -> h i j", q, k) * scale
        attn_batched = sim_batched.softmax(-1)

        out = torch.einsum("h i j, h j d -> h i d", attn_batched, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def slerp(self, x: torch.Tensor, y: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        x_norm = x.norm(p=2)
        y_norm = y.norm(p=2)
        if y_norm < 1e-12:
            return x

        y_normalized = y / y_norm
        y_same_length = y_normalized * x_norm
        dot_xy = (x * y_same_length).sum()
        cos_theta = torch.clamp(dot_xy / (x_norm * x_norm), -1.0, 1.0)
        theta = torch.acos(cos_theta)
        if torch.isclose(theta, torch.tensor(0.0)):
            return x

        sin_theta = torch.sin(theta)
        s1 = torch.sin((1.0 - t) * theta) / sin_theta
        s2 = torch.sin(t * theta) / sin_theta
        return s1 * x + s2 * y_same_length

    def slerp_batch(
        self, 
        x: torch.Tensor,
        y: torch.Tensor,
        t: float = 0.5,
        eps: float = 1e-12,
        log_slerp: bool = False
    ) -> torch.Tensor:
        """
        Variation of SLERP for batches that allows for linear or logarithmic interpolation of magnitudes.
        """
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        y_norm = y.norm(p=2, dim=-1, keepdim=True)
        y_zero_mask = (y_norm < eps)

        x_unit = x / (x_norm + eps)
        y_unit = y / (y_norm + eps)
        dot_xy = (x_unit * y_unit).sum(dim=-1, keepdim=True)
        cos_theta = torch.clamp(dot_xy, -1.0, 1.0)

        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        theta_zero_mask = (theta.abs() < 1e-7)

        sin_theta_safe = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)
        s1 = torch.sin((1.0 - t) * theta) / sin_theta_safe
        s2 = torch.sin(t * theta) / sin_theta_safe
        dir_interp = s1 * x_unit + s2 * y_unit

        if not log_slerp:
            mag_interp = (1.0 - t) * x_norm + t * y_norm
        else:
            mag_interp = (x_norm ** (1.0 - t)) * (y_norm ** t)

        out = mag_interp * dir_interp
        out = torch.where(y_zero_mask | theta_zero_mask, x, out)
        return out


    def slerp_fixed_length_batch(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        t: float = 0.5, 
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        performing SLERP while preserving the norm of the source tensor x
        """
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        y_norm = y.norm(p=2, dim=-1, keepdim=True)
        y_zero_mask = (y_norm < eps)
        y_normalized = y / (y_norm + eps)
        y_same_length = y_normalized * x_norm
        dot_xy = (x * y_same_length).sum(dim=-1, keepdim=True)
        cos_theta = torch.clamp(dot_xy / (x_norm * x_norm + eps), -1.0, 1.0)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)

        sin_theta_safe = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)
        s1 = torch.sin((1.0 - t) * theta) / sin_theta_safe
        s2 = torch.sin(t * theta) / sin_theta_safe
        out = s1 * x + s2 * y_same_length
        theta_zero_mask = (theta.abs() < 1e-7)

        out = torch.where(y_zero_mask | theta_zero_mask, x, out)
        return out

    def apply_lerp_dift(
        self,
        q_t: torch.Tensor,
        k_s: torch.Tensor,
        k_t: torch.Tensor,
        v_s: torch.Tensor,
        v_t: torch.Tensor,
        updated_k_s: torch.Tensor,
        updated_v_s: torch.Tensor,
        newness: torch.Tensor,
        sim: torch.Tensor,
        attn_t: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        alpha = self.alpha
        k_lerp = k_s + alpha * (k_t - k_s)
        v_lerp = v_s + alpha * (v_t - v_s)
        if alpha > 0:
            k_t_new = newness * k_t + (1 - newness) * k_lerp
            v_t_new = newness * v_t + (1 - newness) * v_lerp
        else:
            k_t_new = k_s
            v_t_new = v_s

        out_t = self.attn_batch(q_t, k_t_new, v_t_new, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)
        return out_t

    def apply_slerp_dift(
        self,
        q_t: torch.Tensor,
        k_s: torch.Tensor,
        k_t: torch.Tensor,
        v_s: torch.Tensor,
        v_t: torch.Tensor,
        sim: torch.Tensor,
        attn_t: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        **kwargs
    ) -> torch.Tensor:
        updated_k_s, updated_v_s, newness = self.process_dift_features(
            kwargs.get("dift_features"), k_s, k_t, v_s, v_t, return_newness=self.newness
        )
        alpha = self.alpha
        log_slerp = "log" in self.mode

        # Interpolate from k_t->updated_k_s so that if alpha=0, we get k_t
        k_slerp = self.slerp_fixed_length_batch(k_t, updated_k_s, t=1-alpha)
        v_slerp = self.slerp_batch(v_t, updated_v_s, t=1-alpha, log_slerp=log_slerp)

        if alpha > 0:
            k_t_new = newness * k_t + (1 - newness) * k_slerp
            v_t_new = newness * v_t + (1 - newness) * v_slerp
        else:
            k_t_new = k_s
            v_t_new = v_s

        out_t = self.attn_batch(q_t, k_t_new, v_t_new, sim, attn_t, is_cross, place_in_unet, num_heads, **kwargs)
        return out_t

    def process_dift_features(
        self,
        dift_features: torch.Tensor,
        k_s: torch.Tensor,
        k_t: torch.Tensor,
        v_s: torch.Tensor,
        v_t: torch.Tensor,
        return_newness: bool = True
    ):
        dift_s, _, dift_t = dift_features.chunk(3)
        k_s1 = k_s.permute(0, 2, 1).reshape(k_s.shape[0], k_s.shape[2], int(k_s.shape[1]**0.5), -1)
        v_s1 = v_s.permute(0, 2, 1).reshape(v_s.shape[0], v_s.shape[2], int(v_s.shape[1]**0.5), -1)

        k_s1 = k_s1.reshape(-1, k_s1.shape[-2], k_s1.shape[-1])
        v_s1 = v_s1.reshape(-1, v_s1.shape[-2], v_s1.shape[-1])

        ################# uncomment only for visualization #################
        # result = gen_nn_map(
        #     [dift_s[0], dift_s[0]],
        #     dift_s[0],
        #     dift_t[0],
        #     kernel_size=1,
        #     stride=1,
        #     device=k_s.device,
        #     timestep=self.cur_step,
        #     visualize=True,
        #     return_newness=return_newness
        # )
        #####################################################################
        
        resized_src = F.interpolate(dift_s[0].unsqueeze(0), size=k_s1.shape[-1], mode='bilinear', align_corners=False).squeeze(0)
        resized_tgt = F.interpolate(dift_t[0].unsqueeze(0), size=k_s1.shape[-1], mode='bilinear', align_corners=False).squeeze(0)

        result = gen_nn_map(
            [k_s1, v_s1],
            resized_src,
            resized_tgt,
            kernel_size=1,
            stride=1,
            device=k_s.device,
            timestep=self.cur_step,
            return_newness=return_newness
        )

        if return_newness:
            updated_k_s, updated_v_s, newness = result
        else:
            updated_k_s, updated_v_s = result
            newness = torch.zeros_like(updated_k_s[:1]).to(k_s.device)

        newness = newness.view(-1).unsqueeze(0).unsqueeze(-1)
        updated_k_s = updated_k_s.reshape(k_s.shape[0], k_s.shape[2], -1).permute(0, 2, 1)
        updated_v_s = updated_v_s.reshape(v_s.shape[0], v_s.shape[2], -1).permute(0, 2, 1)

        return updated_k_s, updated_v_s, newness
    
    def sinkhorn(self, cost_matrix, max_iter=50, epsilon=1e-8):
        n, m = cost_matrix.shape
        K = torch.exp(-cost_matrix / cost_matrix.std())  # Kernelized cost matrix
        u = torch.ones(n, device=cost_matrix.device) / n
        v = torch.ones(m, device=cost_matrix.device) / m
        
        for _ in range(max_iter):
            u_prev = u.clone()
            u = 1.0 / (K @ v)
            v = 1.0 / (K.T @ u)
            if torch.max(torch.abs(u - u_prev)) < epsilon:
                break
        
        P = torch.diag(u) @ K @ torch.diag(v)
        return P

    def align_queries_via_matching(self, q_s: torch.Tensor, q_t: torch.Tensor, beta: float = 0.5, device: str = "cuda"):
        q_s = q_s.to(device)
        q_t = q_t.to(device)

        B, _, _ = q_s.shape
        q_t_updated = torch.zeros_like(q_t, device=device)

        for b in range(B):
            ########################### L2 ##############################
            # cost_matrix1 = (q_s[b].unsqueeze(1) - q_t[b].unsqueeze(0)).pow(2).sum(dim=-1)
            ######################### cosine ############################
            cost_matrix1 = - F.cosine_similarity(
                            q_s[b].unsqueeze(1), q_t[b].unsqueeze(0), dim=-1)
            #############################################################
            # cost_matrix2 = (q_t[b].unsqueeze(1) - q_t[b].unsqueeze(0)).pow(2).sum(dim=-1)
            cost_matrix2 = torch.abs(torch.arange(q_t[b].shape[0], device=device).unsqueeze(0) - 
                         torch.arange(q_t[b].shape[0], device=device).unsqueeze(1)).float()
            cost_matrix2 = cost_matrix2 ** 0.5
            # cost_matrix2 = torch.where(cost_matrix2 > 0, 1.0, 0.0)
            
            mean1 = cost_matrix1.mean()
            std1  = cost_matrix1.std()
            mean2 = cost_matrix2.mean()
            std2  = cost_matrix2.std()
            cost_func_1_std = (cost_matrix1 - mean1) / (std1 + 1e-8)
            cost_func_2_std = (cost_matrix2 - mean2) / (std2 + 1e-8)

            cost_matrix = beta * cost_func_1_std + (1.0 - beta) * cost_func_2_std
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            q_t_updated[b] = q_t[b][col_ind]

            # P = self.sinkhorn(cost_matrix)
            # col_ind = P.argmax(dim=1)
            # idea 1
            # q_t_updated[b] = q_t[b][col_ind]
            # idea 2
            # q_t_updated[b] = P @ q_t[b]

        return q_t_updated

    