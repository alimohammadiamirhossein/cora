import torch
import torch.nn as nn
from einops import rearrange, repeat

from typing import Any
from model.directional_attentions import DirectionalAttentionControl, AttentionBase
from utils import find_smallest_key_with_suffix


def register_attention_editor_diffusers(model: Any, editor: AttentionBase):
    def ca_forward(self, place_in_unet):
        def forward(
            x: torch.Tensor, 
            encoder_hidden_states: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            context: torch.Tensor = None,
            mask: torch.Tensor = None
        ):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            h = self.heads
            is_cross = context is not None
            context = context if is_cross else x

            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            dift_features_dict = getattr(model.unet.latent_store, 'dift_features', {})
            dift_features_key = find_smallest_key_with_suffix(dift_features_dict, suffix='_1')
            dift_features = dift_features_dict.get(dift_features_key, None)

            attn = sim.softmax(dim=-1)
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads,
                scale=self.scale,
                dift_features=dift_features
            )

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]

            return to_out(out)
        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count
