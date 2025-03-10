# modified from https://github.com/unity-research/IP-Adapter-Instruct/blob/main/ip_adapter/utils.py
# Copyright 2024 unity-research/IP-Adapter-Instruct
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

#It's literally just the original one with a modified proj out.
class ResamplerSD3(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        print(embedding_dim, dim)
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out_big = nn.Linear(dim, output_dim)

        # init weights to almost zero for training stability
        nn.init.uniform_(self.proj_out_big.weight, -0.01, 0.01)
        nn.init.zeros_(self.proj_out_big.bias)



        self.norm_out_big = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)
        #print(x.shape)
        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out_big(latents)
        return self.norm_out_big(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


class ResamplerSD3_Instruct(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim_image_embeds=768,
        embedding_dim_instruct_embeds=768,

        output_dim=1024,
        ff_mult=4,
        ff_mult_secondary=2,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        #self.pos_emb = nn.Embedding(max_seq_len, embedding_dim_image_embeds) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        print(embedding_dim_image_embeds, dim)
        self.proj_in = nn.Linear(embedding_dim_image_embeds, dim)
        self.proj_new_input = nn.Linear(embedding_dim_instruct_embeds, dim)
        self.proj_out_big = nn.Linear(dim, output_dim)
        self.proj_prompt_input = nn.Linear(embedding_dim_instruct_embeds, dim)

        # init weights to almost zero for training stability
        nn.init.uniform_(self.proj_out_big.weight, -0.01, 0.01)
        nn.init.zeros_(self.proj_out_big.bias)



        self.norm_out_big = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.prompt_layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult_secondary),
            ])
            for _ in range(depth)
        ])

        self.new_input_layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult_secondary),
            ])
            for _ in range(depth)
        ])
        randomness_scale=1e-4  # Scale of randomness to add

        with torch.no_grad():
            for layer in self.new_input_layers:
                ff_layer = layer[1]  # Get the FeedForward part
                last_linear = ff_layer[-1]  # Get the last Linear layer
                random_values = torch.randn_like(last_linear.weight) * randomness_scale
                last_linear.weight.copy_(random_values)
            for layer in self.prompt_layers:
                ff_layer = layer[1]  # Get the FeedForward part
                last_linear = ff_layer[-1]  # Get the last Linear layer
                random_values = torch.randn_like(last_linear.weight) * randomness_scale
                last_linear.weight.copy_(random_values)


                
    def forward(self, x, new_input,prompt_input):
                
        x_input = self.proj_in(x)
        latents = self.latents.repeat(x_input.size(0), 1, 1)
        new_input_transformed = self.proj_new_input(new_input)
        prompt_input_transformed = self.proj_prompt_input(prompt_input)

        for (orig_attn, orig_ff), (new_attn, new_ff), (prompt_attn,prompt_ff) in zip(self.layers, self.new_input_layers,self.prompt_layers):

            # Update latents from the original input path
            latents = orig_attn(x_input, latents) + latents
            latents = orig_ff(latents) + latents
            
            latents = prompt_attn(prompt_input_transformed, latents) + latents
            latents = prompt_ff(latents) + latents

            # Update latents from the new input path
            latents = new_attn(new_input_transformed, latents) + latents
            latents = new_ff(latents) + latents

        latents = self.proj_out_big(latents)
        return self.norm_out_big(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
