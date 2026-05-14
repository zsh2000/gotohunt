# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import os

import torch
from torch import nn, Tensor
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:, :, i] for i in range(3)]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:, :, i] for i in range(3)]

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel(
                [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
            ):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


"""
Following is written by GPT-4o
"""


class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Separate projection layers for query, key, and value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias=None,
        qpos=None,
        kpos=None,
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = (
            self.q_proj(query)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(key)
            .reshape(B, M, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .reshape(B, M, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias=None,
        qpos=None,
        kpos=None,
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias)

        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Compute memory-efficient attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.transpose(1, 3)
        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionRope(AttentionRope):
    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        xpos=None,
        # --- Precomputed frame selection (for 'even' strategy) ---
        frame_neighbors: Tensor = None,
        # --- Per-layer frame selection (for 'topk' / 'incvggt_max' strategy) ---
        topk_frames: int = None,
        score_method: str = "topk",
        include_self: bool = True,
        candidate_mask: Tensor = None,
        # --- Shared params for both modes ---
        num_frames: int = None,
        tokens_per_frame: int = None,
        patch_start_idx: int = 0,
        # --- Token downsampling for K/V context ---
        token_downsample: int = 1,
        token_keep_rate: float = 1.0,
        token_selection_method: str = "diverse",
        tokens_h: int = None,
        tokens_w: int = None,
        # --- Batched SDPA (single call instead of per-frame loop) ---
        batched_sdpa: bool = False,
        # --- Attention temperature (dilution probe) ---
        # tau < 1.0  => sharpen post-softmax distribution (less dilution)
        # tau == 1.0 => identical to original behaviour (uses fused SDPA)
        # tau > 1.0  => flatter distribution (sanity control)
        tau: float = 1.0,
    ) -> Tensor:
        """
        Forward pass with optional sparse frame attention.

        Three modes:
        1. frame_neighbors is set: use precomputed per-frame neighbor indices (even strategy).
        2. topk_frames is set: compute per-layer max-pooled P×P scores from Q,K,
           select top-K frames, then do sparse attention (topk strategy).
        3. Neither is set: standard full attention (original behavior).

        Args:
            x: (B, N_total, C) input tokens
            attn_bias: optional attention bias
            xpos: (B, N_total, 2) RoPE positions
            frame_neighbors: (NF, K) precomputed frame indices per query frame.
            topk_frames: If set (int), compute per-layer topk frame selection.
            include_self: For topk mode, whether to always include the query
                frame itself in the selection.
            candidate_mask: (NF, NF) boolean mask for covisibility pre-filtering.
                mask[i,j]=True means frame j is a valid candidate for query i.
            num_frames: NF (required for any frame selection mode).
            tokens_per_frame: hw (required for any frame selection mode).
            patch_start_idx: Number of register/special tokens per frame to
                exclude from topk scoring (default 0).
        """
        B, N_total, C = x.shape
        head_dim = C // self.num_heads

        qkv = (
            self.qkv(x)
            .reshape(B, N_total, 3, self.num_heads, head_dim)
            .transpose(1, 3)
        )
        # q, k, v: (B, num_heads, N_total, head_dim)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # Determine frame_neighbors for sparse attention
        if topk_frames is not None and frame_neighbors is None:
            # Per-layer topk: compute Q-K scores from this layer's Q,K
            from mapanything.models.external.pi3.layers.frame_selection import (
                compute_cosine_frame_scores,
                compute_incvggt_frame_scores,
                select_topk_from_scores,
            )

            if score_method in ("incvggt_max", "incvggt_mean"):
                scores = compute_incvggt_frame_scores(
                    q, k, num_frames, tokens_per_frame, patch_start_idx,
                    pool="max" if score_method == "incvggt_max" else "mean",
                )
            else:
                scores = compute_cosine_frame_scores(
                    q, k, num_frames, tokens_per_frame, patch_start_idx,
                    pool="mean" if score_method == "topk_mean" else "max",
                )
            frame_neighbors = select_topk_from_scores(
                scores, topk_frames, include_self, candidate_mask
            )

            # --- DEBUG: print topk frame selections per layer ---
            NF_dbg = num_frames
            sample_idxs = [0, NF_dbg // 4, NF_dbg // 2, 3 * NF_dbg // 4, NF_dbg - 1]
            sample_idxs = sorted(set(i for i in sample_idxs if 0 <= i < NF_dbg))
            lines = []
            for si in sample_idxs:
                nb = frame_neighbors[si].tolist()
                lines.append(f"  query {si:3d} -> {nb}")
            print(f"[{score_method} debug] NF={NF_dbg}, K={frame_neighbors.shape[1]}, "
                  f"selections:\n" + "\n".join(lines))
            # --- END DEBUG ---

        if frame_neighbors is not None:
            # Sparse frame attention: each query frame attends only to K selected frames.
            NF = num_frames
            hw = tokens_per_frame
            K = frame_neighbors.shape[1]

            # Reshape to frame level: (B, num_heads, NF, hw, head_dim)
            q_frames = q.reshape(B, self.num_heads, NF, hw, head_dim)
            k_frames = k.reshape(B, self.num_heads, NF, hw, head_dim)
            v_frames = v.reshape(B, self.num_heads, NF, hw, head_dim)

            # Optionally downsample K/V tokens (diverse FPS, activation, or uniform stride)
            per_frame_indices = None  # list[Tensor] when per-frame selection is used
            _per_token_attn = False   # True when per_token_activation handles attention directly
            if token_keep_rate < 1.0:
                if token_selection_method == "per_token_activation":
                    # Selection happens during attention (needs full Q@K^T);
                    # keep k_frames/v_frames at full resolution.
                    _per_token_attn = True
                    hw_ctx = hw
                elif token_selection_method == "per_frame_activation":
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_per_frame_activation,
                    )
                    per_frame_indices = select_tokens_per_frame_activation(
                        k_frames, token_keep_rate, patch_start_idx,
                    )
                    hw_ctx = per_frame_indices[0].shape[0]
                elif token_selection_method == "per_frame_diverse":
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_per_frame_diverse,
                    )
                    per_frame_indices = select_tokens_per_frame_diverse(
                        k_frames, token_keep_rate, patch_start_idx,
                    )
                    hw_ctx = per_frame_indices[0].shape[0]
                elif token_selection_method == "activation":
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_activation,
                    )
                    per_frame_indices = select_tokens_activation(
                        k_frames, token_keep_rate, patch_start_idx,
                    )
                    hw_ctx = per_frame_indices[0].shape[0]
                else:
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_diverse,
                    )
                    per_frame_indices = select_tokens_diverse(
                        k_frames, token_keep_rate, patch_start_idx,
                    )
                    hw_ctx = per_frame_indices[0].shape[0]
            elif token_downsample > 1 and tokens_h is not None and tokens_w is not None:
                t = token_downsample
                reg_indices = torch.arange(patch_start_idx, device=q.device)
                h_indices = torch.arange(0, tokens_h, t, device=q.device)
                w_indices = torch.arange(0, tokens_w, t, device=q.device)
                spatial_indices = (h_indices[:, None] * tokens_w + w_indices[None, :]).flatten() + patch_start_idx
                keep_indices = torch.cat([reg_indices, spatial_indices])
                k_frames = k_frames[:, :, :, keep_indices, :]
                v_frames = v_frames[:, :, :, keep_indices, :]
                hw_ctx = keep_indices.shape[0]
            else:
                hw_ctx = hw

            use_flash = q.dtype == torch.bfloat16

            if _per_token_attn:
                # Per-token top-k: compute full Q@K^T, select top-k keys per
                # query token per key frame, then softmax + weighted sum.
                from mapanything.models.external.pi3.layers.frame_selection import (
                    per_token_topk_attention,
                )
                out = torch.empty(B, NF, hw, self.num_heads, head_dim,
                                  device=q.device, dtype=q.dtype)
                for i in range(NF):
                    q_i = q_frames[:, :, i]       # (B, heads, hw, head_dim)
                    nb = frame_neighbors[i]        # (K,)
                    k_nb = k_frames[:, :, nb]      # (B, heads, K, hw, head_dim)
                    v_nb = v_frames[:, :, nb]      # (B, heads, K, hw, head_dim)
                    o_i = per_token_topk_attention(
                        q_i, k_nb, v_nb, token_keep_rate, patch_start_idx,
                    )
                    out[:, i] = o_i.transpose(1, 2)
                x = out.reshape(B, NF * hw, C)
            elif per_frame_indices is not None:
                # Per-frame token selection: each frame has different K/V tokens,
                # so we must use a per-frame loop.
                out = torch.empty(B, NF, hw, self.num_heads, head_dim,
                                  device=q.device, dtype=q.dtype)

                for i in range(NF):
                    q_i = q_frames[:, :, i]  # (B, num_heads, hw, head_dim)
                    nb = frame_neighbors[i]  # (K,)

                    # Gather K/V from neighbors, each with its own token indices
                    k_parts = []
                    v_parts = []
                    for n in nb:
                        idx = per_frame_indices[n]  # (hw_ctx,)
                        k_parts.append(k_frames[:, :, n, idx, :])  # (B, heads, hw_ctx, dim)
                        v_parts.append(v_frames[:, :, n, idx, :])
                    k_i = torch.cat(k_parts, dim=2)  # (B, heads, K*hw_ctx, dim)
                    v_i = torch.cat(v_parts, dim=2)

                    if use_flash:
                        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            o_i = scaled_dot_product_attention(q_i, k_i, v_i)
                    else:
                        with nn.attention.sdpa_kernel(
                            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            o_i = scaled_dot_product_attention(q_i, k_i, v_i)

                    out[:, i] = o_i.transpose(1, 2)

                x = out.reshape(B, NF * hw, C)
            elif batched_sdpa:
                # Flat SDPA path: all query tokens attend to the union of
                # selected frames' K/V in a single SDPA call.
                # Q: (B, heads, NF*hw, dim)  K/V: (B, heads, K_unique*hw_ctx, dim)
                # Much more memory-efficient than the old gather-all approach.
                unique_frames = frame_neighbors.unique().sort()[0]
                K_unique = unique_frames.shape[0]

                k_sel = k_frames[:, :, unique_frames]  # (B, heads, K_unique, hw_ctx, dim)
                v_sel = v_frames[:, :, unique_frames]
                k_flat = k_sel.reshape(B, self.num_heads, K_unique * hw_ctx, head_dim)
                v_flat = v_sel.reshape(B, self.num_heads, K_unique * hw_ctx, head_dim)

                if use_flash:
                    with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out = scaled_dot_product_attention(q, k_flat, v_flat)
                else:
                    with nn.attention.sdpa_kernel(
                        [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                    ):
                        out = scaled_dot_product_attention(q, k_flat, v_flat)

                # (B, heads, NF*hw, dim) -> (B, NF*hw, C)
                x = out.transpose(1, 2).reshape(B, N_total, C)
            else:
                # Per-frame loop: process one query frame at a time to avoid
                # materialising the full (B, heads, NF, K, hw, head_dim) gathered
                # tensor, which OOMs on limited-memory GPUs.
                out = torch.empty(B, NF, hw, self.num_heads, head_dim,
                                  device=q.device, dtype=q.dtype)

                for i in range(NF):
                    # q_i: (B, num_heads, hw, head_dim)
                    q_i = q_frames[:, :, i]

                    # Gather K and V only for the K neighbours of frame i
                    nb = frame_neighbors[i]                       # (K,)
                    k_i = k_frames[:, :, nb].reshape(B, self.num_heads, K * hw_ctx, head_dim)
                    v_i = v_frames[:, :, nb].reshape(B, self.num_heads, K * hw_ctx, head_dim)

                    if use_flash:
                        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            o_i = scaled_dot_product_attention(q_i, k_i, v_i)
                    else:
                        with nn.attention.sdpa_kernel(
                            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            o_i = scaled_dot_product_attention(q_i, k_i, v_i)

                    # o_i: (B, num_heads, hw, head_dim) -> (B, hw, num_heads, head_dim)
                    out[:, i] = o_i.transpose(1, 2)

                x = out.reshape(B, NF * hw, C)
        else:
            # Optionally downsample K/V tokens (diverse FPS, activation, or uniform stride)
            per_frame_indices_noframenb = None
            _per_token_attn_noframenb = False
            if token_keep_rate < 1.0 and num_frames is not None:
                NF = num_frames
                hw = tokens_per_frame
                head_dim = C // self.num_heads
                psi = patch_start_idx

                k_f = k.reshape(B, self.num_heads, NF, hw, head_dim)
                v_f = v.reshape(B, self.num_heads, NF, hw, head_dim)

                if token_selection_method == "per_token_activation":
                    _per_token_attn_noframenb = True
                elif token_selection_method.startswith("per_frame_"):
                    if token_selection_method == "per_frame_activation":
                        from mapanything.models.external.pi3.layers.frame_selection import (
                            select_tokens_per_frame_activation,
                        )
                        per_frame_indices_noframenb = select_tokens_per_frame_activation(k_f, token_keep_rate, psi)
                    else:
                        from mapanything.models.external.pi3.layers.frame_selection import (
                            select_tokens_per_frame_diverse,
                        )
                        per_frame_indices_noframenb = select_tokens_per_frame_diverse(k_f, token_keep_rate, psi)
                elif token_selection_method == "activation":
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_activation,
                    )
                    per_frame_indices_noframenb = select_tokens_activation(k_f, token_keep_rate, psi)
                else:
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_diverse,
                    )
                    per_frame_indices_noframenb = select_tokens_diverse(k_f, token_keep_rate, psi)
            elif token_downsample > 1 and tokens_h is not None and tokens_w is not None and num_frames is not None:
                NF = num_frames
                hw = tokens_per_frame
                head_dim = C // self.num_heads
                psi = patch_start_idx

                k_f = k.reshape(B, self.num_heads, NF, hw, head_dim)
                v_f = v.reshape(B, self.num_heads, NF, hw, head_dim)

                t = token_downsample
                reg_indices = torch.arange(psi, device=q.device)
                h_indices = torch.arange(0, tokens_h, t, device=q.device)
                w_indices = torch.arange(0, tokens_w, t, device=q.device)
                spatial_indices = (h_indices[:, None] * tokens_w + w_indices[None, :]).flatten() + psi
                keep_indices = torch.cat([reg_indices, spatial_indices])

                k = k_f[:, :, :, keep_indices, :].reshape(B, self.num_heads, NF * keep_indices.shape[0], head_dim)
                v = v_f[:, :, :, keep_indices, :].reshape(B, self.num_heads, NF * keep_indices.shape[0], head_dim)

            if _per_token_attn_noframenb:
                # Per-token top-k without frame_neighbors: loop over query frames,
                # each attending to ALL NF key frames with per-token selection.
                from mapanything.models.external.pi3.layers.frame_selection import (
                    per_token_topk_attention,
                )
                head_dim = C // self.num_heads
                q_f = q.reshape(B, self.num_heads, NF, hw, head_dim)
                out = torch.empty(B, NF, hw, self.num_heads, head_dim,
                                  device=q.device, dtype=q.dtype)
                for i in range(NF):
                    q_i = q_f[:, :, i]  # (B, heads, hw, dim)
                    o_i = per_token_topk_attention(
                        q_i, k_f, v_f, token_keep_rate, psi,
                    )
                    out[:, i] = o_i.transpose(1, 2)
                x = out.reshape(B, NF * hw, C)
                x = x.transpose(1, 2).reshape([B, N_total, C])
            else:
                if per_frame_indices_noframenb is not None:
                    # Per-frame token selection without frame_neighbors: each frame
                    # attends to ALL other frames' K/V, but each with different tokens.
                    head_dim = C // self.num_heads
                    hw_ctx = per_frame_indices_noframenb[0].shape[0]
                    # Gather per-frame K/V and concatenate across frames
                    k_parts = []
                    v_parts = []
                    for f in range(NF):
                        idx = per_frame_indices_noframenb[f]  # (hw_ctx,)
                        k_parts.append(k_f[:, :, f, idx, :])  # (B, heads, hw_ctx, dim)
                        v_parts.append(v_f[:, :, f, idx, :])
                    k = torch.cat(k_parts, dim=2)  # (B, heads, NF*hw_ctx, dim)
                    v = torch.cat(v_parts, dim=2)

                # Standard full attention (Q full resolution, K/V possibly downsampled)
                if tau != 1.0:
                    # Temperature scaling via SDPA's `scale` arg: this computes
                    # softmax((Q K^T) * (scale / tau)) V without materializing the
                    # full (N x N) attention matrix, avoiding OOM for large NF*hw.
                    scale = head_dim ** -0.5
                    if q.dtype == torch.bfloat16:
                        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            x = scaled_dot_product_attention(q, k, v, scale=scale / tau)
                    else:
                        with nn.attention.sdpa_kernel(
                            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            x = scaled_dot_product_attention(q, k, v, scale=scale / tau)
                elif q.dtype == torch.bfloat16:
                    with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        x = scaled_dot_product_attention(q, k, v)
                else:
                    with nn.attention.sdpa_kernel(
                        [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                    ):
                        x = scaled_dot_product_attention(q, k, v)

                x = x.transpose(1, 2).reshape([B, N_total, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)

    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(
        B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads
    )

    qkv = qkv.transpose(1, 3)
    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    score = (
        (
            q.permute(0, 2, 1, 3)
            * blk_class.attn.scale
            @ k.permute(0, 2, 1, 3).transpose(-2, -1)
        )
        .sum(dim=1)
        .reshape(B, frame_num, token_length, frame_num, token_length)
        .mean(dim=[2, 4])
        .sum(-1)
    )

    return score
