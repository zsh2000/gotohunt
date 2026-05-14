# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.attention import SDPBackend

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self,
        x: Tensor,
        pos=None,
        # --- Precomputed frame selection (for 'even'/'diverse'/'random' strategy) ---
        frame_neighbors: Tensor = None,
        # --- Per-layer frame selection (for 'topk' / 'incvggt_max' strategy) ---
        topk_frames: int = None,
        score_method: str = "topk",
        include_self: bool = True,
        include_first: bool = False,
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
    ) -> Tensor:
        B, N, C = x.shape

        # --- Pre-projection token selection for memory efficiency ---
        # For per_frame_diverse / per_frame_activation, run FPS on the
        # pre-projection hidden states (x) so that K/V are only projected
        # for the selected tokens.  This avoids the peak-memory overlap
        # where full K/V tensors and FPS distance matrices coexist.
        _pre_token_indices = None
        _use_pre_select = (
            token_keep_rate < 1.0
            and token_selection_method in ("per_frame_diverse", "per_frame_activation")
            and num_frames is not None
            and tokens_per_frame is not None
            # RoPE is handled below by gathering positions for selected K tokens
        )
        if _use_pre_select:
            _nf = num_frames
            _hw = tokens_per_frame
            # Reshape x to match k_frames format expected by selection fns
            x_as_k = (
                x.reshape(B, _nf, _hw, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)  # (B, heads, NF, hw, head_dim)
            )
            with torch.no_grad():
                if token_selection_method == "per_frame_diverse":
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_per_frame_diverse,
                    )
                    _pre_token_indices = select_tokens_per_frame_diverse(
                        x_as_k, token_keep_rate, patch_start_idx,
                    )
                else:  # per_frame_activation
                    from mapanything.models.external.pi3.layers.frame_selection import (
                        select_tokens_per_frame_activation,
                    )
                    _pre_token_indices = select_tokens_per_frame_activation(
                        x_as_k, token_keep_rate, patch_start_idx,
                    )
            del x_as_k

        # --- QKV projection ---
        if _pre_token_indices is not None:
            # Split projection: Q for all tokens, K/V only for selected tokens.
            _nf = num_frames
            _hw = tokens_per_frame
            _hw_sel = _pre_token_indices[0].shape[0]

            W = self.qkv.weight  # (3*C, C)
            b = self.qkv.bias    # (3*C,) or None
            W_q, W_kv = W[:C], W[C:]
            b_q = b[:C] if b is not None else None
            b_kv = b[C:] if b is not None else None

            # Q: project all tokens
            q = F.linear(x, W_q, b_q)
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # K/V: gather selected tokens per frame, then project
            x_frames = x.reshape(B, _nf, _hw, C)
            idx_stack = torch.stack(_pre_token_indices)  # (NF, _hw_sel)
            idx_exp = idx_stack.unsqueeze(0).unsqueeze(-1).expand(B, _nf, _hw_sel, C)
            x_sel = torch.gather(x_frames, 2, idx_exp)  # (B, NF, _hw_sel, C)
            del x_frames

            # Gather positions for selected K tokens (for RoPE)
            pos_sel = None
            if self.rope is not None and pos is not None:
                pos_frames = pos.reshape(B, _nf, _hw, 2)
                idx_pos = idx_stack.unsqueeze(0).unsqueeze(-1).expand(B, _nf, _hw_sel, 2)
                pos_sel = torch.gather(pos_frames, 2, idx_pos).reshape(B, _nf * _hw_sel, 2)
                del pos_frames, idx_pos

            del idx_exp

            kv = F.linear(x_sel.reshape(B, _nf * _hw_sel, C), W_kv, b_kv)
            del x_sel
            kv = (
                kv.reshape(B, _nf * _hw_sel, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv.unbind(0)
            del kv

            q = self.q_norm(q)
            k = self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)          # full positions for all query tokens
                k = self.rope(k, pos_sel)      # positions of selected K tokens only
        else:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

        # Determine frame_neighbors for sparse attention
        if topk_frames is not None and frame_neighbors is None:
            from mapanything.models.external.pi3.layers.frame_selection import (
                compute_cosine_frame_scores,
                compute_incvggt_frame_scores,
                ensure_first_frame_included,
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
            if include_first:
                frame_neighbors = ensure_first_frame_included(frame_neighbors)

            # --- DEBUG: print topk frame selections per layer ---
            NF_dbg = num_frames
            sample_idxs = [0, NF_dbg // 4, NF_dbg // 2, 3 * NF_dbg // 4, NF_dbg - 1]
            sample_idxs = sorted(set(i for i in sample_idxs if 0 <= i < NF_dbg))
            lines = []
            for si in sample_idxs:
                nb = frame_neighbors[si].tolist()
                lines.append(f"  query {si:3d} -> {nb}")
            print(f"[VGGT {score_method} debug] NF={NF_dbg}, K={frame_neighbors.shape[1]}, "
                  f"selections:\n" + "\n".join(lines))
            # --- END DEBUG ---

        if frame_neighbors is not None:
            # Sparse frame attention: each query frame attends only to K selected frames.
            NF = num_frames
            hw = tokens_per_frame
            K = frame_neighbors.shape[1]
            head_dim = self.head_dim

            q_frames = q.reshape(B, self.num_heads, NF, hw, head_dim)

            # Optionally downsample K/V tokens (diverse FPS, activation, or uniform stride)
            per_frame_indices = None  # list[Tensor] when per-frame selection is used
            _per_token_attn = False   # True when per_token_activation handles attention directly

            if _pre_token_indices is not None:
                # K/V were already projected for selected tokens only.
                # k, v have shape (B, heads, NF * _hw_sel, head_dim).
                hw_ctx = _pre_token_indices[0].shape[0]
                k_frames = k.reshape(B, self.num_heads, NF, hw_ctx, head_dim)
                v_frames = v.reshape(B, self.num_heads, NF, hw_ctx, head_dim)
            else:
                k_frames = k.reshape(B, self.num_heads, NF, hw, head_dim)
                v_frames = v.reshape(B, self.num_heads, NF, hw, head_dim)

                if token_keep_rate < 1.0:
                    if token_selection_method == "per_token_activation":
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
            dropout_p = self.attn_drop.p if self.training else 0.0

            if _per_token_attn:
                # Per-token top-k: compute full Q@K^T, select top-k keys per
                # query token per key frame, then softmax + weighted sum.
                from mapanything.models.external.pi3.layers.frame_selection import (
                    per_token_topk_attention,
                )
                out = torch.empty(
                    B, NF, hw, self.num_heads, head_dim,
                    device=q.device, dtype=q.dtype,
                )
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
                out = torch.empty(
                    B, NF, hw, self.num_heads, head_dim,
                    device=q.device, dtype=q.dtype,
                )

                for i in range(NF):
                    q_i = q_frames[:, :, i]  # (B, heads, hw, head_dim)
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
                            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p)
                    else:
                        with nn.attention.sdpa_kernel(
                            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p)

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
                        out = F.scaled_dot_product_attention(q, k_flat, v_flat, dropout_p=dropout_p)
                else:
                    with nn.attention.sdpa_kernel(
                        [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                    ):
                        out = F.scaled_dot_product_attention(q, k_flat, v_flat, dropout_p=dropout_p)

                # (B, heads, NF*hw, dim) -> (B, NF*hw, C)
                x = out.transpose(1, 2).reshape(B, N, C)
            else:
                # Per-frame loop: process one query frame at a time to avoid
                # materialising the full gathered tensor, which OOMs on
                # limited-memory GPUs.
                out = torch.empty(
                    B, NF, hw, self.num_heads, head_dim,
                    device=q.device, dtype=q.dtype,
                )

                for i in range(NF):
                    q_i = q_frames[:, :, i]  # (B, heads, hw, head_dim)
                    nb = frame_neighbors[i]  # (K,)
                    k_i = k_frames[:, :, nb].reshape(B, self.num_heads, K * hw_ctx, head_dim)
                    v_i = v_frames[:, :, nb].reshape(B, self.num_heads, K * hw_ctx, head_dim)

                    if use_flash:
                        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p)
                    else:
                        with nn.attention.sdpa_kernel(
                            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p)

                    out[:, i] = o_i.transpose(1, 2)  # (B, hw, heads, head_dim)

                x = out.reshape(B, NF * hw, C)
        else:
            # Optionally downsample K/V tokens (diverse FPS, activation, or uniform stride)
            per_frame_indices_noframenb = None
            _per_token_attn_noframenb = False
            if _pre_token_indices is not None:
                # K/V already projected for selected tokens only.
                # k, v are (B, heads, NF * _hw_sel, hd) — no further selection needed.
                pass
            elif token_keep_rate < 1.0 and num_frames is not None:
                NF = num_frames
                hw = tokens_per_frame
                head_dim = self.head_dim
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
                head_dim = self.head_dim
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
                head_dim = self.head_dim
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
                x = x.transpose(1, 2).reshape(B, N, C)
            else:
                if per_frame_indices_noframenb is not None:
                    # Per-frame token selection without frame_neighbors: each frame
                    # attends to ALL other frames' K/V, but each with different tokens.
                    head_dim = self.head_dim
                    hw_ctx = per_frame_indices_noframenb[0].shape[0]
                    k_parts = []
                    v_parts = []
                    for f in range(NF):
                        idx = per_frame_indices_noframenb[f]  # (hw_ctx,)
                        k_parts.append(k_f[:, :, f, idx, :])  # (B, heads, hw_ctx, dim)
                        v_parts.append(v_f[:, :, f, idx, :])
                    k = torch.cat(k_parts, dim=2)  # (B, heads, NF*hw_ctx, dim)
                    v = torch.cat(v_parts, dim=2)

                if self.fused_attn:
                    dropout_p = self.attn_drop.p if self.training else 0.0
                    if q.dtype == torch.bfloat16:
                        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            x = F.scaled_dot_product_attention(
                                q, k, v, dropout_p=dropout_p,
                            )
                    else:
                        with nn.attention.sdpa_kernel(
                            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            x = F.scaled_dot_product_attention(
                                q, k, v, dropout_p=dropout_p,
                            )
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-2, -1)
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = attn @ v

                x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class MemEffAttention(Attention):
#     def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
#         assert pos is None
#         if not XFORMERS_AVAILABLE:
#             if attn_bias is not None:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return super().forward(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

#         q, k, v = unbind(qkv, 2)

#         x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
