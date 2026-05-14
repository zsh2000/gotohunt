# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mapanything.models.external.dinov2.hub.backbones import (
    dinov2_vits14,
    dinov2_vitb14,
    dinov2_vitl14,
    dinov2_vitg14,
    dinov2_vits14_reg,
    dinov2_vitb14_reg,
    dinov2_vitl14_reg,
    dinov2_vitg14_reg,
)
from mapanything.models.external.vggt.layers import PatchEmbed
from mapanything.models.external.vggt.layers.block import Block
from mapanything.models.external.vggt.layers.rope import (
    PositionGetter,
    RotaryPositionEmbedding2D,
)

logger = logging.getLogger(__name__)

_LOCAL_DINOV2_MODELS = {
    "dinov2_vits14": dinov2_vits14,
    "dinov2_vitb14": dinov2_vitb14,
    "dinov2_vitl14": dinov2_vitl14,
    "dinov2_vitg14": dinov2_vitg14,
    "dinov2_vits14_reg": dinov2_vits14_reg,
    "dinov2_vitb14_reg": dinov2_vitb14_reg,
    "dinov2_vitl14_reg": dinov2_vitl14_reg,
    "dinov2_vitg14_reg": dinov2_vitg14_reg,
}

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, embed_dim)
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            ### From original VGGT codebase: Doesn't load pre-trained DINOv2 weights
            # vit_models = {
            #     "dinov2_vitl14_reg": vit_large,
            #     "dinov2_vitb14_reg": vit_base,
            #     "dinov2_vits14_reg": vit_small,
            #     "dinov2_vitg2_reg": vit_giant2,
            # }

            # self.patch_embed = vit_models[patch_embed](
            #     img_size=img_size,
            #     patch_size=patch_size,
            #     num_register_tokens=num_register_tokens,
            #     interpolate_antialias=interpolate_antialias,
            #     interpolate_offset=interpolate_offset,
            #     block_chunks=block_chunks,
            #     init_values=init_values,
            # )

            ### Use local DINOv2 with FlashAttention and gradient checkpointing
            if patch_embed not in _LOCAL_DINOV2_MODELS:
                raise ValueError(
                    f"Unknown DINOv2 model: {patch_embed}. "
                    f"Available: {list(_LOCAL_DINOV2_MODELS.keys())}"
                )
            self.patch_embed = _LOCAL_DINOV2_MODELS[patch_embed](pretrained=True)
            for i in range(len(self.patch_embed.blocks)):
                self.patch_embed.blocks[i] = (
                    self.wrap_module_with_gradient_checkpointing(
                        self.patch_embed.blocks[i]
                    )
                )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    ### Gradient Checkpointing Wrapper from UniCeption:
    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        """
        Wrapper for Gradient Checkpointing
        References: https://github.com/microsoft/MoGe
        """

        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__

            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

        module.__class__ = _CheckpointingWrapper
        return module

    def forward(
        self,
        images: torch.Tensor,
        retain_indices: set = None,
        frame_selector=None,
        token_downsample: int = 1,
        token_keep_rate: float = 1.0,
        token_selection_method: str = "diverse",
        ds_keep_register: bool = False,
        backbone_minibatch_size: int = 0,
        global_as_frame_layers: set = None,
        global_as_meanpool_layers: set = None,
        layer_config: dict = None,
        adaptive_ds_threshold: float = None,
        adaptive_frame_threshold: float = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            retain_indices (set, optional): Set of block output indices to retain.
                If provided, only these indices will have actual tensors in the output list;
                all other positions will be None. This saves significant memory when heads
                only need a subset of intermediate outputs (e.g., {4, 11, 17, 23}).
                If None, all outputs are retained (original behavior).
            frame_selector: Optional FrameSelector for sparse cross-frame attention.
                If None, uses full attention (original behavior).

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        if backbone_minibatch_size > 0 and B * S > backbone_minibatch_size:
            chunks = images.split(backbone_minibatch_size, dim=0)
            outs = []
            for chunk in chunks:
                out = self.patch_embed.forward_features(chunk)
                if isinstance(out, dict):
                    out = out["x_norm_patchtokens"]
                outs.append(out)
            patch_tokens = torch.cat(outs, dim=0)
        else:
            patch_tokens = self.patch_embed.forward_features(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2)
                .to(images.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # Prepare frame selection parameters for global attention blocks
        fs_kwargs = {}
        if frame_selector is not None and S > 1:
            from mapanything.models.external.pi3.layers.frame_selection import (
                compute_covisibility_candidate_mask,
                ensure_first_frame_included,
                select_frames_closest,
                select_frames_even,
                select_frames_diverse,
                select_frames_diverse_self,
                select_frames_random,
            )

            # Compute covisibility candidate mask (shared across all layers)
            candidate_mask = None
            if frame_selector.use_covisibility:
                candidate_mask = compute_covisibility_candidate_mask(
                    frame_selector.covisibility_matrix,
                    frame_selector.covisibility_percentile,
                    device=images.device,
                )
                if candidate_mask.shape[0] != S:
                    if candidate_mask.shape[0] > S:
                        candidate_mask = candidate_mask[:S, :S]
                    else:
                        padded = torch.ones(S, S, device=images.device, dtype=torch.bool)
                        n = candidate_mask.shape[0]
                        padded[:n, :n] = candidate_mask
                        candidate_mask = padded

            tokens_h = H // self.patch_size
            tokens_w = W // self.patch_size
            token_downsample = frame_selector.token_downsample
            token_keep_rate = frame_selector.token_keep_rate
            token_selection_method = frame_selector.token_selection_method
            token_ds_layers = frame_selector.token_ds_layers
            psi = self.patch_start_idx if frame_selector.ds_keep_register else 0
            batched_sdpa = frame_selector.batched_sdpa

            if frame_selector.strategy in ("topk", "topk_mean", "incvggt_max", "incvggt_mean"):
                # Per-layer top-K. 'topk' / 'topk_mean' = cosine-sim with
                # max/mean pool. 'incvggt_max' / 'incvggt_mean' = raw scaled
                # Q@K^T reduced over tokens/heads with max or mean pooling.
                fs_kwargs = dict(
                    topk_frames=frame_selector.top_k,
                    score_method=frame_selector.strategy,
                    include_self=frame_selector.include_self,
                    include_first=frame_selector.include_first,
                    candidate_mask=candidate_mask,
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "even":
                frame_neighbors = select_frames_even(
                    num_frames=S,
                    top_k=frame_selector.top_k,
                    device=images.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                if frame_selector.include_first:
                    frame_neighbors = ensure_first_frame_included(frame_neighbors)
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "diverse":
                import time as _time
                _t0 = _time.perf_counter()
                frame_neighbors = select_frames_diverse(
                    covisibility_matrix=frame_selector.covisibility_matrix,
                    top_k=frame_selector.top_k,
                    device=images.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                if frame_selector.include_first:
                    frame_neighbors = ensure_first_frame_included(frame_neighbors)
                _t1 = _time.perf_counter()
                print(f"[TIMER] select_frames_diverse (VGGT aggregator): {_t1 - _t0:.4f}s")
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "diverse_self":
                import time as _time
                _t0 = _time.perf_counter()
                frame_neighbors = select_frames_diverse_self(
                    covisibility_matrix=frame_selector.covisibility_matrix,
                    top_k=frame_selector.top_k,
                    device=images.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                if frame_selector.include_first:
                    frame_neighbors = ensure_first_frame_included(frame_neighbors)
                _t1 = _time.perf_counter()
                print(f"[TIMER] select_frames_diverse_self (VGGT aggregator): {_t1 - _t0:.4f}s")
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "random":
                frame_neighbors = select_frames_random(
                    num_frames=S,
                    top_k=frame_selector.top_k,
                    device=images.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                if frame_selector.include_first:
                    frame_neighbors = ensure_first_frame_included(frame_neighbors)
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "closest":
                # K closest-in-index frames per query (K/2 before + K/2 after,
                # compensating from the other side when one side is short).
                frame_neighbors = select_frames_closest(
                    num_frames=S,
                    top_k=frame_selector.top_k,
                    device=images.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                if frame_selector.include_first:
                    frame_neighbors = ensure_first_frame_included(frame_neighbors)
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )

            # --- DEBUG: print frame selections ---
            if frame_selector.strategy in ("topk", "topk_mean", "incvggt_max", "incvggt_mean"):
                print(f"[VGGT {frame_selector.strategy} debug] NF={S}, K={frame_selector.top_k}, "
                      f"selections computed per-layer")
            elif frame_selector.strategy in ("even", "diverse", "diverse_self", "random", "closest"):
                K_dbg = frame_neighbors.shape[1]
                sample_idxs = sorted(set(
                    i for i in [0, S // 4, S // 2, 3 * S // 4, S - 1]
                    if 0 <= i < S
                ))
                lines = []
                for si in sample_idxs:
                    nb = frame_neighbors[si].tolist()
                    lines.append(f"  query {si:3d} -> {nb}")
                print(f"[VGGT {frame_selector.strategy} debug] NF={S}, K={K_dbg}, "
                      f"selections (same for all layers):\n" + "\n".join(lines))
            # --- END DEBUG ---

        # Standalone token_downsample / token_keep_rate without frame selection:
        # pass spatial info so the attention layer can downsample K/V globally.
        if not fs_kwargs and (token_downsample > 1 or token_keep_rate < 1.0) and S > 1:
            tokens_h = H // self.patch_size
            tokens_w = W // self.patch_size
            psi = self.patch_start_idx if ds_keep_register else 0
            fs_kwargs = dict(
                num_frames=S,
                tokens_per_frame=P,
                patch_start_idx=psi,
                token_downsample=token_downsample,
                token_keep_rate=token_keep_rate,
                token_selection_method=token_selection_method,
                tokens_h=tokens_h,
                tokens_w=tokens_w,
            )

        # Always pass num_frames / tokens_per_frame so downstream hooks
        # (e.g. attention diagnostics) can identify global-attention layers.
        if not fs_kwargs and S > 1:
            fs_kwargs = dict(
                num_frames=S,
                tokens_per_frame=P,
            )

        frame_idx = 0
        global_idx = 0
        output_list = []
        out_idx = 0

        _token_ds_layers = None
        if frame_selector is not None and hasattr(frame_selector, 'token_ds_layers'):
            _token_ds_layers = frame_selector.token_ds_layers

        _token_ds_entropy_threshold = None
        if frame_selector is not None and hasattr(frame_selector, 'token_ds_entropy_threshold'):
            _token_ds_entropy_threshold = frame_selector.token_ds_entropy_threshold

        _global_as_frame = global_as_frame_layers  # from direct kwarg
        if _global_as_frame is None and frame_selector is not None and hasattr(frame_selector, 'global_as_frame_layers'):
            _global_as_frame = frame_selector.global_as_frame_layers

        _global_as_meanpool = global_as_meanpool_layers  # from direct kwarg
        if _global_as_meanpool is None and frame_selector is not None and hasattr(frame_selector, 'global_as_meanpool_layers'):
            _global_as_meanpool = frame_selector.global_as_meanpool_layers

        # Merge layer_config into routing sets and per-layer overrides
        _layer_overrides = {}  # {layer_idx: {token_keep_rate, token_selection_method, ...}}
        if layer_config is not None:
            if _global_as_frame is None:
                _global_as_frame = set()
            if _global_as_meanpool is None:
                _global_as_meanpool = set()
            for idx, cfg in layer_config.items():
                strategy = cfg.get("strategy", "global")
                if strategy == "meanpool":
                    _global_as_meanpool.add(idx)
                    _global_as_frame.discard(idx)
                elif strategy == "frame":
                    _global_as_frame.add(idx)
                    _global_as_meanpool.discard(idx)
                # Collect per-layer overrides (token_keep_rate, token_selection_method, etc.)
                overrides = {k: v for k, v in cfg.items() if k != "strategy"}
                if overrides:
                    _layer_overrides[idx] = overrides

        # Adaptive entropy-based routing state machine.
        # States: "frame" -> "ds" -> "full" (one-directional transitions).
        # - "frame": Ent/Max >= adaptive_frame_threshold -> global_as_frame
        # - "ds": Ent/Max >= adaptive_ds_threshold -> token downsampling
        # - "full": Ent/Max < adaptive_ds_threshold -> full attention (terminal)
        _use_adaptive = (adaptive_ds_threshold is not None) and fs_kwargs
        if _use_adaptive:
            if adaptive_frame_threshold is not None:
                _adaptive_state = "frame"
            else:
                _adaptive_state = "ds"
            _adaptive_log = {}  # {global_idx: (ent_ratio, strategy)}
        else:
            _adaptive_state = None
            _adaptive_log = None

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    # --- Adaptive entropy routing (when enabled) ---
                    if _adaptive_state is not None:
                        # Ensure tokens are in global layout (B, S*P, C) for entropy probe
                        _tokens_for_ent = tokens.view(B, S * P, C) if tokens.shape != (B, S * P, C) else tokens
                        _pos_for_ent = pos.view(B, S * P, 2) if (pos is not None and pos.shape != (B, S * P, 2)) else pos
                        ent_ratio = compute_entropy_ratio(
                            self.global_blocks[global_idx], _tokens_for_ent, _pos_for_ent,
                            num_frames=fs_kwargs.get("num_frames", S),
                            tokens_per_frame=fs_kwargs.get("tokens_per_frame", P),
                        )

                        # State transitions (one-directional: frame -> ds -> full)
                        if _adaptive_state == "frame":
                            if adaptive_frame_threshold is not None and ent_ratio >= adaptive_frame_threshold:
                                # High entropy: route to frame attention
                                _adaptive_log[global_idx] = (ent_ratio, "FRAME")
                                tokens, global_idx, global_intermediates = (
                                    self._process_global_as_frame_attention(
                                        tokens, B, S, P, C, global_idx, pos=pos,
                                    )
                                )
                                goto_next = True
                            else:
                                # Transition to DS state
                                _adaptive_state = "ds"
                                goto_next = False
                        else:
                            goto_next = False

                        if not goto_next and _adaptive_state == "ds":
                            if ent_ratio >= adaptive_ds_threshold:
                                # Medium entropy: apply token downsampling
                                _adaptive_log[global_idx] = (ent_ratio, "TOKEN_DS")
                                tokens, global_idx, global_intermediates = (
                                    self._process_global_attention(
                                        tokens, B, S, P, C, global_idx, pos=pos,
                                        token_ds_layers=None,
                                        token_ds_entropy_threshold=None,
                                        layer_overrides=_layer_overrides,
                                        **fs_kwargs
                                    )
                                )
                                goto_next = True
                            else:
                                # Transition to full state (terminal)
                                _adaptive_state = "full"
                                goto_next = False

                        if not goto_next and _adaptive_state == "full":
                            _adaptive_log[global_idx] = (ent_ratio, "FULL")
                            tokens, global_idx, global_intermediates = (
                                self._process_global_attention(
                                    tokens, B, S, P, C, global_idx, pos=pos,
                                    token_ds_layers=None,
                                    token_ds_entropy_threshold=None,
                                    layer_overrides=_layer_overrides,
                                    **{**fs_kwargs, "token_keep_rate": 1.0, "token_downsample": 1}
                                )
                            )

                    # --- Static routing (original behavior) ---
                    elif _global_as_meanpool is not None and global_idx in _global_as_meanpool:
                        tokens, global_idx, global_intermediates = (
                            self._process_global_as_meanpool(
                                tokens, B, S, P, C, global_idx, pos=pos,
                            )
                        )
                    elif _global_as_frame is not None and global_idx in _global_as_frame:
                        tokens, global_idx, global_intermediates = (
                            self._process_global_as_frame_attention(
                                tokens, B, S, P, C, global_idx, pos=pos,
                            )
                        )
                    else:
                        tokens, global_idx, global_intermediates = (
                            self._process_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos,
                                token_ds_layers=_token_ds_layers,
                                token_ds_entropy_threshold=_token_ds_entropy_threshold,
                                layer_overrides=_layer_overrides,
                                **fs_kwargs
                            )
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                if retain_indices is None or out_idx in retain_indices:
                    # Only materialize the concatenated tensor for needed indices
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)
                else:
                    output_list.append(None)
                out_idx += 1

        # Print adaptive routing summary
        if _adaptive_log:
            frame_layers = sorted(k for k, (_, s) in _adaptive_log.items() if s == "FRAME")
            ds_layers = sorted(k for k, (_, s) in _adaptive_log.items() if s == "TOKEN_DS")
            full_layers = sorted(k for k, (_, s) in _adaptive_log.items() if s == "FULL")
            parts = []
            if frame_layers:
                parts.append(f"FRAME={frame_layers}")
            if ds_layers:
                parts.append(f"TOKEN_DS={ds_layers}")
            if full_layers:
                parts.append(f"FULL={full_layers}")
            layer_detail = "  ".join(
                f"L{k}:{s}({e:.3f})" for k, (e, s) in sorted(_adaptive_log.items())
            )
            print(f"  [adaptive summary] {' | '.join(parts)}")
            print(f"  [adaptive detail]  {layer_detail}")

        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_as_frame_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Run a global attention block with frame-attention layout.

        Instead of rearranging tokens to (B, S*P, C) for joint cross-frame
        attention, keeps the per-frame layout (B*S, P, C) so attention runs
        independently per frame. Uses the global block's weights unchanged.
        Reduces cost from O((NL)^2) to O(NL^2).
        """
        # Keep per-frame layout
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _process_global_as_meanpool(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Replace global attention with mean-pooling of V.

        For early layers where attention weights are nearly uniform,
        output ≈ mean(V) for every query. We skip Q/K entirely:
          1. Project V only (last third of qkv weights)
          2. Mean-pool V across all S*P tokens → single vector
          3. Broadcast back, apply output projection + LayerScale
          4. FFN runs unchanged

        Cost: O(NL) instead of O((NL)^2). Memory: negligible vs attention.
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        intermediates = []

        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]

            # --- Attention replacement: mean-pool of V ---
            normed = block.norm1(tokens)

            # Extract V projection weights from fused qkv
            W = block.attn.qkv.weight  # (3*C, C)
            b = block.attn.qkv.bias    # (3*C,) or None
            W_v = W[2 * C:]
            b_v = b[2 * C:] if b is not None else None

            v = torch.nn.functional.linear(normed, W_v, b_v)  # (B, S*P, C)
            v_mean = v.mean(dim=1, keepdim=True)               # (B, 1, C)
            v_mean = v_mean.expand_as(v)                       # (B, S*P, C)

            attn_out = block.attn.proj(v_mean)                 # output projection
            attn_out = block.attn.proj_drop(attn_out)
            attn_out = block.ls1(attn_out)                     # LayerScale

            tokens = tokens + attn_out

            # --- FFN runs unchanged ---
            tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None,
                                   token_ds_layers=None, token_ds_entropy_threshold=None,
                                   layer_overrides=None, **fs_kwargs):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            layer_kwargs = fs_kwargs

            if token_ds_entropy_threshold is not None and fs_kwargs:
                # Dynamic per-layer decision: compute Ent/Max and compare to threshold
                ent_ratio = compute_entropy_ratio(
                    self.global_blocks[global_idx], tokens, pos,
                    num_frames=fs_kwargs.get("num_frames", S),
                    tokens_per_frame=fs_kwargs.get("tokens_per_frame", P),
                )
                if ent_ratio >= token_ds_entropy_threshold:
                    logging.info(
                        f"  [entropy] layer {global_idx}: Ent/Max={ent_ratio:.3f} "
                        f">= {token_ds_entropy_threshold} -> DOWNSAMPLE"
                    )
                else:
                    logging.info(
                        f"  [entropy] layer {global_idx}: Ent/Max={ent_ratio:.3f} "
                        f"< {token_ds_entropy_threshold} -> skip downsample"
                    )
                    layer_kwargs = {**fs_kwargs, "token_keep_rate": 1.0, "token_downsample": 1}
            elif token_ds_layers is not None and fs_kwargs:
                if global_idx not in token_ds_layers:
                    # Disable all intra-frame token downsampling for this layer
                    layer_kwargs = {**fs_kwargs, "token_keep_rate": 1.0, "token_downsample": 1}

            # Apply per-layer overrides from layer_config
            if layer_overrides and global_idx in layer_overrides:
                layer_kwargs = {**layer_kwargs, **layer_overrides[global_idx]}
            tokens = self.global_blocks[global_idx](tokens, pos=pos, **layer_kwargs)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


@torch.no_grad()
def compute_entropy_ratio(block, tokens, pos, num_frames, tokens_per_frame,
                          sample_heads=4, sample_q_tokens=32):
    """Lightweight entropy probe: compute Ent/Max for a global attention layer.

    Samples a subset of heads and query tokens to estimate how uniform the
    inter-frame attention distribution is.  Returns a scalar float in [0, 1].
    """
    attn_module = block.attn
    x = block.norm1(tokens)
    B, N, C = x.shape
    num_heads = attn_module.num_heads
    head_dim = C // num_heads

    NF = num_frames
    hw = tokens_per_frame

    # Compute Q, K
    qkv = (
        attn_module.qkv(x)
        .reshape(B, N, 3, num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, _v = qkv.unbind(0)
    q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    if attn_module.rope is not None and pos is not None:
        q = attn_module.rope(q, pos)
        k = attn_module.rope(k, pos)

    # Reshape to per-frame: (B, heads, NF, hw, dim)
    q_frames = q.reshape(B, num_heads, NF, hw, head_dim)
    k_frames = k.reshape(B, num_heads, NF, hw, head_dim)

    # Sample heads
    h_sample = min(sample_heads, num_heads)
    h_list = torch.linspace(0, num_heads - 1, h_sample).long()

    # Sample query tokens per frame
    n_qtok = min(sample_q_tokens, hw)
    if n_qtok < hw:
        qtok_idx = torch.linspace(0, hw - 1, n_qtok).long().to(q.device)
    else:
        qtok_idx = torch.arange(hw, device=q.device)

    # Pick a single representative query frame (middle frame)
    qi = NF // 2
    # All frames as context (full attention)
    K = NF

    # Q: (B, h_sample, n_qtok, dim)
    q_i = q_frames[:, h_list][:, :, qi][:, :, qtok_idx]
    # K: (B, h_sample, NF*hw, dim)
    k_i = k_frames[:, h_list].reshape(B, h_sample, K * hw, head_dim)

    scale = head_dim ** -0.5
    logits = (q_i.float() @ k_i.float().transpose(-2, -1)) * scale
    attn_weights = logits.softmax(dim=-1)  # (B, h_sample, n_qtok, K*hw)

    eps = 1e-10
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (B, h_sample, n_qtok)
    max_entropy = math.log(K * hw)

    ent_ratio = entropy.mean().item() / max_entropy if max_entropy > 0 else 0.0
    return ent_ratio


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
