import logging
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from mapanything.models.external.dinov2.hub.backbones import dinov2_vitl14_reg
from mapanything.models.external.dinov2.layers import Mlp
from mapanything.models.external.pi3.layers.attention import FlashAttentionRope
from mapanything.models.external.pi3.layers.block import BlockRope
from mapanything.models.external.pi3.layers.camera_head import CameraHead
from mapanything.models.external.pi3.layers.pos_embed import PositionGetter, RoPE2D
from mapanything.models.external.pi3.layers.frame_selection import (
    FrameSelector,
    compute_covisibility_candidate_mask,
    select_frames_closest,
    select_frames_even,
    select_frames_diverse,
    select_frames_diverse_self,
    select_frames_random,
)
from mapanything.models.external.pi3.layers.transformer_head import (
    LinearPts3d,
    TransformerDecoder,
)


def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        pos_type="rope100",
        decoder_size="large",
    ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):  # eg rope100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == "small":
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == "base":
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == "large":
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList(
            [
                BlockRope(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=0.01,
                    qk_norm=True,
                    attn_class=FlashAttentionRope,
                    rope=self.rope,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W, frame_selector=None, token_downsample=1, token_keep_rate=1.0, token_selection_method="diverse", ds_keep_register=False, layer_config=None, adaptive_ds_threshold=None, adaptive_frame_threshold=None, tau=1.0, tau_layers=None):
        """
        Decode hidden states through alternating single-view and cross-view layers.

        Args:
            hidden: Encoder output (B*N, hw_patches, dim)
            N: Number of views/frames
            H, W: Image height and width
            frame_selector: Optional FrameSelector for sparse cross-view attention.
                If None, uses full attention (original behavior).
            token_downsample: Spatial stride factor for K/V token downsampling.
                Can be used independently of frame_selector. Default: 1 (no downsampling).

        Returns:
            Tuple of (concatenated last two layer outputs, positions)
        """
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith("rope"):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * N, self.patch_start_idx, 2)
                .to(hidden.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # Prepare frame selection parameters for cross-view layers
        # These are set once and passed to every odd-layer block.
        fs_kwargs = {}  # extra kwargs for cross-view blocks
        if frame_selector is not None and N > 1:
            # Compute covisibility candidate mask (shared across all layers)
            candidate_mask = None
            if frame_selector.use_covisibility:
                candidate_mask = compute_covisibility_candidate_mask(
                    frame_selector.covisibility_matrix,
                    frame_selector.covisibility_percentile,
                    device=hidden.device,
                )
                # Handle size mismatch
                if candidate_mask.shape[0] != N:
                    if candidate_mask.shape[0] > N:
                        candidate_mask = candidate_mask[:N, :N]
                    else:
                        padded = torch.ones(N, N, device=hidden.device, dtype=torch.bool)
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
                # Per-layer top-K: each cross-view layer computes its own
                # selection. 'topk' / 'topk_mean' use cosine-sim with max/mean
                # pool. 'incvggt_max' / 'incvggt_mean' use raw scaled Q@K^T
                # logits reduced over tokens and heads with max/mean pool.
                fs_kwargs = dict(
                    topk_frames=frame_selector.top_k,
                    score_method=frame_selector.strategy,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                    num_frames=N,
                    tokens_per_frame=hw,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "even":
                # Precomputed: same selection for all layers
                frame_neighbors = select_frames_even(
                    num_frames=N,
                    top_k=frame_selector.top_k,
                    device=hidden.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=N,
                    tokens_per_frame=hw,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "diverse":
                # FPS on covisibility: maximally diverse K frames per query
                import time as _time
                _t0 = _time.perf_counter()
                frame_neighbors = select_frames_diverse(
                    covisibility_matrix=frame_selector.covisibility_matrix,
                    top_k=frame_selector.top_k,
                    device=hidden.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                _t1 = _time.perf_counter()
                print(f"[TIMER] select_frames_diverse (Pi3 decoder): {_t1 - _t0:.4f}s")
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=N,
                    tokens_per_frame=hw,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "diverse_self":
                # Per-query FPS seeded by the query frame itself
                import time as _time
                _t0 = _time.perf_counter()
                frame_neighbors = select_frames_diverse_self(
                    covisibility_matrix=frame_selector.covisibility_matrix,
                    top_k=frame_selector.top_k,
                    device=hidden.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                _t1 = _time.perf_counter()
                print(f"[TIMER] select_frames_diverse_self (Pi3 decoder): {_t1 - _t0:.4f}s")
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=N,
                    tokens_per_frame=hw,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )
            elif frame_selector.strategy == "random":
                # Random K frames per query
                frame_neighbors = select_frames_random(
                    num_frames=N,
                    top_k=frame_selector.top_k,
                    device=hidden.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=N,
                    tokens_per_frame=hw,
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
                    num_frames=N,
                    top_k=frame_selector.top_k,
                    device=hidden.device,
                    include_self=frame_selector.include_self,
                    candidate_mask=candidate_mask,
                )
                fs_kwargs = dict(
                    frame_neighbors=frame_neighbors,
                    num_frames=N,
                    tokens_per_frame=hw,
                    patch_start_idx=psi,
                    token_downsample=token_downsample,
                    token_keep_rate=token_keep_rate,
                    token_selection_method=token_selection_method,
                    tokens_h=tokens_h,
                    tokens_w=tokens_w,
                    batched_sdpa=batched_sdpa,
                )

            # --- DEBUG: print precomputed frame selections (even / diverse / diverse_self / random / closest) ---
            if frame_selector.strategy in ("even", "diverse", "diverse_self", "random", "closest"):
                K_dbg = frame_neighbors.shape[1]
                sample_idxs = sorted(set(
                    i for i in [0, N // 4, N // 2, 3 * N // 4, N - 1]
                    if 0 <= i < N
                ))
                lines = []
                for si in sample_idxs:
                    nb = frame_neighbors[si].tolist()
                    lines.append(f"  query {si:3d} -> {nb}")
                print(f"[{frame_selector.strategy} debug] NF={N}, K={K_dbg}, "
                      f"selections (same for all layers):\n" + "\n".join(lines))
            # --- END DEBUG ---

        # Standalone token_downsample / token_keep_rate without frame selection:
        # pass spatial info so the attention layer can downsample K/V globally.
        if not fs_kwargs and (token_downsample > 1 or token_keep_rate < 1.0) and N > 1:
            tokens_h = H // self.patch_size
            tokens_w = W // self.patch_size
            psi = self.patch_start_idx if ds_keep_register else 0
            fs_kwargs = dict(
                num_frames=N,
                tokens_per_frame=hw,
                patch_start_idx=psi,
                token_downsample=token_downsample,
                token_keep_rate=token_keep_rate,
                token_selection_method=token_selection_method,
                tokens_h=tokens_h,
                tokens_w=tokens_w,
            )

        # Always pass num_frames / tokens_per_frame so downstream hooks
        # (e.g. attention diagnostics) can identify cross-view attention layers.
        if not fs_kwargs and N > 1:
            fs_kwargs = dict(
                num_frames=N,
                tokens_per_frame=hw,
            )

        # Determine which global layers need token downsampling
        _token_ds_layers = None
        if frame_selector is not None and frame_selector.token_ds_layers is not None:
            _token_ds_layers = frame_selector.token_ds_layers

        _token_ds_entropy_threshold = None
        if frame_selector is not None and getattr(frame_selector, 'token_ds_entropy_threshold', None) is not None:
            _token_ds_entropy_threshold = frame_selector.token_ds_entropy_threshold

        # Parse layer_config for per-layer overrides and routing
        _layer_overrides = {}
        _global_as_frame = set()
        _global_as_meanpool = set()
        if layer_config is not None:
            for idx, cfg in layer_config.items():
                strategy = cfg.get("strategy", "global")
                if strategy == "meanpool":
                    _global_as_meanpool.add(idx)
                elif strategy == "frame":
                    _global_as_frame.add(idx)
                overrides = {k: v for k, v in cfg.items() if k != "strategy"}
                if overrides:
                    _layer_overrides[idx] = overrides

        # ----- Attention-temperature (dilution) probe -----
        # tau != 1.0 sharpens (or flattens) the post-softmax distribution at the
        # listed cross-view layers. tau == 1.0 keeps the SDPA fast path intact.
        # tau_layers is a set/list of cross-view layer indices in [0, dec_depth/2);
        # None means apply to every cross-view layer.
        if tau_layers is not None:
            _tau_layer_set = set(int(l) for l in tau_layers)
        else:
            _tau_layer_set = None
        if tau != 1.0:
            _applied = sorted(_tau_layer_set) if _tau_layer_set is not None else "ALL"
            print(f"[tau probe] tau={tau} applied to cross-view layers: {_applied}")

        C = hidden.shape[-1]

        # Adaptive entropy-based routing state machine (mirrors VGGT aggregator).
        # States: "frame" -> "ds" -> "full" (one-directional transitions).
        _use_adaptive = (adaptive_ds_threshold is not None) and bool(fs_kwargs)
        if _use_adaptive:
            _adaptive_state = "frame" if adaptive_frame_threshold is not None else "ds"
            _adaptive_log = {}  # {global_idx: (ent_ratio, strategy)}
        else:
            _adaptive_state = None
            _adaptive_log = None

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                # Even layers: per-view self-attention (no frame selection)
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
                hidden = blk(hidden, xpos=pos)
            else:
                global_idx = i // 2

                if _adaptive_state is not None:
                    # --- Adaptive entropy routing ---
                    # Probe needs (B, N*hw, C) layout
                    pos = pos.reshape(B, N * hw, -1)
                    hidden = hidden.reshape(B, N * hw, -1)
                    ent_ratio = _compute_entropy_ratio_pi3(
                        blk, hidden, pos,
                        num_frames=fs_kwargs.get("num_frames", N),
                        tokens_per_frame=fs_kwargs.get("tokens_per_frame", hw),
                    )

                    # State transitions (frame -> ds -> full, one-directional)
                    goto_next = False
                    if _adaptive_state == "frame":
                        if adaptive_frame_threshold is not None and ent_ratio >= adaptive_frame_threshold:
                            _adaptive_log[global_idx] = (ent_ratio, "FRAME")
                            # per-frame layout
                            pos = pos.reshape(B * N, hw, -1)
                            hidden = hidden.reshape(B * N, hw, -1)
                            hidden = blk(hidden, xpos=pos)
                            goto_next = True
                        else:
                            _adaptive_state = "ds"

                    if not goto_next and _adaptive_state == "ds":
                        if ent_ratio >= adaptive_ds_threshold:
                            _adaptive_log[global_idx] = (ent_ratio, "TOKEN_DS")
                            layer_kwargs = fs_kwargs
                            if _layer_overrides and global_idx in _layer_overrides:
                                layer_kwargs = {**layer_kwargs, **_layer_overrides[global_idx]}
                            hidden = blk(hidden, xpos=pos, **layer_kwargs)
                            goto_next = True
                        else:
                            _adaptive_state = "full"

                    if not goto_next and _adaptive_state == "full":
                        _adaptive_log[global_idx] = (ent_ratio, "FULL")
                        full_kwargs = {**fs_kwargs, "token_keep_rate": 1.0, "token_downsample": 1}
                        if _layer_overrides and global_idx in _layer_overrides:
                            full_kwargs = {**full_kwargs, **_layer_overrides[global_idx]}
                        _tau_for_layer = tau if (
                            tau != 1.0 and (_tau_layer_set is None or global_idx in _tau_layer_set)
                        ) else 1.0
                        if _tau_for_layer != 1.0:
                            full_kwargs = {**full_kwargs, "tau": _tau_for_layer}
                        hidden = blk(hidden, xpos=pos, **full_kwargs)

                elif global_idx in _global_as_meanpool:
                    print(f"[MEANPOOL] global_idx={global_idx}")
                    # Mean-pool: skip attention, output = mean(V) broadcast
                    hidden = hidden.reshape(B, N * hw, -1)
                    normed = blk.norm1(hidden)
                    attn = blk.attn
                    # Extract V weights from fused qkv
                    W = attn.qkv.weight  # (3*C, C)
                    b = attn.qkv.bias
                    W_v = W[2 * C:]
                    b_v = b[2 * C:] if b is not None else None
                    v = torch.nn.functional.linear(normed, W_v, b_v)
                    v_mean = v.mean(dim=1, keepdim=True).expand_as(v)
                    attn_out = attn.proj(v_mean)
                    attn_out = attn.proj_drop(attn_out)
                    hidden = hidden + blk.ls1(attn_out)
                    # FFN
                    hidden = hidden + blk.ls2(blk.mlp(blk.norm2(hidden)))

                elif global_idx in _global_as_frame:
                    # Per-frame attention: keep (B*N, hw, C) layout instead of (B, N*hw, C)
                    pos = pos.reshape(B * N, hw, -1)
                    hidden = hidden.reshape(B * N, hw, -1)
                    hidden = blk(hidden, xpos=pos)

                else:
                    # Normal cross-view attention
                    pos = pos.reshape(B, N * hw, -1)
                    hidden = hidden.reshape(B, N * hw, -1)
                    layer_kwargs = fs_kwargs

                    # Per-layer attention temperature for dilution probe
                    _tau_for_layer = tau if (
                        tau != 1.0 and (_tau_layer_set is None or global_idx in _tau_layer_set)
                    ) else 1.0
                    if _tau_for_layer != 1.0:
                        layer_kwargs = {**layer_kwargs, "tau": _tau_for_layer}

                    if _token_ds_entropy_threshold is not None and fs_kwargs:
                        # Dynamic per-layer decision via entropy probe
                        ent_ratio = _compute_entropy_ratio_pi3(
                            blk, hidden, pos,
                            num_frames=fs_kwargs.get("num_frames", N),
                            tokens_per_frame=fs_kwargs.get("tokens_per_frame", hw),
                        )
                        if ent_ratio >= _token_ds_entropy_threshold:
                            logging.info(
                                f"  [entropy] layer {global_idx}: Ent/Max={ent_ratio:.3f} "
                                f">= {_token_ds_entropy_threshold} -> DOWNSAMPLE"
                            )
                        else:
                            logging.info(
                                f"  [entropy] layer {global_idx}: Ent/Max={ent_ratio:.3f} "
                                f"< {_token_ds_entropy_threshold} -> skip downsample"
                            )
                            layer_kwargs = {**fs_kwargs, "token_keep_rate": 1.0, "token_downsample": 1}
                    elif _token_ds_layers is not None and fs_kwargs:
                        if global_idx not in _token_ds_layers:
                            layer_kwargs = {**fs_kwargs, "token_keep_rate": 1.0, "token_downsample": 1}
                    # Apply per-layer overrides from layer_config
                    if _layer_overrides and global_idx in _layer_overrides:
                        layer_kwargs = {**layer_kwargs, **_layer_overrides[global_idx]}
                    hidden = blk(hidden, xpos=pos, **layer_kwargs)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        # Print adaptive routing summary (mirrors VGGT aggregator)
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

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(
            B * N, hw, -1
        )

    def forward(self, imgs, frame_selector=None, token_downsample=1, token_keep_rate=1.0, token_selection_method="diverse", ds_keep_register=False, backbone_minibatch_size=0, layer_config=None, adaptive_ds_threshold=None, adaptive_frame_threshold=None, tau=1.0, tau_layers=None):
        """
        Forward pass.

        Args:
            imgs: (B, N, C, H, W) input images
            frame_selector: Optional FrameSelector for sparse cross-view attention.
                If None, uses full attention (original behavior).
            token_downsample: Spatial stride factor for K/V token downsampling.
                Can be used independently of frame_selector. Default: 1 (no downsampling).
            backbone_minibatch_size: Process DINOv2 backbone in mini-batches of this
                size (0 = all at once). Reduces peak GPU memory.
        """
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode by dinov2
        imgs = imgs.reshape(B * N, _, H, W)
        if backbone_minibatch_size > 0 and B * N > backbone_minibatch_size:
            chunks = imgs.split(backbone_minibatch_size, dim=0)
            outs = []
            for chunk in chunks:
                out = self.encoder(chunk, is_training=True)
                if isinstance(out, dict):
                    out = out["x_norm_patchtokens"]
                outs.append(out)
            hidden = torch.cat(outs, dim=0)
        else:
            hidden = self.encoder(imgs, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(
            hidden, N, H, W,
            frame_selector=frame_selector,
            token_downsample=token_downsample,
            token_keep_rate=token_keep_rate,
            token_selection_method=token_selection_method,
            ds_keep_register=ds_keep_register,
            layer_config=layer_config,
            adaptive_ds_threshold=adaptive_ds_threshold,
            adaptive_frame_threshold=adaptive_frame_threshold,
            tau=tau,
            tau_layers=tau_layers,
        )

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head(
                [conf_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
            ).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum(
                "bnij, bnhwj -> bnhwi", camera_poses, homogenize_points(local_points)
            )[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )


@torch.no_grad()
def _compute_entropy_ratio_pi3(block, tokens, pos, num_frames, tokens_per_frame,
                                sample_heads=4, sample_q_tokens=32):
    """Lightweight entropy probe for Pi3 blocks (uses xpos/RoPE)."""
    attn_module = block.attn
    x = block.norm1(tokens)
    B, N, C = x.shape
    num_heads = attn_module.num_heads
    head_dim = C // num_heads

    NF = num_frames
    hw = tokens_per_frame

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

    q_frames = q.reshape(B, num_heads, NF, hw, head_dim)
    k_frames = k.reshape(B, num_heads, NF, hw, head_dim)

    h_sample = min(sample_heads, num_heads)
    h_list = torch.linspace(0, num_heads - 1, h_sample).long()

    n_qtok = min(sample_q_tokens, hw)
    if n_qtok < hw:
        qtok_idx = torch.linspace(0, hw - 1, n_qtok).long().to(q.device)
    else:
        qtok_idx = torch.arange(hw, device=q.device)

    qi = NF // 2
    K = NF

    q_i = q_frames[:, h_list][:, :, qi][:, :, qtok_idx]
    k_i = k_frames[:, h_list].reshape(B, h_sample, K * hw, head_dim)

    scale = head_dim ** -0.5
    logits = (q_i.float() @ k_i.float().transpose(-2, -1)) * scale
    attn_weights = logits.softmax(dim=-1)

    eps = 1e-10
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
    max_entropy = math.log(K * hw)

    ent_ratio = entropy.mean().item() / max_entropy if max_entropy > 0 else 0.0
    return ent_ratio
