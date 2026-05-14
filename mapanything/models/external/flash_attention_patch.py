"""
Monkey-patch utilities to replace auto-selected SDPA backends with explicit
Flash Attention backend selection in external libraries (DA3, UniCeption).

Follows the same pattern as Pi-3's FlashAttention class:
  - bf16 + no attn_mask  → SDPBackend.FLASH_ATTENTION
  - otherwise            → [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend

logger = logging.getLogger(__name__)


def _flash_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, **kwargs):
    """Drop-in replacement for F.scaled_dot_product_attention with explicit backend."""
    if attn_mask is None and q.dtype == torch.bfloat16:
        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, **kwargs
            )
    else:
        with nn.attention.sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        ):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, **kwargs
            )


# ---------------------------------------------------------------------------
# DA3 (depth_anything_3) patches
# ---------------------------------------------------------------------------

def _da3_dino_attn_forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
    """Patched forward for depth_anything_3.model.dinov2.layers.attention.Attention."""
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)
    if self.fused_attn:
        expanded_mask = (
            (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
            if attn_mask is not None
            else None
        )
        x = _flash_sdpa(
            q, k, v,
            attn_mask=expanded_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
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


def _da3_util_attn_forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
    """Patched forward for depth_anything_3.model.utils.attention.Attention."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    q = self.rope(q, pos) if self.rope is not None else q
    k = self.rope(k, pos) if self.rope is not None else k
    x = _flash_sdpa(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=self.attn_drop.p if self.training else 0.0,
    )
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_da3_attention():
    """Monkey-patch DA3 attention classes to use explicit Flash Attention backend."""
    try:
        import depth_anything_3.model.dinov2.layers.attention as da3_dino_attn
        da3_dino_attn.Attention.forward = _da3_dino_attn_forward
        logger.info("Patched DA3 DINOv2 attention with Flash Attention backend selection")
    except ImportError:
        logger.warning("depth_anything_3.model.dinov2.layers.attention not found, skipping patch")

    try:
        import depth_anything_3.model.utils.attention as da3_util_attn
        da3_util_attn.Attention.forward = _da3_util_attn_forward
        logger.info("Patched DA3 utils attention with Flash Attention backend selection")
    except ImportError:
        logger.warning("depth_anything_3.model.utils.attention not found, skipping patch")


# ---------------------------------------------------------------------------
# UniCeption patches
# ---------------------------------------------------------------------------

def _uc_attn_forward(self, x: torch.Tensor, xpos: torch.Tensor = None) -> torch.Tensor:
    """Patched forward for uniception Attention (self-attention)."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)

    if self.custom_positional_encoding is not None:
        assert xpos is not None, (
            "Positions of tokens (xpos) are a required input when using custom positional encoding"
        )
        q = self.custom_positional_encoding(q, xpos)
        k = self.custom_positional_encoding(k, xpos)

    if self.use_scalable_softmax:
        q = q * torch.log(torch.tensor(N, device=q.device))

    if self.use_entropy_scaling:
        scaling_factor = torch.sqrt(
            (self.entropy_scaling_growth_factor * torch.log(torch.tensor(N, device=q.device)))
            / torch.log(torch.tensor(self.base_token_count_for_entropy_scaling, device=q.device))
        )
        q = q * scaling_factor

    if self.fused_attn:
        x = _flash_sdpa(
            q, k, v,
            dropout_p=(self.attn_drop.p if self.training else 0.0),
            scale=self.scale,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _uc_cross_attn_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    qpos: torch.Tensor = None,
    kpos: torch.Tensor = None,
) -> torch.Tensor:
    """Patched forward for uniception CrossAttention."""
    B, Nq, C = query.shape
    Nk = key.shape[1]
    Nv = value.shape[1]

    q = self.projq(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.projk(key).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.projv(value).reshape(B, Nv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.custom_positional_encoding is not None:
        assert qpos is not None, (
            "Positions of queries (qpos) are a required input when using custom positional encoding"
        )
        assert kpos is not None, (
            "Positions of keys (kpos) are a required input when using custom positional encoding"
        )
        q = self.custom_positional_encoding(q, qpos)
        k = self.custom_positional_encoding(k, kpos)

    if self.use_scalable_softmax:
        q = q * torch.log(torch.tensor(Nq, device=q.device))

    if self.use_entropy_scaling:
        scaling_factor = torch.sqrt(
            (self.entropy_scaling_growth_factor * torch.log(torch.tensor(Nq, device=q.device)))
            / torch.log(torch.tensor(self.base_token_count_for_entropy_scaling, device=q.device))
        )
        q = q * scaling_factor

    if self.fused_attn:
        x = _flash_sdpa(
            q, k, v,
            dropout_p=(self.attn_drop.p if self.training else 0.0),
            scale=self.scale,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, Nq, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _uc_diff_attn_forward(self, x: torch.Tensor, xpos: torch.Tensor = None) -> torch.Tensor:
    """Patched forward for uniception DiffAttention."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim * 2)
    q, k, v = torch.chunk(qkv, 3, dim=2)

    q = q.view(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    k = k.view(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    v = v.view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)

    q, k = self.q_norm(q), self.k_norm(k)

    if self.custom_positional_encoding is not None:
        assert xpos is not None, (
            "Positions of tokens (xpos) are a required input when using custom positional encoding"
        )
        q = self.custom_positional_encoding(q, xpos)
        k = self.custom_positional_encoding(k, xpos)

    q1, q2 = q.chunk(2, dim=1)
    k1, k2 = k.chunk(2, dim=1)

    if self.fused_attn:
        attn1 = _flash_sdpa(
            q1, k1, v,
            dropout_p=(self.attn_drop.p if self.training else 0.0),
            scale=self.scale,
        )
        attn2 = _flash_sdpa(
            q2, k2, v,
            dropout_p=(self.attn_drop.p if self.training else 0.0),
            scale=self.scale,
        )
    else:
        q1 = q1 * self.scale
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn1 = attn @ v

        q2 = q2 * self.scale
        attn = q2 @ k2.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn2 = attn @ v

    lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
    lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
    lambda_full = lambda_1 - lambda_2 + self.lambda_init
    attn = attn1 - lambda_full * attn2

    attn = self.subln(attn)
    attn = attn * (1 - self.lambda_init)
    attn = attn.reshape(B, N, self.num_heads * 2 * self.head_dim)

    x = self.proj(attn)
    x = self.proj_drop(x)
    return x


def _uc_diff_cross_attn_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    qpos: torch.Tensor = None,
    kpos: torch.Tensor = None,
) -> torch.Tensor:
    """Patched forward for uniception DiffCrossAttention."""
    B, Nq, C = query.shape
    Nk = key.shape[1]
    Nv = value.shape[1]

    q = self.projq(query).reshape(B, Nq, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.projk(key).reshape(B, Nk, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.projv(value).reshape(B, Nv, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.custom_positional_encoding is not None:
        assert qpos is not None, (
            "Positions of queries (qpos) are a required input when using custom positional encoding"
        )
        assert kpos is not None, (
            "Positions of keys (kpos) are a required input when using custom positional encoding"
        )
        q = self.custom_positional_encoding(q, qpos)
        k = self.custom_positional_encoding(k, kpos)

    q1, q2 = q.chunk(2, dim=1)
    k1, k2 = k.chunk(2, dim=1)

    if self.fused_attn:
        attn1 = _flash_sdpa(
            q1, k1, v,
            dropout_p=(self.attn_drop.p if self.training else 0.0),
            scale=self.scale,
        )
        attn2 = _flash_sdpa(
            q2, k2, v,
            dropout_p=(self.attn_drop.p if self.training else 0.0),
            scale=self.scale,
        )
    else:
        q1 = q1 * self.scale
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn1 = attn @ v

        q2 = q2 * self.scale
        attn = q2 @ k2.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn2 = attn @ v

    attn1 = attn1.transpose(1, 2)
    attn2 = attn2.transpose(1, 2)

    lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
    lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
    lambda_full = lambda_1 - lambda_2 + self.lambda_init
    attn = attn1 - lambda_full * attn2

    attn = self.subln(attn)
    attn = attn * (1 - self.lambda_init)
    attn = attn.reshape(B, Nq, self.num_heads * 2 * self.head_dim)

    x = self.proj(attn)
    x = self.proj_drop(x)
    return x


def patch_uniception_attention():
    """Monkey-patch UniCeption attention classes to use explicit Flash Attention backend."""
    try:
        import uniception.models.utils.transformer_blocks as uc_blocks

        uc_blocks.Attention.forward = _uc_attn_forward
        logger.info("Patched UniCeption Attention with Flash Attention backend selection")

        uc_blocks.CrossAttention.forward = _uc_cross_attn_forward
        logger.info("Patched UniCeption CrossAttention with Flash Attention backend selection")

        uc_blocks.DiffAttention.forward = _uc_diff_attn_forward
        logger.info("Patched UniCeption DiffAttention with Flash Attention backend selection")

        uc_blocks.DiffCrossAttention.forward = _uc_diff_cross_attn_forward
        logger.info("Patched UniCeption DiffCrossAttention with Flash Attention backend selection")
    except ImportError:
        logger.warning("uniception.models.utils.transformer_blocks not found, skipping patch")
