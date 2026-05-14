"""
Frame Selection Strategies for Pi3 Sparse Cross-View Attention.

For each query frame, selects K frames (from all N frames, both past and future)
to attend to in the cross-view attention layers (odd layers in the decoder).
This reduces O(N^2) cross-view attention to O(N*K).

Strategies:
- 'topk': Per-layer selection using max-pooled P×P cosine-similarity between
          Q and K (heads concatenated). Each cross-view layer independently
          selects K frames per query frame.
- 'topk_mean': Same as 'topk' but with mean-pooling over the P×P similarity
          map instead of max.
- 'incvggt_max': Per-layer selection using raw scaled (Q@K^T) logits (IncVGGT-
          style). Max-pool over key tokens, query tokens, and heads.
- 'incvggt_mean': Same as 'incvggt_max' but with mean-pooling over tokens/heads
          instead of max.
- 'even': Precomputed uniform sampling of K frames across the sequence.
- 'random': Random K-frame sampling per query frame (same for all layers).

Covisibility pre-filter (optional, works with any strategy):
    When enabled, each query frame first filters to the top-X% of candidate
    frames by shared 3D points, then applies topk or even within that filtered set.
"""

import numpy as np
import random as _random
import torch
from typing import Optional


def compute_covisibility_candidate_mask(
    covisibility_matrix: np.ndarray,
    percentile: float = 50.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute a per-frame candidate mask from a covisibility matrix.

    For each query frame i, keeps only the frames j whose covisibility score
    covisibility_matrix[i, j] is above the (100 - percentile)-th percentile
    of all scores in row i.

    Args:
        covisibility_matrix: (N, N) array of shared 3D point counts.
        percentile: Keep the top-X% of frames. E.g., 50.0 keeps the top 50%.
        device: Target torch device.

    Returns:
        candidate_mask: (N, N) boolean tensor. mask[i, j] = True means frame j
            is a valid candidate for query frame i.
    """
    covis = covisibility_matrix.copy().astype(np.float64)
    N = covis.shape[0]

    # Zero out self-covisibility so it doesn't affect the percentile threshold
    np.fill_diagonal(covis, 0.0)

    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        row = covis[i]
        # Compute threshold: keep top percentile%
        threshold = np.percentile(row, 100.0 - percentile)
        mask[i] = row >= threshold

    # Always allow self-attention (will be handled by include_self)
    np.fill_diagonal(mask, True)

    result = torch.from_numpy(mask)
    if device is not None:
        result = result.to(device)
    return result


@torch.no_grad()
def compute_cosine_frame_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_start_idx: int = 0,
    pool: str = "max",
) -> torch.Tensor:
    """
    Compute pairwise frame-level scores using cosine-similarity pooling
    over P×P Q-K similarity maps (patch tokens only, excluding register tokens).

    For each pair of frames (i, j), computes the P×P cosine similarity map
    between frame i's query patch tokens and frame j's key patch tokens,
    then pools (max or mean) to produce one scalar score.

    Register tokens are excluded from scoring because they tend to have
    similar representations across frames, which corrupts the frame-level
    ranking when max-pooling is used.

    Memory-efficient implementation: processes one frame pair at a time.

    Args:
        q: (B, num_heads, N*hw, head_dim) query tensor after RoPE
        k: (B, num_heads, N*hw, head_dim) key tensor after RoPE
        num_frames: N
        tokens_per_frame: hw (including register tokens)
        patch_start_idx: Number of register/special tokens to skip (default 0).
        pool: 'max' (original topk behaviour) or 'mean'.

    Returns:
        scores: (B, N, N) pairwise frame scores
    """
    assert pool in ("max", "mean"), f"pool must be 'max' or 'mean', got '{pool}'"
    B, num_heads, _, head_dim = q.shape
    NF = num_frames
    hw = tokens_per_frame
    C = num_heads * head_dim

    # Concatenate heads and reshape to per-frame tokens: (B, NF, hw, C)
    q_frames = q.permute(0, 2, 1, 3).reshape(B, NF, hw, C)
    k_frames = k.permute(0, 2, 1, 3).reshape(B, NF, hw, C)

    # Skip register tokens — only use patch tokens for scoring
    q_patches = q_frames[:, :, patch_start_idx:]  # (B, NF, hw_patches, C)
    k_patches = k_frames[:, :, patch_start_idx:]  # (B, NF, hw_patches, C)

    scores = torch.zeros(B, NF, NF, device=q.device, dtype=q.dtype)

    for i in range(NF):
        # q_i: (B, hw_patches, C)
        q_i = q_patches[:, i]
        # L2-normalize for cosine similarity
        q_i = torch.nn.functional.normalize(q_i, dim=-1, p=2)

        for j in range(NF):
            # k_j: (B, hw_patches, C)
            k_j = k_patches[:, j]
            k_j = torch.nn.functional.normalize(k_j, dim=-1, p=2)

            # sim: (B, hw_patches, hw_patches) — cosine similarity between token pairs
            sim = torch.bmm(q_i, k_j.transpose(-1, -2))

            # Pool over all patch token pairs -> scalar score per batch element
            sim_flat = sim.reshape(B, -1)
            if pool == "max":
                scores[:, i, j] = sim_flat.max(dim=-1).values
            else:
                scores[:, i, j] = sim_flat.mean(dim=-1)

    return scores


@torch.no_grad()
def compute_maxpool_frame_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_start_idx: int = 0,
) -> torch.Tensor:
    """Max-pooled cosine-similarity scoring ('topk' strategy). Thin wrapper
    around compute_cosine_frame_scores(..., pool='max')."""
    return compute_cosine_frame_scores(
        q, k, num_frames, tokens_per_frame, patch_start_idx, pool="max",
    )


@torch.no_grad()
def compute_meanpool_frame_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_start_idx: int = 0,
) -> torch.Tensor:
    """Mean-pooled cosine-similarity scoring ('topk_mean' strategy). Thin
    wrapper around compute_cosine_frame_scores(..., pool='mean')."""
    return compute_cosine_frame_scores(
        q, k, num_frames, tokens_per_frame, patch_start_idx, pool="mean",
    )


@torch.no_grad()
def compute_incvggt_frame_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_start_idx: int = 0,
    pool: str = "max",
) -> torch.Tensor:
    """
    IncVGGT-style frame scoring: raw scaled Q@K^T logits, pooled over key
    tokens, query tokens, and heads.

    For each pair of frames (i, j):
        logits_ij = (q_i * scale) @ k_j^T                # (B, H, P, P) per-head
        scores_ij = logits_ij.pool(key) .pool(query) .pool(head)   -> (B,)

    Adapted from IncVGGT's streaming selection (which keeps heads separate for
    per-head top-k against a single current-frame query) to the batch/all-pairs
    setting used here. Heads are collapsed with the same pool so one score is
    produced per frame pair.

    Args:
        q: (B, num_heads, N*hw, head_dim) query tensor after RoPE.
        k: (B, num_heads, N*hw, head_dim) key tensor after RoPE.
        num_frames: N.
        tokens_per_frame: hw (including register tokens).
        patch_start_idx: Number of register/special tokens to skip.
        pool: 'max' (IncVGGT-native) or 'mean' (average pooling).

    Returns:
        scores: (B, N, N) pairwise frame scores.
    """
    assert pool in ("max", "mean"), f"pool must be 'max' or 'mean', got '{pool}'"
    B, num_heads, _, head_dim = q.shape
    NF = num_frames
    hw = tokens_per_frame
    scale = head_dim ** -0.5

    # (B, H, NF, hw, D)
    q_frames = q.reshape(B, num_heads, NF, hw, head_dim)
    k_frames = k.reshape(B, num_heads, NF, hw, head_dim)

    # Drop register tokens — score from patch tokens only
    q_patches = q_frames[:, :, :, patch_start_idx:]  # (B, H, NF, P, D)
    k_patches = k_frames[:, :, :, patch_start_idx:]  # (B, H, NF, P, D)

    scores = torch.zeros(B, NF, NF, device=q.device, dtype=q.dtype)

    def _reduce(t, dim):
        return t.max(dim=dim).values if pool == "max" else t.mean(dim=dim)

    for i in range(NF):
        q_i = q_patches[:, :, i] * scale  # (B, H, P, D)
        for j in range(NF):
            k_j = k_patches[:, :, j]  # (B, H, P, D)
            # (B, H, P, P) raw scaled logits
            logits = q_i @ k_j.transpose(-2, -1)
            # pool over key tokens, then query tokens, then heads -> (B,)
            s = _reduce(_reduce(_reduce(logits, dim=-1), dim=-1), dim=-1)
            scores[:, i, j] = s

    return scores


@torch.no_grad()
def compute_incvggt_max_frame_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_start_idx: int = 0,
) -> torch.Tensor:
    """Max-pooled IncVGGT-style scoring. Thin wrapper around
    compute_incvggt_frame_scores(..., pool='max')."""
    return compute_incvggt_frame_scores(
        q, k, num_frames, tokens_per_frame, patch_start_idx, pool="max",
    )


@torch.no_grad()
def compute_incvggt_mean_frame_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_start_idx: int = 0,
) -> torch.Tensor:
    """Mean-pooled IncVGGT-style scoring. Thin wrapper around
    compute_incvggt_frame_scores(..., pool='mean')."""
    return compute_incvggt_frame_scores(
        q, k, num_frames, tokens_per_frame, patch_start_idx, pool="mean",
    )


def select_topk_from_scores(
    scores: torch.Tensor,
    top_k: int,
    include_self: bool = True,
    candidate_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Select top-K frames per query frame from pairwise scores.

    Args:
        scores: (B, N, N) pairwise frame scores (higher = more relevant).
            Uses the first batch element for selection (shared across batch).
        top_k: Number of frames to select per query frame.
        include_self: If True, always include the query frame itself, selecting
            top-(K-1) others. If False, select top-K purely by score.
        candidate_mask: Optional (N, N) boolean tensor. If provided, only
            consider candidates where mask[i, j] = True for query frame i.

    Returns:
        frame_neighbors: (N, K) indices of selected frames for each query frame.
    """
    B, N, _ = scores.shape
    K = min(top_k, N)

    if K >= N:
        return torch.arange(N, device=scores.device).unsqueeze(0).expand(N, -1)

    # Work with first batch element
    s = scores[0].clone()  # (N, N)

    # Apply candidate mask: set non-candidates to -inf
    if candidate_mask is not None:
        s.masked_fill_(~candidate_mask, float("-inf"))

    if include_self:
        # Mask self to avoid counting it in topk, then add it back
        self_mask = torch.eye(N, device=s.device, dtype=torch.bool)
        s.masked_fill_(self_mask, float("-inf"))

        num_select = K - 1
        _, topk_idx = s.topk(num_select, dim=-1)  # (N, K-1)

        self_idx = torch.arange(N, device=s.device).unsqueeze(1)  # (N, 1)
        frame_neighbors = torch.cat([self_idx, topk_idx], dim=1)  # (N, K)
    else:
        _, topk_idx = s.topk(K, dim=-1)  # (N, K)
        frame_neighbors = topk_idx

    # Sort for consistent ordering
    frame_neighbors, _ = frame_neighbors.sort(dim=1)
    return frame_neighbors


def select_frames_even(
    num_frames: int,
    top_k: int,
    device: torch.device,
    include_self: bool = True,
    candidate_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Select K frames evenly spaced, optionally restricted to candidate frames.

    Args:
        num_frames: N
        top_k: K
        device: torch device
        include_self: If True, always include the query frame itself.
        candidate_mask: Optional (N, N) boolean mask for covisibility filtering.

    Returns:
        frame_neighbors: (N, K) indices of selected frames.
    """
    N = num_frames
    K = min(top_k, N)

    if K >= N:
        return torch.arange(N, device=device).unsqueeze(0).expand(N, -1)

    neighbors = []
    for i in range(N):
        if candidate_mask is not None:
            # Get valid candidates for this query frame
            candidates = torch.where(candidate_mask[i])[0].cpu().numpy().tolist()
        else:
            candidates = list(range(N))

        if include_self and i not in candidates:
            candidates.append(i)
            candidates.sort()

        n_cand = len(candidates)
        k_eff = min(K, n_cand)

        if k_eff >= n_cand:
            selected = candidates[:k_eff]
        else:
            if include_self and i in candidates:
                # Remove self, evenly sample K-1 from rest, then add self back
                others = [c for c in candidates if c != i]
                if len(others) <= k_eff - 1:
                    selected = [i] + others
                else:
                    even_idx = np.linspace(0, len(others) - 1, k_eff - 1).astype(int)
                    selected = [i] + [others[j] for j in even_idx]
            else:
                even_idx = np.linspace(0, n_cand - 1, k_eff).astype(int)
                selected = [candidates[j] for j in even_idx]

        selected = sorted(set(selected))[:K]

        # Pad if needed (shouldn't happen normally)
        while len(selected) < K:
            for c in range(N):
                if c not in selected:
                    selected.append(c)
                    selected.sort()
                    break
            if len(selected) >= K:
                break

        neighbors.append(selected[:K])

    return torch.tensor(neighbors, device=device, dtype=torch.long)


def select_frames_diverse(
    covisibility_matrix: np.ndarray,
    top_k: int,
    device: torch.device,
    include_self: bool = True,
    candidate_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Select K maximally diverse frames per query frame using Farthest Point
    Sampling (FPS) on the covisibility matrix.

    Distance is derived from covisibility: dist[i,j] = max_covis - covis[i,j].
    Higher covisibility ⟹ shorter distance ⟹ more overlap ⟹ less diverse.

    For every query frame i, FPS is run the same way: min_dist starts at +inf
    so the first argmax pick is index 0 (or the first non-excluded index), and
    subsequent picks greedily maximize the minimum distance to the already-
    selected set. This means the FPS-derived neighbor set is the *same* across
    query frames (modulo per-row candidate masks).

    When include_self=True, after FPS we ensure the query frame itself is in
    the result by swapping the most-recently added frame for i (if i is not
    already selected). This preserves the include_self=False behavior bit-for-
    bit and adds self-inclusion as a post-hoc edit.

    Args:
        covisibility_matrix: (N, N) array where larger values = more overlap.
        top_k: K frames to select per query frame.
        device: torch device.
        include_self: If True, force the query frame into the result by
            replacing the last FPS-picked frame with i when i is missing.
        candidate_mask: Optional (N, N) boolean tensor. If provided, only
            consider candidates where mask[i, j] = True for query frame i.

    Returns:
        frame_neighbors: (N, K) tensor of selected frame indices (sorted).
    """
    N = covisibility_matrix.shape[0]
    K = min(top_k, N)

    if K >= N:
        return torch.arange(N, device=device).unsqueeze(0).expand(N, -1)

    covis = covisibility_matrix.astype(np.float64)
    # Convert similarity → distance: higher covisibility = shorter distance
    max_covis = covis.max()
    dist = max_covis - covis

    # Pre-compute per-query excluded mask from candidate_mask
    if candidate_mask is not None:
        excluded = ~candidate_mask.cpu().numpy()  # (N, N) bool
    else:
        excluded = None

    neighbors = []
    for i in range(N):
        # FPS — identical seeding regardless of include_self so the
        # include_self=False code path is preserved exactly.
        selected = []
        min_dist = np.full(N, np.inf)
        k_remaining = K

        # Mask out non-candidate frames so they are never picked
        if excluded is not None:
            min_dist[excluded[i]] = -np.inf

        for _ in range(k_remaining):
            # Pick the frame farthest from the selected set
            best = int(np.argmax(min_dist))
            if min_dist[best] == -np.inf:
                break  # all candidates exhausted
            selected.append(best)
            min_dist[best] = -np.inf  # mark as selected
            # Update: for each remaining frame, track its closest selected neighbour
            np.minimum(min_dist, dist[best], out=min_dist)
            # Re-mark already-selected and excluded frames
            for s in selected:
                min_dist[s] = -np.inf
            if excluded is not None:
                min_dist[excluded[i]] = -np.inf

        # Force self into the set by swapping out the last-added frame.
        if include_self and i not in selected:
            if len(selected) > 0:
                selected[-1] = i
            else:
                selected.append(i)

        selected = sorted(selected)[:K]
        neighbors.append(selected)

    return torch.tensor(neighbors, device=device, dtype=torch.long)


def select_frames_diverse_self(
    covisibility_matrix: np.ndarray,
    top_k: int,
    device: torch.device,
    include_self: bool = True,
    candidate_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Per-query Farthest Point Sampling (FPS) seeded by the query frame itself.

    Unlike `select_frames_diverse`, which seeds FPS the same way for every
    query (so the neighbor set is shared across queries up to a per-row mask),
    this strategy uses query frame i as the FPS seed. Distance from i drives
    the first pick (the frame least covisible with i), and subsequent picks
    greedily maximize the minimum distance to {i} ∪ already-selected, yielding
    a query-dependent set of maximally diverse-from-i frames.

    Distance is derived from covisibility: dist[i,j] = max_covis - covis[i,j],
    matching `select_frames_diverse`. Higher covisibility ⟹ shorter distance.

    Args:
        covisibility_matrix: (N, N) array where larger values = more overlap.
        top_k: K frames to select per query frame.
        device: torch device.
        include_self: If True (default), the K outputs include i itself
            (i is the first FPS pick, plus K-1 farthest-from-i picks). If
            False, i seeds FPS but is excluded from the output, so the K
            outputs are the K most-diverse-from-i non-self frames.
        candidate_mask: Optional (N, N) boolean tensor. If provided, only
            consider candidates where mask[i, j] = True for query frame i.

    Returns:
        frame_neighbors: (N, K) tensor of selected frame indices (sorted).
    """
    N = covisibility_matrix.shape[0]
    K = min(top_k, N)

    if K >= N:
        return torch.arange(N, device=device).unsqueeze(0).expand(N, -1)

    covis = covisibility_matrix.astype(np.float64)
    max_covis = covis.max()
    dist = max_covis - covis  # (N, N), higher = less covisible = more diverse

    if candidate_mask is not None:
        excluded = ~candidate_mask.cpu().numpy()  # (N, N) bool
    else:
        excluded = None

    neighbors = []
    for i in range(N):
        # Seed FPS with i: min_dist starts as the row of distances from i.
        # Marking min_dist[i] = -inf keeps i as a perpetual anchor (it
        # influences distances but is never picked again).
        min_dist = dist[i].copy()
        min_dist[i] = -np.inf
        if excluded is not None:
            min_dist[excluded[i]] = -np.inf

        if include_self:
            selected = [i]
            k_remaining = K - 1
        else:
            selected = []
            k_remaining = K

        for _ in range(k_remaining):
            best = int(np.argmax(min_dist))
            if min_dist[best] == -np.inf:
                break  # all candidates exhausted
            selected.append(best)
            min_dist[best] = -np.inf
            np.minimum(min_dist, dist[best], out=min_dist)
            # Re-mark anchors (i and previously-selected) and excluded frames
            min_dist[i] = -np.inf
            for s in selected:
                min_dist[s] = -np.inf
            if excluded is not None:
                min_dist[excluded[i]] = -np.inf

        selected = sorted(selected)[:K]
        neighbors.append(selected)

    return torch.tensor(neighbors, device=device, dtype=torch.long)


def select_frames_random(
    num_frames: int,
    top_k: int,
    device: torch.device,
    include_self: bool = True,
    candidate_mask: torch.Tensor = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Select K random frames per query frame.

    Args:
        num_frames: N
        top_k: K
        device: torch device
        include_self: If True, always include the query frame itself.
        candidate_mask: Optional (N, N) boolean mask for covisibility filtering.
        seed: Optional random seed for reproducibility.

    Returns:
        frame_neighbors: (N, K) indices of selected frames.
    """
    N = num_frames
    K = min(top_k, N)

    if K >= N:
        return torch.arange(N, device=device).unsqueeze(0).expand(N, -1)

    rng = _random.Random(seed)

    neighbors = []
    for i in range(N):
        if candidate_mask is not None:
            candidates = torch.where(candidate_mask[i])[0].cpu().numpy().tolist()
        else:
            candidates = list(range(N))

        if include_self:
            others = [c for c in candidates if c != i]
            k_sample = min(K - 1, len(others))
            selected = [i] + rng.sample(others, k_sample)
        else:
            k_sample = min(K, len(candidates))
            selected = rng.sample(candidates, k_sample)

        selected = sorted(selected)[:K]

        # Pad if needed
        while len(selected) < K:
            for c in range(N):
                if c not in selected:
                    selected.append(c)
                    selected.sort()
                    break
            if len(selected) >= K:
                break

        neighbors.append(selected[:K])

    return torch.tensor(neighbors, device=device, dtype=torch.long)


def select_frames_closest(
    num_frames: int,
    top_k: int,
    device: torch.device,
    include_self: bool = True,
    candidate_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Select K frames closest in index to the query frame.

    For each query frame i:
      - Prefer K/2 frames before and K/2 after (split evenly).
      - If one side has fewer available candidates than requested, compensate
        by taking additional frames from the other side.
      - When include_self=True, frame i itself is always included, and the
        remaining K-1 slots are split (K-1)//2 before and the rest after.
      - When include_self=False, all K slots are filled from non-self frames
        (so frame 0 gets frames 1..K, frame N-1 gets frames N-1-K..N-2, etc.).

    Args:
        num_frames: N
        top_k: K
        device: torch device.
        include_self: If True, always include the query frame itself.
        candidate_mask: Optional (N, N) boolean mask restricting which frames
            are eligible to be picked as neighbors for each query frame.

    Returns:
        frame_neighbors: (N, K) tensor of selected frame indices (sorted).
    """
    N = num_frames
    K = min(top_k, N)

    if K >= N:
        return torch.arange(N, device=device).unsqueeze(0).expand(N, -1)

    neighbors = []
    for i in range(N):
        if candidate_mask is not None:
            cand_set = set(
                torch.where(candidate_mask[i])[0].cpu().numpy().tolist()
            )
        else:
            cand_set = set(range(N))

        selected = []
        if include_self:
            selected.append(i)
            k_remaining = K - 1
        else:
            cand_set.discard(i)
            k_remaining = K

        # Ensure we don't re-pick frames already in `selected`
        for s in selected:
            cand_set.discard(s)

        # Closest-first ordering on each side
        before_cands = sorted([c for c in cand_set if c < i], reverse=True)
        after_cands = sorted([c for c in cand_set if c > i])

        half_before = k_remaining // 2
        half_after = k_remaining - half_before

        take_before = min(half_before, len(before_cands))
        take_after = min(half_after, len(after_cands))

        # If one side is short, compensate from the other side.
        deficit = (half_before - take_before) + (half_after - take_after)
        if deficit > 0:
            extra_after = min(deficit, len(after_cands) - take_after)
            take_after += extra_after
            deficit -= extra_after
        if deficit > 0:
            extra_before = min(deficit, len(before_cands) - take_before)
            take_before += extra_before
            deficit -= extra_before

        selected.extend(before_cands[:take_before])
        selected.extend(after_cands[:take_after])

        selected = sorted(set(selected))[:K]

        # Pad if still short (e.g. candidate_mask was very restrictive).
        if len(selected) < K:
            # Fall back to closest-by-|j - i| over *all* frames, ignoring the
            # candidate mask, to guarantee a (N, K) output of valid indices.
            by_dist = sorted(
                (j for j in range(N) if j not in selected),
                key=lambda j: (abs(j - i), j),
            )
            for c in by_dist:
                selected.append(c)
                if len(selected) >= K:
                    break
            selected = sorted(selected)[:K]

        neighbors.append(selected[:K])

    return torch.tensor(neighbors, device=device, dtype=torch.long)


@torch.no_grad()
def select_tokens_diverse(
    k_frames: torch.Tensor,
    keep_ratio: float,
    patch_start_idx: int = 0,
) -> list:
    """
    Cross-frame diverse token selection using a two-pass approach:
      1. Run per-frame FPS to get candidates (same count per frame).
      2. Score each candidate by its max cosine similarity to candidates in
         OTHER frames.  Tokens that are redundant cross-frame (high similarity
         to another frame's selection) are dropped first.

    This keeps tokens that are *unique* to their frame — maximising diversity
    across the full cross-frame context — while staying memory-efficient
    (never materialises an (NF*P)^2 distance matrix).

    Args:
        k_frames: (B, num_heads, NF, hw, head_dim) key tensor reshaped per frame.
        keep_ratio: Fraction of *patch* tokens to keep per frame, in (0, 1].
        patch_start_idx: Number of register/special tokens at the start of each
            frame (always kept, not counted toward keep_ratio).

    Returns:
        List of NF tensors, each of shape (num_keep_f,) containing hw-dimension
        indices to keep for that frame. num_keep_f may differ across frames.
    """
    B, num_heads, NF, hw, head_dim = k_frames.shape
    num_patches = hw - patch_start_idx
    num_keep_per_frame = max(1, int(round(keep_ratio * num_patches)))

    if num_keep_per_frame >= num_patches:
        all_idx = torch.arange(hw, device=k_frames.device)
        return [all_idx] * NF

    # ── Pass 1: per-frame FPS to get 2x candidates per frame ────────────
    # Over-select so we have room to prune cross-frame redundancies.
    num_candidates = min(num_patches, num_keep_per_frame * 2)

    # (NF, hw, C) from batch element 0
    feats = k_frames[0].permute(1, 2, 0, 3).reshape(NF, hw, num_heads * head_dim)
    patch_feats = feats[:, patch_start_idx:]  # (NF, num_patches, C)

    candidate_indices = _fps_batched(patch_feats, num_candidates)  # (NF, num_candidates)

    # ── Pass 2: cross-frame redundancy pruning ──────────────────────────
    # For each frame's candidates, compute max similarity to every OTHER
    # frame's candidates.  Drop the most redundant ones first.

    # Gather candidate features: (NF, num_candidates, C)
    cand_feats = torch.gather(
        patch_feats,
        1,
        candidate_indices.unsqueeze(-1).expand(-1, -1, patch_feats.shape[-1]),
    )
    cand_feats_norm = torch.nn.functional.normalize(cand_feats, dim=-1, p=2)

    # Cross-frame redundancy: for each candidate, its max cosine similarity
    # to any candidate in any OTHER frame.
    # We use frame-level mean representations to avoid the O(NF^2 * cand^2) cost.
    # For each frame j, compute its mean candidate feature, then for each
    # candidate in frame i, compute similarity to every other frame's mean.
    # This is O(NF * cand * NF) = O(NF^2 * cand), much more efficient.

    frame_means = cand_feats_norm.mean(dim=1)  # (NF, C)
    frame_means = torch.nn.functional.normalize(frame_means, dim=-1, p=2)

    # (NF, cand, C) @ (C, NF) -> (NF, cand, NF)
    sim_to_means = torch.matmul(
        cand_feats_norm,  # (NF, cand, C)
        frame_means.T,    # (C, NF)
    )  # (NF, cand, NF)

    # Mask self-frame
    for i in range(NF):
        sim_to_means[i, :, i] = -float("inf")

    # Cross-frame redundancy score: max similarity to any other frame's mean
    redundancy = sim_to_means.max(dim=-1).values  # (NF, num_candidates)

    # Keep the num_keep_per_frame least redundant (most unique) candidates per frame
    # Lower redundancy = more unique = keep
    _, keep_order = redundancy.sort(dim=-1)  # ascending: least redundant first
    keep_local = keep_order[:, :num_keep_per_frame]  # (NF, num_keep_per_frame)

    # Map back to hw-space indices
    # candidate_indices[f, keep_local[f]] gives patch-relative indices
    selected = torch.gather(candidate_indices, 1, keep_local)  # (NF, num_keep_per_frame)
    selected = selected + patch_start_idx  # shift to hw-space
    selected, _ = selected.sort(dim=-1)

    reg_indices = torch.arange(patch_start_idx, device=k_frames.device, dtype=torch.long)
    per_frame_indices = []
    for f in range(NF):
        per_frame_indices.append(torch.cat([reg_indices, selected[f]]))

    return per_frame_indices


@torch.no_grad()
def select_tokens_activation(
    k_frames: torch.Tensor,
    keep_ratio: float,
    patch_start_idx: int = 0,
) -> list:
    """
    Cross-frame activation-based token selection using a two-pass approach:
      1. Per-frame top-k by L2 norm to get 2x candidates per frame.
      2. Score each candidate by its max cosine similarity to candidates in
         OTHER frames.  Tokens that are redundant cross-frame are dropped first.

    Args:
        k_frames: (B, num_heads, NF, hw, head_dim) key tensor reshaped per frame.
        keep_ratio: Fraction of *patch* tokens to keep per frame, in (0, 1].
        patch_start_idx: Number of register/special tokens at the start of each
            frame (always kept, not counted toward keep_ratio).

    Returns:
        List of NF tensors, each of shape (num_keep,) containing hw-dimension
        indices to keep for that frame.
    """
    B, num_heads, NF, hw, head_dim = k_frames.shape
    num_patches = hw - patch_start_idx
    num_keep_per_frame = max(1, int(round(keep_ratio * num_patches)))

    if num_keep_per_frame >= num_patches:
        all_idx = torch.arange(hw, device=k_frames.device)
        return [all_idx] * NF

    # ── Pass 1: per-frame top-k by L2 norm to get 2x candidates ────────
    num_candidates = min(num_patches, num_keep_per_frame * 2)

    # (NF, hw, C) from batch element 0
    feats = k_frames[0].permute(1, 2, 0, 3).reshape(NF, hw, num_heads * head_dim)
    patch_feats = feats[:, patch_start_idx:]  # (NF, num_patches, C)

    # Per-frame L2 norms
    patch_norms = patch_feats.norm(dim=-1)  # (NF, num_patches)
    _, topk_idx = patch_norms.topk(num_candidates, dim=-1)  # (NF, num_candidates)
    candidate_indices = topk_idx.sort(dim=-1).values  # (NF, num_candidates)

    # ── Pass 2: cross-frame redundancy pruning ──────────────────────────
    # Gather candidate features: (NF, num_candidates, C)
    cand_feats = torch.gather(
        patch_feats,
        1,
        candidate_indices.unsqueeze(-1).expand(-1, -1, patch_feats.shape[-1]),
    )
    cand_feats_norm = torch.nn.functional.normalize(cand_feats, dim=-1, p=2)

    frame_means = cand_feats_norm.mean(dim=1)  # (NF, C)
    frame_means = torch.nn.functional.normalize(frame_means, dim=-1, p=2)

    # (NF, cand, C) @ (C, NF) -> (NF, cand, NF)
    sim_to_means = torch.matmul(
        cand_feats_norm,  # (NF, cand, C)
        frame_means.T,    # (C, NF)
    )  # (NF, cand, NF)

    # Mask self-frame
    for i in range(NF):
        sim_to_means[i, :, i] = -float("inf")

    # Cross-frame redundancy score: max similarity to any other frame's mean
    redundancy = sim_to_means.max(dim=-1).values  # (NF, num_candidates)

    # Keep the least redundant candidates per frame
    _, keep_order = redundancy.sort(dim=-1)  # ascending: least redundant first
    keep_local = keep_order[:, :num_keep_per_frame]  # (NF, num_keep_per_frame)

    # Map back to hw-space indices
    selected = torch.gather(candidate_indices, 1, keep_local)  # (NF, num_keep_per_frame)
    selected = selected + patch_start_idx  # shift to hw-space
    selected, _ = selected.sort(dim=-1)

    reg_indices = torch.arange(patch_start_idx, device=k_frames.device, dtype=torch.long)
    per_frame_indices = []
    for f in range(NF):
        per_frame_indices.append(torch.cat([reg_indices, selected[f]]))

    return per_frame_indices


@torch.no_grad()
def per_token_topk_attention(
    q: torch.Tensor,
    k_frames: torch.Tensor,
    v_frames: torch.Tensor,
    keep_ratio: float,
    patch_start_idx: int = 0,
) -> torch.Tensor:
    """
    Per-token top-k attention: compute full Q@K^T, then for each query token,
    within each key frame, keep only the top-k key tokens by attention score.
    Register/special tokens are always kept.

    This is the most accurate token selection since it uses the actual attention
    scores (Q@K^T) rather than proxies like key norms. However, it requires
    computing the full attention logits first.

    Args:
        q: (B, heads, hw_q, dim) query tensor for one frame.
        k_frames: (B, heads, NF_ctx, hw_k, dim) key tensor for context frames.
        v_frames: (B, heads, NF_ctx, hw_k, dim) value tensor for context frames.
        keep_ratio: Fraction of *patch* tokens to keep per key frame, in (0, 1].
        patch_start_idx: Number of register/special tokens at the start of each
            frame (always kept, not counted toward keep_ratio).

    Returns:
        out: (B, heads, hw_q, dim) attention output.
    """
    B, heads, hw_q, dim = q.shape
    NF_ctx = k_frames.shape[2]
    hw_k = k_frames.shape[3]
    psi = patch_start_idx
    num_patches = hw_k - psi
    n_keep = max(1, int(round(keep_ratio * num_patches)))

    # Flatten context frames: (B, heads, NF_ctx * hw_k, dim)
    k_flat = k_frames.reshape(B, heads, NF_ctx * hw_k, dim)
    v_flat = v_frames.reshape(B, heads, NF_ctx * hw_k, dim)

    scale = dim ** -0.5

    if n_keep >= num_patches:
        # No selection needed, standard attention
        logits = (q @ k_flat.transpose(-2, -1)) * scale
        attn = logits.softmax(dim=-1)
        return attn @ v_flat

    # Full logits: (B, heads, hw_q, NF_ctx * hw_k) — native dtype (e.g. bf16)
    logits = (q @ k_flat.transpose(-2, -1)) * scale

    # Reshape to per-frame view: (B, heads, hw_q, NF_ctx, hw_k)
    logits_pf = logits.reshape(B, heads, hw_q, NF_ctx, hw_k)

    # Slice out patch token logits (view into logits, no copy)
    patch_logits = logits_pf[:, :, :, :, psi:]  # (B, heads, hw_q, NF_ctx, num_patches)

    # Top-k selection per query token per key frame
    topk_vals, topk_idx = patch_logits.topk(n_keep, dim=-1)
    # (B, heads, hw_q, NF_ctx, n_keep) each

    # In-place mask: set all patch logits to -inf, then restore top-k
    patch_logits.fill_(float('-inf'))
    patch_logits.scatter_(-1, topk_idx, topk_vals)

    # Registers (0:psi) are untouched — they keep original logit values
    # Flatten back and compute attention
    logits = logits_pf.reshape(B, heads, hw_q, NF_ctx * hw_k)
    attn = logits.softmax(dim=-1)
    return attn @ v_flat


def _kmeanspp_batched(patch_feats_batch: torch.Tensor, num_keep: int) -> torch.Tensor:
    """Batched K-means++ initialization on GPU — memory-efficient FPS alternative.

    Instead of precomputing the full (NF, P, P) pairwise distance matrix,
    computes distances lazily from each newly selected point.  Produces the
    same "maximally spread" point set as FPS.

    Complexity: O(K * NF * P * C)  — no P^2 term.
    Memory:     O(NF * P)          — no (NF, P, P) matrix.

    Args:
        patch_feats_batch: (NF, num_patches, C) — already on device.
        num_keep: number of patches to select per frame.

    Returns:
        (NF, num_keep) long tensor of patch-relative indices.
    """
    NF, N, C = patch_feats_batch.shape
    device = patch_feats_batch.device

    # Normalize for cosine distance
    feats = torch.nn.functional.normalize(patch_feats_batch, dim=-1, p=2)
    feats_t = feats.transpose(1, 2).contiguous()  # (NF, C, N) — cached for bmm

    # Seed: highest-norm token per frame
    raw_norms = patch_feats_batch.norm(dim=-1)  # (NF, N)
    seeds = raw_norms.argmax(dim=-1)  # (NF,)

    selected = torch.empty(NF, num_keep, dtype=torch.long, device=device)
    selected[:, 0] = seeds

    # Compute initial distances from seed to all tokens via bmm: O(NF * N * C)
    # Gather seed features: (NF, 1, C)
    seed_feats = feats[torch.arange(NF, device=device), seeds].unsqueeze(1)
    # bmm: (NF, 1, C) @ (NF, C, N) -> (NF, 1, N) -> squeeze -> (NF, N)
    min_dist = 2.0 - 2.0 * torch.bmm(seed_feats, feats_t).squeeze(1)

    # Mark seeds as taken
    min_dist[torch.arange(NF, device=device), seeds] = -float("inf")

    for i in range(1, num_keep):
        # Farthest point — all on GPU, no .item()
        best = min_dist.argmax(dim=-1)  # (NF,)
        selected[:, i] = best

        # Mark selected
        min_dist[torch.arange(NF, device=device), best] = -float("inf")

        # Distance from new point to all tokens via bmm: O(NF * N * C)
        best_feats = feats[torch.arange(NF, device=device), best].unsqueeze(1)
        new_dist = 2.0 - 2.0 * torch.bmm(best_feats, feats_t).squeeze(1)

        # Update min distances
        torch.minimum(min_dist, new_dist, out=min_dist)

        # Re-mask selected point
        min_dist[torch.arange(NF, device=device), best] = -float("inf")

    # Sort indices within each frame for deterministic ordering
    selected, _ = selected.sort(dim=-1)
    return selected


def _fps_batched(patch_feats_batch: torch.Tensor, num_keep: int) -> torch.Tensor:
    """Batched FPS on GPU without per-iteration CPU syncs.

    Args:
        patch_feats_batch: (NF, num_patches, C) — already on device.
        num_keep: number of patches to select per frame.

    Returns:
        (NF, num_keep) long tensor of patch-relative indices.
    """
    NF, N, C = patch_feats_batch.shape
    device = patch_feats_batch.device

    # Normalize and compute full pairwise distance in one batched matmul
    feats = torch.nn.functional.normalize(patch_feats_batch, dim=-1, p=2)
    dist = 2.0 - 2.0 * torch.bmm(feats, feats.transpose(1, 2))  # (NF, N, N)

    # Seed: highest-norm token per frame
    raw_norms = patch_feats_batch.norm(dim=-1)  # (NF, N)
    seeds = raw_norms.argmax(dim=-1)  # (NF,)

    # Gather seed row from dist: (NF, N)
    min_dist = dist[torch.arange(NF, device=device), seeds]  # (NF, N)

    selected = torch.empty(NF, num_keep, dtype=torch.long, device=device)
    selected[:, 0] = seeds

    # Mark seeds as taken
    min_dist[torch.arange(NF, device=device), seeds] = -float("inf")

    for i in range(1, num_keep):
        # Farthest point — all on GPU, no .item()
        best = min_dist.argmax(dim=-1)  # (NF,)
        selected[:, i] = best

        # Mark selected
        min_dist[torch.arange(NF, device=device), best] = -float("inf")

        # Update min distances
        new_dist = dist[torch.arange(NF, device=device), best]  # (NF, N)
        torch.minimum(min_dist, new_dist, out=min_dist)

        # Re-mask all selected so far (only the newly selected need it,
        # but the single scatter is cheaper than tracking)
        min_dist[torch.arange(NF, device=device), best] = -float("inf")

    # Sort indices within each frame for deterministic ordering
    selected, _ = selected.sort(dim=-1)
    return selected


@torch.no_grad()
def select_tokens_per_frame_diverse(
    k_frames: torch.Tensor,
    keep_ratio: float,
    patch_start_idx: int = 0,
) -> list:
    """
    Per-frame diverse token selection using FPS on key features.
    Each frame independently selects its own set of tokens.

    Args:
        k_frames: (B, num_heads, NF, hw, head_dim) key tensor reshaped per frame.
        keep_ratio: Fraction of *patch* tokens to keep per frame, in (0, 1].
        patch_start_idx: Number of register/special tokens at the start of each
            frame (always kept, not counted toward keep_ratio).

    Returns:
        List of NF tensors, each of shape (num_keep,) containing hw-dimension
        indices to keep for that frame. num_keep is the same across frames.
    """
    B, num_heads, NF, hw, head_dim = k_frames.shape
    num_patches = hw - patch_start_idx
    num_keep_patches = max(1, int(round(keep_ratio * num_patches)))

    if num_keep_patches >= num_patches:
        all_idx = torch.arange(hw, device=k_frames.device)
        return [all_idx] * NF

    # (NF, num_patches, num_heads * head_dim) from batch element 0
    feats = k_frames[0].permute(1, 2, 0, 3).reshape(NF, hw, num_heads * head_dim)
    patch_feats = feats[:, patch_start_idx:]  # (NF, num_patches, C)

    # Batched FPS — all frames in parallel, no CPU syncs
    selected = _fps_batched(patch_feats, num_keep_patches)  # (NF, num_keep_patches)
    selected = selected + patch_start_idx  # shift to hw-space indices

    reg_indices = torch.arange(patch_start_idx, device=k_frames.device, dtype=torch.long)

    per_frame_indices = []
    for f in range(NF):
        per_frame_indices.append(torch.cat([reg_indices, selected[f]]))

    return per_frame_indices


@torch.no_grad()
def select_tokens_per_frame_activation(
    k_frames: torch.Tensor,
    keep_ratio: float,
    patch_start_idx: int = 0,
) -> list:
    """
    Per-frame activation-based token selection. Each frame independently
    keeps the tokens with the largest L2 norm of key features.

    Args:
        k_frames: (B, num_heads, NF, hw, head_dim) key tensor reshaped per frame.
        keep_ratio: Fraction of *patch* tokens to keep per frame, in (0, 1].
        patch_start_idx: Number of register/special tokens at the start of each
            frame (always kept, not counted toward keep_ratio).

    Returns:
        List of NF tensors, each of shape (num_keep,) containing hw-dimension
        indices to keep for that frame. num_keep is the same across frames.
    """
    B, num_heads, NF, hw, head_dim = k_frames.shape
    assert B == 1, (
        f"select_tokens_per_frame_activation uses batch element 0 to choose tokens, "
        f"but activation patterns are content-dependent so this is unsafe for B>1. "
        f"Got B={B}."
    )
    num_patches = hw - patch_start_idx
    num_keep_patches = max(1, int(round(keep_ratio * num_patches)))

    if num_keep_patches >= num_patches:
        all_idx = torch.arange(hw, device=k_frames.device)
        return [all_idx] * NF

    # (NF, hw, num_heads * head_dim) from batch element 0
    feats = k_frames[0].permute(1, 2, 0, 3).reshape(NF, hw, num_heads * head_dim)
    reg_indices = torch.arange(patch_start_idx, device=k_frames.device, dtype=torch.long)

    per_frame_indices = []
    for f in range(NF):
        patch_norms = feats[f, patch_start_idx:].norm(dim=-1)  # (num_patches,)
        _, topk_idx = patch_norms.topk(num_keep_patches)
        topk_idx_sorted = topk_idx.sort().values
        patch_indices = topk_idx_sorted + patch_start_idx
        per_frame_indices.append(torch.cat([reg_indices, patch_indices]))

    return per_frame_indices


class FrameSelector:
    """
    Configuration for frame selection in Pi3 cross-view attention.

    Strategies:
    - 'topk': Per-layer top-K selection using max-pooled P×P cosine-similarity
              between Q and K (heads concatenated). Computed independently at
              each cross-view layer.
    - 'topk_mean': Same scoring as 'topk' but with mean-pooling over the P×P
                   similarity map instead of max.
    - 'incvggt_max': Per-layer top-K selection using raw scaled Q@K^T logits
                    (IncVGGT-style), max-pooled over key tokens, query tokens,
                    and heads. Computed independently at each cross-view layer.
    - 'incvggt_mean': Same scoring as 'incvggt_max' but with mean-pooling over
                    tokens and heads instead of max-pooling.
    - 'even': Precomputed uniform K-frame sampling (same for all layers).
    - 'diverse': Farthest Point Sampling on the covisibility matrix.
                 Selects K frames that are maximally different from each other
                 (minimum pairwise covisibility). Requires covisibility_matrix.
    - 'random': Random K-frame sampling per query frame (same for all layers).
    - 'closest': Pick the K frames closest in index to the query frame,
                 preferring K/2 before and K/2 after (compensating from the
                 other side when one side runs out).

    Covisibility pre-filter (applies to topk, even, random, and closest):
        When use_covisibility=True, a candidate mask is computed from the
        covisibility matrix, restricting selection to the top-X% of frames
        with shared 3D points per query frame.
    """

    def __init__(
        self,
        strategy: str = "topk",
        top_k: int = 10,
        include_self: bool = True,
        include_first: bool = False,
        token_downsample: int = 1,
        token_keep_rate: float = 1.0,
        token_selection_method: str = "diverse",
        token_ds_layers: list = None,
        token_ds_entropy_threshold: float = None,
        ds_keep_register: bool = False,
        use_covisibility: bool = False,
        covisibility_matrix: Optional[np.ndarray] = None,
        covisibility_percentile: float = 50.0,
        batched_sdpa: bool = False,
        global_as_frame_layers: set = None,
        global_as_meanpool_layers: set = None,
    ):
        """
        Args:
            strategy: 'topk', 'even', 'diverse', or 'random'
            top_k: Number of frames each query frame attends to.
            include_self: Whether to always include the query frame itself.
            include_first: Whether to always include frame 0 in the selection.
            token_downsample: Spatial stride factor for downsampling K/V tokens
                from selected neighbor frames. Default 1 (no downsampling).
            token_keep_rate: Fraction of patch tokens to keep per frame via
                diverse (FPS) token selection. In (0, 1]. Default 1.0 (keep all).
                Mutually exclusive with token_downsample > 1.
            ds_keep_register: If True, preserve register/camera tokens during
                K/V downsampling. If False (default), include them in the
                spatial stride grid.
            global_as_frame_layers: Set of global layer indices (0-indexed) to
                convert to frame attention. These layers skip the (B*S,P,C)->
                (B,S*P,C) rearrangement and run attention per-frame using the
                global block's weights, reducing cost from O((NL)^2) to O(NL^2).
            use_covisibility: If True, pre-filter candidates using covisibility
                (for topk/even). For 'diverse', the matrix is always used.
            covisibility_matrix: (N, N) array of shared 3D point counts.
                Required when use_covisibility=True or strategy='diverse'.
            covisibility_percentile: Keep the top-X% of frames by covisibility.
            batched_sdpa: If True, gather K/V for all frames at once and use a
                single batched SDPA call (higher memory, but eliminates bf16
                rounding divergence from per-frame loop). Default False.
        """
        assert strategy in ("topk", "topk_mean", "incvggt_max", "incvggt_mean", "even", "diverse", "diverse_self", "random", "closest"), (
            f"strategy must be 'topk', 'topk_mean', 'incvggt_max', 'incvggt_mean', 'even', 'diverse', 'diverse_self', 'random', or 'closest', got '{strategy}'"
        )
        if strategy in ("diverse", "diverse_self"):
            assert covisibility_matrix is not None, (
                f"covisibility_matrix is required for strategy='{strategy}'"
            )
        if use_covisibility:
            assert covisibility_matrix is not None, (
                "covisibility_matrix is required when use_covisibility=True"
            )
        if token_keep_rate < 1.0 and token_downsample > 1:
            raise ValueError(
                "Both token_keep_rate (<1.0) and token_downsample (>1) are set. "
                "These are mutually exclusive: use token_keep_rate for diverse FPS "
                "token selection, OR token_downsample for uniform spatial stride, "
                "but not both."
            )
        _valid_methods = ("diverse", "activation", "per_frame_diverse", "per_frame_activation", "per_token_activation")
        assert token_selection_method in _valid_methods, (
            f"token_selection_method must be one of {_valid_methods}, got '{token_selection_method}'"
        )
        self.strategy = strategy
        self.top_k = top_k
        self.include_self = include_self
        self.include_first = include_first
        self.token_downsample = token_downsample
        self.token_keep_rate = token_keep_rate
        self.token_selection_method = token_selection_method
        self.token_ds_layers = set(token_ds_layers) if token_ds_layers is not None else None
        self.token_ds_entropy_threshold = token_ds_entropy_threshold
        if token_ds_entropy_threshold is not None and token_ds_layers is not None:
            import logging
            logging.warning(
                "Both token_ds_entropy_threshold and token_ds_layers are set. "
                "token_ds_entropy_threshold takes priority (token_ds_layers will be ignored)."
            )
        self.ds_keep_register = ds_keep_register
        self.use_covisibility = use_covisibility
        self.covisibility_matrix = covisibility_matrix
        self.covisibility_percentile = covisibility_percentile
        self.batched_sdpa = batched_sdpa
        self.global_as_frame_layers = global_as_frame_layers
        self.global_as_meanpool_layers = global_as_meanpool_layers


def ensure_first_frame_included(frame_neighbors: torch.Tensor) -> torch.Tensor:
    """
    Ensure frame 0 is included in every query frame's neighbor set.
    For rows where frame 0 is missing, replaces the last neighbor with 0 and re-sorts.

    Args:
        frame_neighbors: (N, K) tensor of frame indices.

    Returns:
        (N, K) tensor with frame 0 guaranteed in each row.
    """
    has_first = (frame_neighbors == 0).any(dim=1)  # (N,)
    if has_first.all():
        return frame_neighbors
    result = frame_neighbors.clone()
    missing = ~has_first
    result[missing, -1] = 0
    result, _ = result.sort(dim=1)
    return result
