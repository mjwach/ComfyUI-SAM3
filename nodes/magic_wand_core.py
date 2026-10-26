"""
Magic-wand region growing for image batches (treated as video frames).

Pure torch, no ComfyUI imports, so it can be exercised on its own.

Overview
--------
For each seed point (on frame 0) a per-pixel *match strength* map is built
from the distance between every pixel's chosen color quality and the seed's:

    d <= ft            -> strength 1          (full match)
    ft < d < ft + pt   -> strength falls linearly from 1 to 0 (partial match)
    otherwise          -> strength 0

where ft = full_tolerance / 100 and pt = partial_tolerance / 100, and d is
normalized so that 0 means identical and 1 means "as different as this
quality can possibly be".

The selection then grows from the seed through 6-connected neighbours
(left/right, up/down, previous/next frame).  A move from pixel p to an
adjacent pixel q is allowed when

    strength[q] > 0  and  (p is "free"  or  strength[q] <= strength[p])

where p is free if it is a full match or its strength exceeds the growth
threshold.  Pixels of zero strength are never entered, so they never spread.

Growing is done with whole-tensor "run sweeps" instead of a pixel-by-pixel
flood: along one axis, every maximal run of consecutive allowed moves is
found with a cumulative sum, and any reached pixel floods the rest of its
run in one cummax.  Sweeping x, y and t in both directions and repeating
until nothing changes needs only as many rounds as the region has "turns",
which is usually a handful, and every sweep is a few vectorized ops that run
equally well on GPU or CPU.
"""

import torch

QUALITIES = ["color", "red", "green", "blue", "hue", "chroma", "lightness", "chrominance", "frame change"]
BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64]
BLOCK_REDUCTIONS = ["max", "average", "min"]

# Pixel values are usually k/255, so anything far below 1/255 is safe.  This
# only keeps a value that sits exactly on a tolerance boundary (e.g. seed 0.5,
# FT 15, pixel 0.65) from being lost to float rounding.
_EPS = 1e-5

# Rough number of elements processed at once by the sweeps, to bound the
# temporary memory (int32 segment ids + cummax results) for long videos.
_CHUNK_ELEMS = 1 << 24


# ---------------------------------------------------------------------------
# Color qualities
# ---------------------------------------------------------------------------
#
# Every quality is a per-pixel value (scalar or small vector) plus a distance
# scaled so that 0 means identical and 1 means as different as that quality
# can be for colors in the sRGB gamut, which is what makes tolerance 100
# always reach everything.

# Largest distances between sRGB colors, found by brute force over the sRGB
# cube: OKLab distance peaks at black<->white (1.0), chroma at pure magenta,
# and lightness-relative chrominance (below) at about 0.988.
_OKLAB_MAX_DIST = 1.0
_OKLAB_MAX_CHROMA = 0.3225
_MAX_REL_CHROMINANCE_DIST = 0.9881

# Chrominance is a/L, b/L.  OKLab is homogeneous: scaling linear light by k
# scales L, a and b all by k^(1/3), so these ratios are exactly unchanged by
# shadows and brightness changes.  Below this lightness (about sRGB 25/255)
# the divisor stops shrinking, so near-black noise fades toward neutral
# instead of being amplified into wild colors.
_CHROMINANCE_L_FLOOR = 0.15

# Below this OKLab chroma a hue is increasingly meaningless (8-bit noise near
# gray is ~0.001-0.003; muted pastels and skin start around 0.03-0.05).
_HUE_CHROMA_RAMP = 0.03

_TWO_PI = 6.283185307179586


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, torch.pow((c + 0.055) / 1.055, 2.4))


def oklab(rgb):
    """sRGB-encoded rgb [..., 3] in 0..1 -> OKLab [..., 3].

    Written as explicit elementwise arithmetic (no matmul) so that a given
    color always converts to bit-identical values regardless of tensor shape,
    which keeps tolerance 0 an exact-match test.
    """
    rgb = torch.clamp(rgb, min=0.0, max=1.0)
    r = _srgb_to_linear(rgb[..., 0])
    g = _srgb_to_linear(rgb[..., 1])
    b = _srgb_to_linear(rgb[..., 2])
    l = torch.pow(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b, 1.0 / 3.0)
    m = torch.pow(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b, 1.0 / 3.0)
    s = torch.pow(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b, 1.0 / 3.0)
    return torch.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], dim=-1)


def _frame_change(src, t0, t1):
    """OKLab difference between each frame and the previous one.

    Frame 0 has no previous frame, so it is compared with frame 1 instead
    (otherwise points placed on frame 0 could never select motion).
    """
    T = src.shape[0]
    frames = src[t0:t1]
    if T < 2:
        return torch.zeros_like(frames[..., 0])
    prev_idx = [t - 1 if t > 0 else 1 for t in range(t0, t1)]
    prev = torch.stack([src[i] for i in prev_idx], dim=0)
    diff = oklab(frames) - oklab(prev)
    return torch.sqrt((diff * diff).sum(-1)) / _OKLAB_MAX_DIST


def compute_quality(src, t0, t1, quality):
    """Quality values for frames t0..t1-1 of src [T, H, W, 3].

    Returns [n, H, W] for scalar qualities or [n, H, W, k] for vector ones.
    """
    if quality == "red":
        return src[t0:t1, ..., 0]
    if quality == "green":
        return src[t0:t1, ..., 1]
    if quality == "blue":
        return src[t0:t1, ..., 2]
    if quality == "frame change":
        return _frame_change(src, t0, t1)
    lab = oklab(src[t0:t1])
    if quality == "color":
        return lab
    if quality == "lightness":
        return lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    if quality == "chrominance":
        L = torch.clamp(lab[..., 0], min=_CHROMINANCE_L_FLOOR)
        return torch.stack([a / L, b / L], dim=-1)
    chroma = torch.sqrt(a * a + b * b)
    if quality == "chroma":
        return chroma
    if quality == "hue":
        h = torch.remainder(torch.atan2(b, a) / _TWO_PI, 1.0)
        w = torch.clamp(chroma / _HUE_CHROMA_RAMP, min=0.0, max=1.0)
        return torch.stack([h, w], dim=-1)
    raise ValueError(f"Unknown color quality: {quality!r}")


def distance(q, seed, quality):
    """Normalized distance (0..1) between quality values q and a seed value."""
    if quality == "color":
        diff = q - seed
        d = torch.sqrt((diff * diff).sum(-1)) / _OKLAB_MAX_DIST
    elif quality == "chrominance":
        diff = q - seed
        d = torch.sqrt((diff * diff).sum(-1)) / _MAX_REL_CHROMINANCE_DIST
    elif quality == "chroma":
        d = torch.abs(q - seed) / _OKLAB_MAX_CHROMA
    elif quality == "hue":
        # Hue difference (opposite hues = 1) counts in proportion to how
        # colorful both pixels are; a colorless pixel is maximally far from a
        # colorful one and identical to another colorless one.
        dh = torch.abs(q[..., 0] - seed[..., 0])
        dh = 2.0 * torch.minimum(dh, 1.0 - dh)
        w = q[..., 1]
        ws = seed[..., 1]
        d = w * ws * dh + torch.abs(w - ws)
    else:  # red, green, blue, lightness, frame change
        d = torch.abs(q - seed)
    return torch.clamp(d, min=0.0, max=1.0)


def strength_from_distance(d, full_tol, partial_tol):
    """Map distances to match strengths in [0, 1] (see module docstring)."""
    ft = full_tol / 100.0
    pt = partial_tol / 100.0
    full = d <= ft + _EPS
    if pt > 0:
        partial = torch.clamp(1.0 - (d - ft) / pt, min=0.0, max=1.0)
        return torch.where(full, torch.ones_like(d), partial)
    return full.to(d.dtype)


# ---------------------------------------------------------------------------
# Region growing
# ---------------------------------------------------------------------------

def _sweep_forward(reached, s, free, dim):
    """One forward run-sweep along `dim` (see module docstring)."""
    n = s.shape[dim]
    if n < 2:
        return reached
    prev = [slice(None)] * s.dim()
    cur = [slice(None)] * s.dim()
    prev[dim] = slice(0, n - 1)
    cur[dim] = slice(1, n)
    prev = tuple(prev)
    cur = tuple(cur)

    # brk[i] is True when the move (i-1) -> i is NOT allowed; index 0 always breaks.
    brk = torch.ones_like(reached)
    s_cur = s[cur]
    brk[cur] = ~((s_cur > 0) & (free[prev] | (s_cur <= s[prev])))

    seg = torch.cumsum(brk.to(torch.int32), dim=dim, dtype=torch.int32)
    marker = torch.where(reached, seg, torch.full_like(seg, -1))
    last_reached_seg = torch.cummax(marker, dim=dim)[0]
    return reached | (last_reached_seg == seg)


def _sweep(reached, s, free, dim, reverse):
    if reverse:
        r = _sweep_forward(torch.flip(reached, dims=[dim]),
                           torch.flip(s, dims=[dim]),
                           torch.flip(free, dims=[dim]), dim)
        return torch.flip(r, dims=[dim])
    return _sweep_forward(reached, s, free, dim)


def _sweep_chunked(reached, s, free, dim, reverse):
    """Run a sweep along `dim` in chunks taken along some other axis."""
    T, H, W = s.shape
    if dim == 0:
        chunk_axis, total, per = 1, H, max(1, _CHUNK_ELEMS // max(1, T * W))
    else:
        chunk_axis, total, per = 0, T, max(1, _CHUNK_ELEMS // max(1, H * W))
    if per >= total:
        return _sweep(reached, s, free, dim, reverse)
    out = torch.empty_like(reached)
    for start in range(0, total, per):
        idx = [slice(None)] * 3
        idx[chunk_axis] = slice(start, min(total, start + per))
        idx = tuple(idx)
        out[idx] = _sweep(reached[idx], s[idx], free[idx], dim, reverse)
    return out


def grow_region(s, seed_mask, growth_threshold, max_rounds=100000):
    """Grow a selection over strength map s [T, H, W] from seed_mask [T, H, W].

    Returns a bool tensor of reached pixels.
    """
    free = (s > growth_threshold) | (s >= 1.0)
    reached = seed_mask & (s > 0)
    count = int(reached.sum())
    if count == 0:
        return reached
    # Spatial axes first (cheapest and usually most productive), then time.
    order = [(2, False), (2, True), (1, False), (1, True), (0, False), (0, True)]
    for _ in range(max_rounds):
        for dim, rev in order:
            if s.shape[dim] > 1:
                reached = _sweep_chunked(reached, s, free, dim, rev)
        new_count = int(reached.sum())
        if new_count == count:
            break
        count = new_count
    return reached


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def block_reduce(mask, block_size, reduction):
    """Reduce each block_size x block_size block to one value and expand it
    back over the block, so the result keeps the input's dimensions."""
    if block_size == 1:
        return mask
    T, H, W = mask.shape
    b = block_size
    v = mask.reshape(T, H // b, b, W // b, b)
    if reduction == "max":
        r = v.amax(dim=(2, 4))
    elif reduction == "min":
        r = v.amin(dim=(2, 4))
    elif reduction == "average":
        r = v.mean(dim=(2, 4))
    else:
        raise ValueError(f"Unknown block reduction: {reduction!r}")
    return torch.repeat_interleave(torch.repeat_interleave(r, b, dim=1), b, dim=2)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def _as_rgb(t):
    """[T, H, W, C] -> float32 [T, H, W, 3] (extra channels dropped, gray expanded)."""
    t = t.to(torch.float32)
    c = t.shape[-1]
    if c >= 3:
        return t[..., :3]
    return torch.stack([t[..., 0]] * 3, dim=-1)


def magic_wand(images, points, quality="color", full_tolerance=10.0, partial_tolerance=0.0,
               growth_threshold=0.5, block_size=1, block_reduction="max",
               apply_mask_threshold=False, mask_threshold=0.5, guide=None):
    """Compute magic-wand masks.

    images: [T, H, W, C] float tensor in 0..1, already on the device the work
            should run on.  Defines the output size.
    points: iterable of (x, y) integer pixel coordinates on frame 0.
    guide:  optional [T or 1, H, W, C] tensor on the same device; when given,
            all similarity measurements are made on it instead of `images`.
    Returns a float32 mask tensor [T, H, W] on the same device.
    """
    if quality not in QUALITIES:
        raise ValueError(f"Unknown color quality: {quality!r}")
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {BLOCK_SIZES}, got {block_size}")
    T, H, W = images.shape[0], images.shape[1], images.shape[2]
    if H % block_size or W % block_size:
        raise ValueError(
            f"Image size {W}x{H} is not a multiple of block_size {block_size}. "
            f"Resize or crop the input, or choose a smaller block size.")

    if guide is None:
        src = _as_rgb(images)
    else:
        gT, gH, gW = guide.shape[0], guide.shape[1], guide.shape[2]
        if (gH, gW) != (H, W) or gT not in (1, T):
            raise ValueError(
                f"guide_image is {gT} frame(s) of {gW}x{gH}, but image is {T} frame(s) of {W}x{H}. "
                f"The guide must match the image's size and have either the same number of frames or just one.")
        src = _as_rgb(guide)
        if gT == 1 and T > 1:
            src = src.expand(T, H, W, 3)

    out = torch.zeros((T, H, W), dtype=torch.float32, device=images.device)
    if not points:
        out = block_reduce(out, block_size, block_reduction)
        return (out >= mask_threshold).to(torch.float32) if apply_mask_threshold else out

    first = compute_quality(src, 0, 1, quality)[0]

    # Seeds whose quality value is identical produce identical strength maps,
    # so they can share one grow pass with several starting pixels.
    groups = {}
    for x, y in points:
        x = min(max(int(x), 0), W - 1)
        y = min(max(int(y), 0), H - 1)
        seed_val = first[y, x]
        key = tuple(seed_val.reshape(-1).tolist())
        groups.setdefault(key, (seed_val, []))[1].append((x, y))
    del first

    frames_per_chunk = max(1, _CHUNK_ELEMS // max(1, H * W))
    for seed_val, coords in groups.values():
        s = torch.empty((T, H, W), dtype=torch.float32, device=images.device)
        for t0 in range(0, T, frames_per_chunk):
            t1 = min(T, t0 + frames_per_chunk)
            q = compute_quality(src, t0, t1, quality)
            s[t0:t1] = strength_from_distance(distance(q, seed_val, quality),
                                              full_tolerance, partial_tolerance)
        seed_mask = torch.zeros((T, H, W), dtype=torch.bool, device=images.device)
        for x, y in coords:
            seed_mask[0, y, x] = True
        reached = grow_region(s, seed_mask, growth_threshold)
        out = torch.maximum(out, torch.where(reached, s, torch.zeros_like(s)))
        del s, seed_mask, reached

    out = block_reduce(out, block_size, block_reduction)
    if apply_mask_threshold:
        out = (out >= mask_threshold).to(torch.float32)
    return out
