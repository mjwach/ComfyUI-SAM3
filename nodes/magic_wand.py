"""
Magic Wand Mask -- paint-program style "magic wand" color selection over an
image batch (treated as consecutive video frames), driven by points placed
on the first frame in an interactive canvas like SAM3 Point Collector's.

The region-growing algorithm itself lives in magic_wand_core.py.
"""

import base64
import io as stdio
import json
import logging
import math

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
from comfy_api.latest import io

from .magic_wand_core import (BLOCK_REDUCTIONS, BLOCK_SIZES, QUALITIES, magic_wand)
from .sam3_interactive import _BG_IMAGE_PREFIX, SAM3PointCollector, _save_reference_image

log = logging.getLogger("sam3")


def _mask_preview_base64(mask):
    """First mask frame -> base64 RGBA PNG: a tint whose opacity follows the mask."""
    m = mask[0].detach().float().clamp(0, 1).cpu().numpy()
    rgba = np.zeros(m.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = np.round(m * 150).astype(np.uint8)
    buf = stdio.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class MagicWandMask(io.ComfyNode):
    """
    Interactive Magic Wand selection.

    Left-click the canvas to place green seed points (drag a point to move
    it, right-click a point to remove it).  From every seed, a selection
    grows through spatially and temporally contiguous pixels whose chosen
    color quality is similar to the seed's, and the result is output as a
    mask batch the same size as the input.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagicWandMask",
            display_name="Magic Wand Mask",
            category="SAM3",
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Image batch, treated as consecutive video frames. The first frame is shown in "
                            "the canvas; left-click it to place seed points. The output mask has its size."),
                io.Image.Input(
                    "guide_image", optional=True,
                    tooltip="Optional. When connected, similarity is measured on this image instead of on "
                            "'image', while points are still placed on (and the mask still sized to) 'image'. "
                            "Use it for a depth map, edge map, flow visualization, another model's output, etc. "
                            "It must match the image's width and height and have either the same number of "
                            "frames or a single frame (used for every frame). For a grayscale map such as "
                            "depth, match on 'red', which compares the stored values directly."),
                io.Combo.Input(
                    "match_on", options=QUALITIES, default="color",
                    tooltip="Which quality of the color is compared with the color under each point.\n"
                            "color: perceptual difference of the whole color (OKLab distance).\n"
                            "red / green / blue: one stored channel.\n"
                            "hue: perceptual hue angle (OKLCh); opposite hues are 100% apart. Colorless "
                            "pixels match each other and are maximally far from colorful ones.\n"
                            "chroma: how vivid a pixel's color is, from 0 for grays up (OKLCh C; replaces HSV "
                            "saturation, which treats near-black noise as strongly saturated).\n"
                            "lightness: perceived brightness (OKLab L).\n"
                            "chrominance: the color relative to its own brightness, which stays the same when a "
                            "surface falls into shadow or the lighting dims or brightens; good for following an "
                            "object through uneven light (very dark pixels count as neutral).\n"
                            "frame change: how much each pixel changed since the previous frame (frame 0 is "
                            "compared with frame 1); place a point on something moving to select motion, or on "
                            "something still to select the static parts."),
                io.Float.Input(
                    "full_tolerance", default=10.0, min=0.0, max=100.0, step=1.0, round=0.01,
                    tooltip="Pixels whose difference from the seed is at most this percentage of the largest "
                            "possible difference count as full matches (mask value 1). 0 = exact color only, "
                            "100 = every color."),
                io.Float.Input(
                    "partial_tolerance", default=0.0, min=0.0, max=100.0, step=1.0, round=0.01,
                    tooltip="Width (in the same percentage units) of a band just outside the full-match range "
                            "where pixels match partially, fading linearly from 1 at the edge of the full range "
                            "to 0 at the outer edge of this band."),
                io.Float.Input(
                    "growth_threshold", default=0.5, min=0.0, max=1.0, step=0.01, round=0.001,
                    tooltip="Full matches, and partial matches stronger than this, spread the selection to any "
                            "matching neighbor. Weaker partial matches only spread to neighbors that match no "
                            "more strongly than they do. Pixels with zero strength never spread."),
                io.Combo.Input(
                    "block_size", options=BLOCK_SIZES, default=1,
                    tooltip="Size of the square blocks the output mask is quantized into. The input's width and "
                            "height must be multiples of it."),
                io.Combo.Input(
                    "block_reduction", options=BLOCK_REDUCTIONS, default="max",
                    tooltip="How each block's value is derived from its pixels' match strengths."),
                io.Boolean.Input(
                    "apply_mask_threshold", default=False,
                    tooltip="Binarize the final mask: values at or above mask_threshold become 1, the rest 0."),
                io.Float.Input(
                    "mask_threshold", default=0.5, min=0.0, max=1.0, step=0.01, round=0.001,
                    tooltip="Threshold used when apply_mask_threshold is on."),
                io.Boolean.Input(
                    "save_reference_image", default=True,
                    tooltip="Save the reference image shown in the canvas to the ComfyUI input folder, "
                            "so it's still there when you reopen this workflow. Turn off if you'd rather "
                            "this node not write files to disk. Note that all input files created by this "
                            "node will have the prefix " + _BG_IMAGE_PREFIX + ", so you should be able to "
                            "quickly manually delete them at any time."),
                io.String.Input("coordinates", multiline=False, default="[]"),
                io.String.Input("bg_image_filename", multiline=False, default=""),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image, match_on, full_tolerance, partial_tolerance, growth_threshold,
                block_size, block_reduction, apply_mask_threshold, mask_threshold,
                save_reference_image=True, coordinates="[]", bg_image_filename="", guide_image=None):
        img_base64 = SAM3PointCollector._tensor_to_base64(image)
        bg_filename = _save_reference_image(img_base64) if save_reference_image else ""

        try:
            raw_points = json.loads(coordinates) if coordinates and coordinates.strip() else []
        except json.JSONDecodeError:
            raw_points = []
        points = []
        for p in raw_points:
            try:
                points.append((math.floor(float(p["x"])), math.floor(float(p["y"]))))
            except (KeyError, TypeError, ValueError):
                continue

        params = dict(
            quality=match_on,
            full_tolerance=float(full_tolerance),
            partial_tolerance=float(partial_tolerance),
            growth_threshold=float(growth_threshold),
            block_size=int(block_size),
            block_reduction=block_reduction,
            apply_mask_threshold=bool(apply_mask_threshold),
            mask_threshold=float(mask_threshold),
        )

        device = mm.get_torch_device()
        try:
            guide = guide_image.to(device) if guide_image is not None else None
            mask = magic_wand(image.to(device), points, guide=guide, **params)
        except Exception as e:
            if not mm.is_oom(e) or torch.device(device).type == "cpu":
                raise
            log.warning("Magic Wand Mask: out of GPU memory, retrying on CPU")
            mm.soft_empty_cache()
            guide = guide_image.cpu() if guide_image is not None else None
            mask = magic_wand(image.cpu(), points, guide=guide, **params)

        mask = mask.cpu()
        log.info(f"Magic Wand Mask: {len(points)} point(s), {match_on}"
                 f"{' (on guide image)' if guide_image is not None else ''}, FT={full_tolerance}, "
                 f"PT={partial_tolerance}, output {tuple(mask.shape)}")

        ui_payload = {
            "bg_image": [img_base64],
            "bg_image_filename": [bg_filename],
            "mask_preview": [_mask_preview_base64(mask)],
        }
        return io.NodeOutput(mask, ui=ui_payload)


NODE_CLASS_MAPPINGS = {
    "MagicWandMask": MagicWandMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicWandMask": "Magic Wand Mask",
}
