# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Temporal pre-embedding fusion for multi-frame (9-channel) RF-DETR input.

Motivation:
    The naive multi-frame approach stacks ``num_frames`` RGB frames into a
    ``3 * num_frames`` channel tensor and widens the patch-embed ``Conv2d`` to
    match.  Because that conv sums over all input channels and its weights are
    initialised as identical per-frame copies, the patch embedding computes the
    *temporal average* of the frames - a low-pass / motion-blur operation that
    destroys exactly the change signal the model is meant to attend to, and
    feeds an out-of-distribution (blurred) image to the pretrained backbone.

Approach:
    Keep the pretrained 3-channel patch embedding intact and place a small
    learnable module in front of it that maps the ``3 * num_frames`` channels
    down to 3.  Its learnable branch consumes temporal *differences*
    ``[f_cur, f_cur - f_0, ..., f_cur - f_{n-2}]`` so it responds directly to
    motion.  The output convolution is zero-initialised (ControlNet-style "zero
    conv"), so at initialisation the module output equals the current frame
    exactly: the backbone receives the same sharp, in-distribution image as the
    single-frame baseline (guaranteed parity), and the temporal contribution is
    learned as a residual that grows from zero.

Background-subtraction add-weight (``bg_subtract=True``):
    The zero-init residual must *discover* motion from scratch, which is too weak
    for the small, distant, semi-transparent moving smoke/fire seen in CCTV. This
    mode instead injects the change signal explicitly: it estimates a background
    as the mean of the past frames, forms the foreground ``fg = f_cur - bg``, and
    add-weights it back into the current frame with a learnable per-channel gain
    ``alpha`` (initialised positive, so the enhancement is active from step one).
    The output is ``f_cur + alpha * fg + residual`` - a temporal high-pass /
    unsharp mask that amplifies contrast exactly where motion exists, raising the
    patch-embed token magnitude (hence attention) on moving regions, while static
    background (``fg approx 0``) stays in-distribution. The learnable branch is
    additionally fed ``fg`` and its magnitude so the refinement is motion-aware.
"""

import torch
from torch import nn

__all__ = ["TemporalPreEmbed"]


class TemporalPreEmbed(nn.Module):
    """Fuse ``num_frames`` stacked RGB frames into a single 3-channel tensor.

    The input is expected to be the already-normalised channel-concatenation of
    ``num_frames`` frames in temporal order, i.e. shape ``(B, 3 * num_frames,
    H, W)`` where the *last* frame group is the current/annotated frame.

    At initialisation the module is an identity on the current frame when
    ``bg_subtract=False`` (``forward(x) == current_frame``); with
    ``bg_subtract=True`` it adds an explicit background-subtraction term
    ``alpha * (current - mean(past))`` that is zero on static input but amplifies
    moving regions from step one. In both cases the learned residual is 0 at init.

    Args:
        num_frames: Number of stacked frames (input channels = ``3 * num_frames``).
        mid_channels: Width of the internal bottleneck.
        bg_subtract: If True, add an explicit background-subtraction add-weight
            term and feed the foreground to the learnable branch.
        gain_init: Initial value of the per-channel add-weight gain ``alpha``
            (only used when ``bg_subtract=True``).
        coherence_gate: If True (with ``bg_subtract``), gate the foreground to
            keep only motion that is *consistent across every frame transition*
            (present, in the same direction, in all of ``cur - f_i``). Random
            sensor noise is uncorrelated between transitions and is suppressed,
            so the add-weight amplifies genuine moving regions without amplifying
            noise on near-static / faint inputs (the failure mode of plain
            ``bg_subtract``).
    """

    def __init__(
        self,
        num_frames: int = 3,
        mid_channels: int = 16,
        bg_subtract: bool = False,
        gain_init: float = 0.5,
        coherence_gate: bool = False,
    ) -> None:
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if bg_subtract and num_frames < 2:
            raise ValueError(
                f"bg_subtract=True needs at least 2 frames (a past frame to form the "
                f"background), got num_frames={num_frames}."
            )
        if coherence_gate and not bg_subtract:
            raise ValueError("coherence_gate=True requires bg_subtract=True.")
        self.num_frames = num_frames
        self.bg_subtract = bg_subtract
        self.coherence_gate = coherence_gate

        # When subtracting the background, the learnable branch additionally sees
        # the explicit foreground (3ch) and its magnitude (1ch) so its refinement
        # is motion-aware.
        in_channels = 3 * num_frames + (3 + 1 if bg_subtract else 0)

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.conv_dw = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels
        )
        self.act2 = nn.GELU()
        self.conv_out = nn.Conv2d(mid_channels, 3, kernel_size=1)

        # Per-channel add-weight gain for the explicit foreground term. Positive
        # init => motion is amplified from the first step (the change signal is
        # injected, not discovered from zero). On static input fg==0 so the term
        # vanishes regardless of alpha (static parity preserved).
        self.alpha: nn.Parameter | None = (
            nn.Parameter(torch.full((3,), float(gain_init))) if bg_subtract else None
        )

        # Zero-init the output conv so the residual is 0 at init => the learned
        # refinement starts from nothing. Earlier layers keep their default init
        # so gradients flow into conv_out from step one.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3 * num_frames, H, W)`` to ``(B, 3, H, W)``.

        Args:
            x: Normalised, channel-concatenated frames in temporal order.

        Returns:
            A 3-channel tensor: current frame + (optional) background-subtraction
            add-weight term + learned temporal residual.
        """
        if x.shape[1] != 3 * self.num_frames:
            raise ValueError(
                f"Expected {3 * self.num_frames} input channels for "
                f"num_frames={self.num_frames}, got {x.shape[1]}."
            )
        frames = x.chunk(self.num_frames, dim=1)
        current = frames[-1]

        if self.bg_subtract:
            # Per-transition differences cur - f_i (one per past frame).
            diffs = [current - f for f in frames[:-1]]
            # Background-subtracted foreground = mean of the transitions.
            foreground = torch.stack(diffs, dim=0).mean(dim=0)
            if self.coherence_gate:
                # Keep only motion consistent across ALL transitions: project
                # each diff onto the foreground direction and take the per-pixel
                # minimum. If every transition moves the same way (coherent
                # motion) this stays positive; if any transition disagrees
                # (uncorrelated sensor noise) the min goes negative and relu
                # zeroes it. Magnitude is the conservative (min) consistent
                # motion; direction is the foreground sign.
                sign = torch.sign(foreground)
                proj = torch.stack([d * sign for d in diffs], dim=0)  # (n_past,B,3,H,W)
                coherent = torch.relu(proj.min(dim=0).values)
                foreground = sign * coherent
            magnitude = foreground.abs().sum(dim=1, keepdim=True)
            feat = torch.cat([*frames, foreground, magnitude], dim=1)
            residual = self.conv_out(self.act2(self.conv_dw(self.act1(self.conv_in(feat)))))
            return current + self.alpha.view(1, 3, 1, 1) * foreground + residual

        # Default: current frame + per-frame differences -> zero-init residual.
        motion = torch.cat([current] + [current - f for f in frames[:-1]], dim=1)
        residual = self.conv_out(self.act2(self.conv_dw(self.act1(self.conv_in(motion)))))
        return current + residual
