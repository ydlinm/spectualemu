"""Shared illumination geometry for the Step 2 and Step 4 simulations.

All distances in the public configuration are explicit: pixel size is in
micrometres and beam radii are in millimetres.  The generated arrays remain
indexed in pixels, but their Gaussian scale is derived from the physical
parameters instead of an image-size fraction.
"""

from dataclasses import dataclass
from math import log, sqrt
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class IlluminationConfig:
    """Physical and sampling parameters shared by the simulation steps."""

    image_size: int = 256
    pixel_size_um: float = 15.0
    beam_half_intensity_radius_mm: float = 4.0
    grid_shape: Tuple[int, int] = (7, 7)
    pitch_pixels: int = 30

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        if self.beam_half_intensity_radius_mm <= 0:
            raise ValueError("beam_half_intensity_radius_mm must be positive")
        if len(self.grid_shape) != 2 or any(n <= 0 for n in self.grid_shape):
            raise ValueError("grid_shape must contain two positive integers")
        if self.pitch_pixels <= 0:
            raise ValueError("pitch_pixels must be positive")

    @property
    def pixel_size_mm(self) -> float:
        return self.pixel_size_um / 1000.0

    @property
    def field_width_mm(self) -> float:
        return self.image_size * self.pixel_size_mm

    @property
    def field_diagonal_radius_mm(self) -> float:
        # Pixel centres are located from 0 to image_size - 1.
        half_extent = (self.image_size - 1) / 2.0 * self.pixel_size_mm
        return sqrt(2.0) * half_extent

    @property
    def pitch_mm(self) -> float:
        return self.pitch_pixels * self.pixel_size_mm

    @property
    def beam_sigma_mm(self) -> float:
        # I(r) = exp(-(r/sigma)^2), I(r_half) = 0.5.
        return self.beam_half_intensity_radius_mm / sqrt(log(2.0))


def gaussian_beam_profile(
    image_size: int,
    pixel_size_um: float,
    half_intensity_radius_mm: float,
    profile_type: str = "gaussian",
) -> np.ndarray:
    """Return a unit-normalized radial illumination map.

    ``half_intensity_radius_mm`` is defined analytically, even when the
    finite image does not reach that radius.  A super-Gaussian uses the same
    half-intensity radius by deriving its own sigma.
    """

    if profile_type not in {"gaussian", "supergaussian"}:
        raise ValueError("profile_type must be 'gaussian' or 'supergaussian'")
    if image_size <= 0 or pixel_size_um <= 0 or half_intensity_radius_mm <= 0:
        raise ValueError("image_size, pixel_size_um and radius must be positive")

    pixel_size_mm = pixel_size_um / 1000.0
    center = (image_size - 1) / 2.0
    yy, xx = np.ogrid[:image_size, :image_size]
    radius_mm = np.hypot(yy - center, xx - center) * pixel_size_mm

    order = 2 if profile_type == "gaussian" else 4
    sigma_mm = half_intensity_radius_mm / (log(2.0) ** (1.0 / order))
    beam = np.exp(-((radius_mm / sigma_mm) ** order))
    return np.clip(beam, 0.0, 1.0).astype(np.float32)
