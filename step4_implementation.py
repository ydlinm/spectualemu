# -*- coding: utf-8 -*-
"""
Step 4: Data Factory (Physics-Corrected Batch Generation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift, rotate
from PIL import Image
import os
from datetime import datetime
import json
from tqdm import tqdm


class SpectralAssets:
    """Container for spectral and optics assets."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.wavelengths = df["wavelength_nm"].values.astype(np.float32)
        self.halogen_spectrum = df["halogen_spectrum"].values.astype(np.float32)
        self.sensor_qe = df["sensor_qe"].values.astype(np.float32)
        self.water_mu_a = df["water_mu_a"].values.astype(np.float32)
        self.lipid_mu_a = df["lipid_mu_a"].values.astype(np.float32)
        if "sebum_mu_a" in df.columns:
            self.sebum_mu_a = df["sebum_mu_a"].values.astype(np.float32)
        else:
            self.sebum_mu_a = 0.3 * self.water_mu_a + 0.7 * self.lipid_mu_a
        if "melanin_mu_a" in df.columns:
            self.melanin_mu_a = df["melanin_mu_a"].values.astype(np.float32)
        else:
            self.melanin_mu_a = (1.7e12 * (self.wavelengths ** -3.48)).astype(np.float32)
        self.prism_shift_px = self._compute_sf11_dispersion_px(self.wavelengths)
        self.num_wavelengths = len(self.wavelengths)

    @staticmethod
    def _compute_sf11_dispersion_px(wavelengths_nm):
        # SF11 Cauchy approximation in um: n(lambda)=1.74 + 0.013/lambda^2
        wl_um = wavelengths_nm / 1000.0
        n = 1.74 + 0.013 / (wl_um ** 2)
        n_center = n[len(n) // 2]

        # Match Step 3 strong-dispersion setting to guarantee visible streaks.
        f_mm = 150.0
        alpha_deg = 45.0
        p_mm = 0.015
        shifts_mm = f_mm * (n - n_center) * np.sin(np.radians(alpha_deg))
        return (shifts_mm / p_mm).astype(np.float32)


class MultiPoreMask:
    """Sparse pore coordinates for the projected dot array."""

    def __init__(self, image_size=256, grid_shape=(7, 7), pitch_pixels=30):
        self.image_size = image_size
        self.grid_shape = grid_shape
        self.pitch_pixels = pitch_pixels
        m, n = grid_shape
        total_x = (n - 1) * pitch_pixels
        total_y = (m - 1) * pitch_pixels
        start_x = (image_size - total_x) / 2.0
        start_y = (image_size - total_y) / 2.0
        pores = []
        for i in range(m):
            for j in range(n):
                x = int(round(start_x + j * pitch_pixels))
                y = int(round(start_y + i * pitch_pixels))
                if 0 <= x < image_size and 0 <= y < image_size:
                    pores.append((y, x))
        self.pores = np.array(pores, dtype=np.int32)

    def get_pores(self):
        return self.pores


class PerlinNoiseGenerator:
    @staticmethod
    def generate_perlin_2d(shape, scale=50, octaves=4, persistence=0.5, seed=None):
        rng = np.random.default_rng(seed)
        h, w = shape
        noise = np.zeros((h, w), dtype=np.float32)
        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave
            layer = rng.standard_normal((h, w)).astype(np.float32)
            layer = gaussian_filter(layer, sigma=scale / max(freq, 1))
            noise += amp * layer
        noise_min = noise.min()
        noise_max = noise.max()
        if noise_max > noise_min:
            noise = (noise - noise_min) / (noise_max - noise_min)
        return noise


class DataFactory:
    def __init__(
        self,
        assets_path,
        image_size=256,
        output_dir="output/datasets",
        sensor_max_adu=4095,
        base_gain=4500.0,
        grid_shape=(7, 7),
        pitch_pixels=30,
    ):
        self.assets = SpectralAssets(assets_path)
        self.image_size = image_size
        self.output_dir = output_dir
        self.sensor_max_adu = sensor_max_adu
        self.base_gain = base_gain
        os.makedirs(output_dir, exist_ok=True)

        self.multi_pore_mask = MultiPoreMask(
            image_size=image_size, grid_shape=grid_shape, pitch_pixels=pitch_pixels
        )
        self.pores = self.multi_pore_mask.get_pores()
        if len(self.pores) == 0:
            raise ValueError("MultiPoreMask has zero pores. Check grid_shape/pitch_pixels.")

        self.beam_map = self._build_beam_profile()
        self.surface_psf = self._gaussian_psf(size=11, sigma=1.2)

        texture_path = "assets/skin_texture.png"
        if os.path.exists(texture_path):
            self.texture_source = Image.open(texture_path).convert("L")
        else:
            noise = PerlinNoiseGenerator.generate_perlin_2d(
                (image_size, image_size), scale=20, octaves=6
            )
            self.texture_source = Image.fromarray((noise * 255).astype(np.uint8))

    def _build_beam_profile(self):
        h = self.image_size
        w = self.image_size
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        sigma = 0.42 * min(h, w)
        beam = np.exp(-(rr / sigma) ** 2).astype(np.float32)
        return beam

    def _gaussian_psf(self, size=11, sigma=1.2):
        half = size // 2
        ax = np.arange(-half, half + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        k /= np.sum(k)
        return k.astype(np.float32)

    def _subsurface_psf(self, wl_idx, mu_eff_center):
        wl = self.assets.wavelengths[wl_idx]
        wl_norm = (wl - self.assets.wavelengths[0]) / (
            self.assets.wavelengths[-1] - self.assets.wavelengths[0] + 1e-6
        )
        size = int(23 + 18 * wl_norm)
        if size % 2 == 0:
            size += 1
        size = int(np.clip(size, 21, 41))

        half = size // 2
        ax = np.arange(-half, half + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        rr = np.sqrt(xx**2 + yy**2)

        sigma = 3.0 + 10.0 * wl_norm
        radial = np.exp(-(rr**2) / (2.0 * sigma**2))
        decay = np.exp(-0.02 * np.clip(mu_eff_center, 0.0, 300.0) * rr)
        psf = radial * decay
        psf_sum = np.sum(psf)
        if psf_sum > 0:
            psf /= psf_sum
        return psf.astype(np.float32)

    def augment_texture(self, seed=None):
        rng = np.random.default_rng(seed)
        texture = np.array(self.texture_source, dtype=np.float32) / 255.0

        # Reflect pad first so random rotation never introduces blank corners.
        pad = self.image_size
        texture = np.pad(texture, ((pad, pad), (pad, pad)), mode="reflect")

        crop_big = int(round(self.image_size * 1.45))
        h, w = texture.shape
        top = rng.integers(0, h - crop_big + 1)
        left = rng.integers(0, w - crop_big + 1)
        patch = texture[top : top + crop_big, left : left + crop_big]

        angle = float(rng.uniform(-180.0, 180.0))
        patch_rot = rotate(
            patch,
            angle=angle,
            reshape=False,
            order=1,
            mode="reflect",
            prefilter=False,
        )

        c0 = (crop_big - self.image_size) // 2
        c1 = c0 + self.image_size
        out = patch_rot[c0:c1, c0:c1]
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        return out

    def generate_concentration_maps(self, conc_water, conc_sebum, seed=None):
        h, w = self.image_size, self.image_size
        perlin_water = PerlinNoiseGenerator.generate_perlin_2d(
            (h, w), scale=80, octaves=3, seed=seed
        )
        map_water = conc_water * (0.8 + 0.4 * perlin_water)

        texture = self.augment_texture(seed=None if seed is None else seed + 1)
        map_sebum = conc_sebum * (0.7 + 0.6 * texture)
        return map_water.astype(np.float32), map_sebum.astype(np.float32)

    def compute_total_mu_a(self, map_water, map_sebum, conc_melanin):
        mu_a_map = (
            map_water[:, :, None] * self.assets.water_mu_a[None, None, :]
            + map_sebum[:, :, None] * self.assets.sebum_mu_a[None, None, :]
            + conc_melanin * self.assets.melanin_mu_a[None, None, :]
        )
        return mu_a_map.astype(np.float32)

    def compute_wavelength_dependent_pathlength(self):
        wl = self.assets.wavelengths
        return (0.4 + (0.8 - 0.4) * (wl - wl[0]) / (wl[-1] - wl[0] + 1e-6)).astype(np.float32)

    def compute_mu_s_prime(self, a, b):
        wl = self.assets.wavelengths
        return (a * (wl / 500.0) ** (-b)).astype(np.float32)

    def _accumulate_psf(self, image, py, px, kernel, amplitude, texture_patch=None):
        h, w = image.shape
        kh, kw = kernel.shape
        hy = kh // 2
        hx = kw // 2

        y0 = max(0, py - hy)
        y1 = min(h, py + hy + 1)
        x0 = max(0, px - hx)
        x1 = min(w, px + hx + 1)

        ky0 = hy - (py - y0)
        ky1 = ky0 + (y1 - y0)
        kx0 = hx - (px - x0)
        kx1 = kx0 + (x1 - x0)

        patch = kernel[ky0:ky1, kx0:kx1]
        if texture_patch is not None:
            image[y0:y1, x0:x1] += amplitude * patch * texture_patch[y0:y1, x0:x1]
        else:
            image[y0:y1, x0:x1] += amplitude * patch

    def build_scene_hypercube(self, mu_a_map, mu_s_prime, map_water, map_sebum, exposure_factor):
        h, w, num_wl = mu_a_map.shape
        scene = np.zeros((h, w, num_wl), dtype=np.float32)

        d_eff = self.compute_wavelength_dependent_pathlength()
        mu_s_3d = mu_s_prime[None, None, :]
        mu_eff = np.sqrt(3.0 * mu_a_map * (mu_a_map + mu_s_3d))

        local_surface_texture = np.clip(0.2 + 0.7 * map_sebum + 0.1 * map_water, 0.0, 1.5)

        for py, px in self.pores:
            beam_weight = float(self.beam_map[py, px])
            if beam_weight < 1e-5:
                continue

            for wl_idx in range(num_wl):
                source_term = (
                    beam_weight
                    * self.assets.halogen_spectrum[wl_idx]
                    * self.assets.sensor_qe[wl_idx]
                    * exposure_factor
                    * self.base_gain
                )

                mu_eff_c = float(mu_eff[py, px, wl_idx])
                surface_amp = source_term * np.exp(-0.30 * mu_a_map[py, px, wl_idx] * d_eff[wl_idx])
                subsurface_amp = source_term * np.exp(-0.85 * mu_eff_c * d_eff[wl_idx])

                self._accumulate_psf(
                    scene[:, :, wl_idx],
                    py,
                    px,
                    self.surface_psf,
                    surface_amp,
                    texture_patch=local_surface_texture,
                )

                sub_psf = self._subsurface_psf(wl_idx, mu_eff_center=mu_eff_c)
                self._accumulate_psf(
                    scene[:, :, wl_idx], py, px, sub_psf, subsurface_amp, texture_patch=None
                )

        return scene

    def apply_dispersion_and_sensor(self, scene_hypercube, snr_mode):
        h, w, num_wl = scene_hypercube.shape
        sensor_ideal = np.zeros((h, w), dtype=np.float32)

        for wl_idx in range(num_wl):
            dx = float(self.assets.prism_shift_px[wl_idx])
            shifted = shift(scene_hypercube[:, :, wl_idx], shift=(0.0, dx), order=1, mode="constant", cval=0.0)
            sensor_ideal += shifted

        peak = float(sensor_ideal.max())
        if peak > 0:
            target_peak = 3600.0
            sensor_ideal *= target_peak / peak

        if snr_mode == "high":
            read_noise_std = 2.0
            dark_current = 1.5
        elif snr_mode == "medium":
            read_noise_std = 4.0
            dark_current = 4.0
        else:
            read_noise_std = 7.0
            dark_current = 8.0

        signal = np.clip(sensor_ideal, 0, None)
        shot = np.random.poisson(signal + 1e-6).astype(np.float32)
        read = np.random.normal(0, read_noise_std, (h, w)).astype(np.float32)
        dark = np.random.poisson(dark_current, (h, w)).astype(np.float32)

        adu = shot + read + dark
        adu = np.clip(adu, 0, self.sensor_max_adu).astype(np.uint16)
        return adu, sensor_ideal

    def generate_batch(self, num_samples=5, seed_offset=0):
        metadata_list = []

        for i in tqdm(range(num_samples), desc="Generating Step4"):
            sample_id = i + seed_offset
            seed = sample_id + 42
            np.random.seed(seed)

            for _ in range(20):
                conc_water = np.random.uniform(0.20, 0.60)
                conc_sebum = np.random.uniform(0.05, 0.25)
                conc_melanin = np.random.uniform(0.00, 0.10)
                if conc_water + conc_sebum + conc_melanin < 0.95:
                    break

            map_water, map_sebum = self.generate_concentration_maps(conc_water, conc_sebum, seed=seed)
            mu_a_map = self.compute_total_mu_a(map_water, map_sebum, conc_melanin)

            a = np.random.uniform(1.0, 2.0)
            b = np.random.uniform(0.5, 1.5)
            mu_s_prime = self.compute_mu_s_prime(a, b)

            exposure_factor = float(np.random.uniform(0.9, 1.2))

            snr_rand = np.random.rand()
            if snr_rand < 0.6:
                snr_mode = "high"
            elif snr_rand < 0.9:
                snr_mode = "medium"
            else:
                snr_mode = "low"

            scene_hypercube = self.build_scene_hypercube(
                mu_a_map, mu_s_prime, map_water, map_sebum, exposure_factor
            )
            sensor_image_adu, sensor_ideal = self.apply_dispersion_and_sensor(scene_hypercube, snr_mode)

            save_path = os.path.join(self.output_dir, f"sample_{sample_id:05d}.npz")
            np.savez_compressed(
                save_path,
                x=sensor_image_adu,
                y=np.stack([map_water, map_sebum], axis=0).astype(np.float32),
                meta=np.array(
                    [conc_water, conc_sebum, conc_melanin, a, b, exposure_factor], dtype=np.float32
                ),
                snr_mode=snr_mode,
            )

            nonzero = np.argwhere(sensor_ideal > (0.05 * sensor_ideal.max() if sensor_ideal.max() > 0 else 0))
            x_span = 0
            if len(nonzero) > 0:
                x_span = int(nonzero[:, 1].max() - nonzero[:, 1].min())

            metadata_list.append(
                {
                    "sample_id": int(sample_id),
                    "conc_water_mean": float(map_water.mean()),
                    "conc_sebum_mean": float(map_sebum.mean()),
                    "conc_melanin": float(conc_melanin),
                    "scatter_a": float(a),
                    "scatter_b": float(b),
                    "exposure_factor": float(exposure_factor),
                    "snr_mode": snr_mode,
                    "sensor_adu_mean": float(sensor_image_adu.mean()),
                    "sensor_adu_max": int(sensor_image_adu.max()),
                    "num_pores": int(len(self.pores)),
                    "dispersion_span_px_at_5pct": x_span,
                }
            )

        meta_path = os.path.join(self.output_dir, "dataset_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_samples": int(num_samples),
                    "image_size": int(self.image_size),
                    "num_wavelengths": int(self.assets.num_wavelengths),
                    "wavelength_range": [
                        float(self.assets.wavelengths[0]),
                        float(self.assets.wavelengths[-1]),
                    ],
                    "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sensor_max_adu": int(self.sensor_max_adu),
                    "base_gain": float(self.base_gain),
                    "pore_grid": list(self.multi_pore_mask.grid_shape),
                    "pore_pitch_pixels": int(self.multi_pore_mask.pitch_pixels),
                    "samples": metadata_list,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def visualize_sample(self, sample_id=0, save_name="step4_validation_sample0.png"):
        sample_path = os.path.join(self.output_dir, f"sample_{sample_id:05d}.npz")
        data = np.load(sample_path, allow_pickle=True)

        sensor_image = data["x"]
        map_water = data["y"][0]
        map_sebum = data["y"][1]

        log_image = np.log1p(sensor_image.astype(np.float32))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(log_image, cmap="gray")
        axes[0].set_title("Input x (Log Scale)")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(map_water, cmap="Blues", vmin=0, vmax=0.7)
        axes[1].set_title("GT Free Water Map")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(map_sebum, cmap="Oranges", vmin=0, vmax=0.4)
        axes[2].set_title("GT Sebum Map")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        save_path = os.path.join("output", save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    factory = DataFactory(
        assets_path="output/step1_standardized_data.csv",
        image_size=256,
        output_dir="output/datasets",
        sensor_max_adu=4095,
        base_gain=4500.0,
        grid_shape=(7, 7),
        pitch_pixels=30,
    )
    factory.generate_batch(num_samples=5, seed_offset=0)
    factory.visualize_sample(sample_id=0, save_name="step4_validation_sample0.png")
