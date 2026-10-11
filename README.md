# Spectral emulation

## Shared illumination model

Step 2 and Step 4 use `illumination.py` for the sampled illumination field. The default configuration is explicit and shared:

- Image: 256 × 256 pixels at 15 μm/pixel, so the sampled field is 3.84 mm wide.
- Beam: Gaussian with an analytical half-intensity radius of 4.0 mm. The finite image does not reach that radius; this is recorded in metadata rather than silently treating the image edge as 4 mm.
- Pore array: 7 × 7 with 30 pixels (0.45 mm) pitch.

The sampled field reaches only a 2.70 mm diagonal radius, so the default corner intensity is about 0.728; the 4 mm half-intensity point is an analytical extrapolation outside the image. The model is a parameterized simulation assumption, not an experimental calibration. Run the consistency checks with:

```bash
python -m unittest discover -s tests -v
```
