import unittest

import numpy as np

from illumination import IlluminationConfig, gaussian_beam_profile
from step2_implementation import BeamProfile, MultiPoreMask as Step2PoreMask
from step4_implementation import DataFactory, MultiPoreMask as Step4PoreMask


class IlluminationConsistencyTests(unittest.TestCase):
    def test_default_sampling_and_pore_pitch_are_explicit_physical_values(self):
        config = IlluminationConfig()

        self.assertAlmostEqual(config.field_width_mm, 3.84)
        self.assertAlmostEqual(config.pitch_mm, 0.45)
        self.assertEqual(config.grid_shape, (7, 7))
        self.assertTrue(
            np.array_equal(
                Step2PoreMask(
                    grid_shape=config.grid_shape,
                    pitch_pixels=config.pitch_pixels,
                    image_size=config.image_size,
                ).get_pores(),
                Step4PoreMask(
                    grid_shape=config.grid_shape,
                    pitch_pixels=config.pitch_pixels,
                    image_size=config.image_size,
                ).get_pores(),
            )
        )

    def test_beam_half_intensity_radius_is_analytically_four_mm(self):
        beam = BeamProfile()

        self.assertAlmostEqual(beam.field_width_mm, 3.84)
        self.assertAlmostEqual(beam.sigma_mm, 4.0 / np.sqrt(np.log(2.0)))
        self.assertAlmostEqual(beam.intensity_at_radius_mm(4.0), 0.5)
        self.assertTrue(np.isfinite(beam.get_map()).all())
        # The 4 mm radius is outside the diagonal of the finite 3.84 mm field.
        self.assertGreater(beam.get_map().min(), 0.5)

    def test_step2_and_step4_use_the_same_beam_map(self):
        config = IlluminationConfig()
        step2_map = BeamProfile(
            image_size=config.image_size,
            pixel_size_um=config.pixel_size_um,
            half_intensity_radius_mm=config.beam_half_intensity_radius_mm,
        ).get_map()

        factory = DataFactory.__new__(DataFactory)
        factory.image_size = config.image_size
        factory.pixel_size_um = config.pixel_size_um
        factory.beam_half_intensity_radius_mm = config.beam_half_intensity_radius_mm
        step4_map = DataFactory._build_beam_profile(factory)

        np.testing.assert_allclose(step2_map, step4_map, rtol=0, atol=1e-7)

    def test_pixel_units_change_the_sampled_physical_profile(self):
        map_15um = gaussian_beam_profile(256, 15.0, 4.0)
        map_150um = gaussian_beam_profile(256, 150.0, 4.0)

        # At 15 μm/px the corner is only 2.7 mm from the centre; at 150 μm/px
        # it is 27 mm away. A mistaken μm-to-mm conversion is obvious.
        self.assertGreater(map_15um[-1, -1], 0.5)
        self.assertLess(map_150um[-1, -1], 1e-10)

    def test_invalid_physical_parameters_fail_fast(self):
        with self.assertRaises(ValueError):
            IlluminationConfig(pixel_size_um=0)
        with self.assertRaises(ValueError):
            IlluminationConfig(beam_half_intensity_radius_mm=-1)
        with self.assertRaises(ValueError):
            gaussian_beam_profile(256, 15, 4, profile_type="box")


if __name__ == "__main__":
    unittest.main()
