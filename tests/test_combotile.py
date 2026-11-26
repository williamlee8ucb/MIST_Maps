"""Unit and integration tests for ComboTile, JTile, LunarTile, and TerrainData."""

import csv
import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ComboTile import ComboTile, JTile, LunarTile, _resolve_hemisphere


def _write_synthetic_tif(path, data, lon_left, lat_top, res_deg, nodata=-9999.0):
    transform = from_origin(lon_left, lat_top, res_deg, res_deg)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


class TestComboTileSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.res = 1.0 / 360.0
        nrows, ncols = 120, 120
        lon_left, lat_top = 10.0, 60.5
        x_idx = np.arange(ncols, dtype=np.float32)
        elev = np.tile(x_idx * 10.0, (nrows, 1))
        self.tif_name = "ALPSMLC30_N060E010_DSM.tif"
        self.tif_path = os.path.join(self.tmpdir.name, self.tif_name)
        _write_synthetic_tif(self.tif_path, elev, lon_left, lat_top, self.res)

    def tearDown(self):
        plt.close("all")
        self.tmpdir.cleanup()

    def test_jtile_per_row_gradient_latitude_scaling(self):
        tile = JTile("060 N / 010 E", root_dir=self.tmpdir.name)
        grad_x, grad_y = tile.compute_gradient()
        mid_row = 60
        _, mid_lat = tile.xy_to_lonlat(0, mid_row)
        expected_dx_m = (
            ComboTile.METERS_PER_DEG_EARTH
            * np.cos(np.radians(mid_lat))
            * self.res
        )
        expected_grad_x = 10.0 / expected_dx_m
        self.assertTrue(np.allclose(grad_x[mid_row, 10:-10], expected_grad_x, rtol=1e-4))
        self.assertTrue(np.allclose(grad_y, 0.0, atol=1e-8))

    def test_map_table_and_bounds(self):
        tile = JTile(self.tif_name, root_dir=self.tmpdir.name)
        table_no_grad = tile.map_table(want_grad=False)
        self.assertEqual(table_no_grad.shape, (120 * 120, 3))

        table_grad = tile.map_table(want_grad=True)
        self.assertEqual(table_grad.shape, (120 * 120, 7))

        min_lon, max_lon, min_lat, max_lat = tile.tile_bounds()
        self.assertLess(min_lon, max_lon)
        self.assertLess(min_lat, max_lat)

        px = tile.get_elevation(10, 20)
        self.assertEqual(px.shape, (3,))
        self.assertAlmostEqual(px[2], 200.0, places=3)

    def test_max_elev_requires_explicit_observer(self):
        tile = JTile(self.tif_name, root_dir=self.tmpdir.name)
        with self.assertRaises(ValueError):
            tile.max_elev_enu()

    def test_plot_methods_and_observer_marker(self):
        tile = JTile(self.tif_name, root_dir=self.tmpdir.name)
        obs = (10.1, 60.3)
        fig1, _ = tile.plot_map(cmap="viridis", observer=obs)
        fig2, _ = tile.plot_slope(want_log=False, observer=obs)
        fig3, _ = tile.plot_slope(want_log=True)

        az = np.arange(0.0, 361.0, 10.0)
        el = np.full_like(az, 5.0)
        r = np.full_like(az, 5000.0)
        z = np.full_like(az, 400.0)
        fig4, _ = tile.plot_horizon(az, el, r, z, polar=False)
        fig5, _ = tile.plot_horizon(az, el, polar=True)
        self.assertTrue(all(f is not None for f in [fig1, fig2, fig3, fig4, fig5]))

    def test_save_horizon_csv_and_sky_view(self):
        az = np.arange(0.0, 361.0, 10.0)
        el = np.zeros_like(az)
        svf_flat = ComboTile.sky_view_fraction(az, el)
        self.assertAlmostEqual(svf_flat["sky_view_solid"], 1.0, places=6)
        self.assertAlmostEqual(svf_flat["sky_view_cos2"], 1.0, places=6)

        el_30 = np.full_like(az, 30.0)
        svf_30 = ComboTile.sky_view_fraction(az, el_30)
        self.assertAlmostEqual(svf_30["sky_view_solid"], 0.5, places=6)
        self.assertAlmostEqual(svf_30["sky_view_cos2"], 0.75, places=6)

        csv_path = os.path.join(self.tmpdir.name, "horizon.csv")
        coords = np.column_stack([np.full_like(az, 10.1), np.full_like(az, 60.3)])
        ComboTile.save_horizon_csv(csv_path, az, el_30, np.full_like(az, 1000.0), np.full_like(az, 500.0), coords)
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), len(az) + 1)
        self.assertEqual(rows[0][0], "azimuth_deg")

    def test_hemisphere_crossing_resolution(self):
        self.assertEqual(_resolve_hemisphere(-1, "N"), ("S", 1))
        self.assertEqual(_resolve_hemisphere(0, "S"), ("N", 0))
        self.assertEqual(_resolve_hemisphere(-2, "W"), ("E", 2))
        self.assertEqual(_resolve_hemisphere(1, "W"), ("W", 1))

    def test_lunartile_accepts_arbitrary_tif_name(self):
        custom_path = os.path.join(self.tmpdir.name, "custom_moon_clip.tif")
        arr = np.zeros((30, 30), dtype=np.float32)
        _write_synthetic_tif(custom_path, arr, -2.0, 2.0, 0.05)
        tile = LunarTile("custom_moon_clip.tif", root_dir=self.tmpdir.name, verbose=False)
        self.assertEqual(tile.shape, (30, 30))

    def test_save_and_load_h5(self):
        try:
            import h5py  # noqa: F401
        except (ImportError, ValueError) as exc:
            self.skipTest(f"h5py unavailable in current Python environment: {exc}")

        from TerrainData import TerrainData

        tile = JTile(self.tif_name, root_dir=self.tmpdir.name)
        h5_path = os.path.join(self.tmpdir.name, "test_tile.h5")
        tile.save_data(h5_path, compute_gradients=True)
        loaded = TerrainData.load_terrain_data(h5_path)
        self.assertEqual(loaded.shape, tile.shape)
        self.assertTrue(np.allclose(loaded.get_elevation(), tile.elevations, atol=1e-3))


class TestRealDatasets(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_arctic_mosaic_masks_nodata_and_matches_known_site(self):
        if not os.path.isdir("Arctic_Data"):
            self.skipTest("Arctic_Data directory not present")
        arctic = ComboTile("79 N / 91 W", root_dir="Arctic_Data", verbose=False)
        az_c, max_el, max_r, max_z, max_coords = arctic.max_elev_enu(
            bin_width=5.0,
            radius_max_meters=20000.0,
            lon0=-90.74778,
            lat0=79.41861,
        )
        self.assertEqual(len(az_c), 73)
        valid_z = max_z[np.isfinite(max_z)]
        self.assertGreater(valid_z.min(), -5000.0)
        peak_idx = int(np.nanargmax(max_el))
        self.assertAlmostEqual(max_el[peak_idx], 14.16, delta=0.05)
        self.assertAlmostEqual(az_c[peak_idx], 115.0, places=2)

    def test_lunar_copernicus_curvature(self):
        lunar_path = os.path.join("Lunar_Test_Data", "lunar_9.62_-20.08.tif")
        if not os.path.exists(lunar_path):
            self.skipTest("Lunar_Test_Data/lunar_9.62_-20.08.tif not present")

        cop = LunarTile("lunar_9.62_-20.08.tif", root_dir="Lunar_Test_Data", verbose=False)
        _, _, az_c, max_curved, _, _, _ = cop.max_elev_enu(
            bin_width=10.0,
            radius_max_meters=100000.0,
            lon0=-20.08,
            lat0=9.62,
            include_curvature=True,
        )
        _, _, _, max_flat, _, _, _ = cop.max_elev_enu(
            bin_width=10.0,
            radius_max_meters=100000.0,
            lon0=-20.08,
            lat0=9.62,
            include_curvature=False,
        )
        self.assertEqual(len(az_c), 37)
        self.assertTrue(np.all(max_curved <= max_flat + 1e-3))
        self.assertAlmostEqual(float(np.max(max_flat - max_curved)), 0.79, delta=0.05)


if __name__ == "__main__":
    unittest.main()
