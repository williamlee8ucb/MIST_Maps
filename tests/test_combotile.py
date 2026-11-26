"""Unit and integration tests for ComboTile, JTile, LunarTile, and TerrainData."""

import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ComboTile import ComboTile, JTile, LunarTile


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
        self.res = 1.0 / 360.0  # ~308 m at equator
        nrows, ncols = 120, 120
        # Build a synthetic linear ramp at 60N where cos(60) = 0.5
        lon_left, lat_top = 10.0, 60.5
        # Create a small 1x1 grid tile named for JTile
        x_idx = np.arange(ncols, dtype=np.float32)
        elev = np.tile(x_idx * 10.0, (nrows, 1))
        self.tif_name = "ALPSMLC30_N060E010_DSM.tif"
        self.tif_path = os.path.join(self.tmpdir.name, self.tif_name)
        _write_synthetic_tif(self.tif_path, elev, lon_left, lat_top, self.res)

    def tearDown(self):
        plt.close("all")
        self.tmpdir.cleanup()

    def test_jtile_gradient_latitude_scaling(self):
        tile = JTile("060 N / 010 E", root_dir=self.tmpdir.name)
        grad_x, grad_y = tile.compute_gradient()
        expected_dx_m = (
            ComboTile.METERS_PER_DEG_EARTH
            * np.cos(np.radians(tile.center[1]))
            * self.res
        )
        expected_grad_x = 10.0 / expected_dx_m
        self.assertTrue(np.allclose(grad_x[:, 10:-10], expected_grad_x, rtol=1e-4))
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

    def test_plot_methods_return_fig_ax(self):
        tile = JTile(self.tif_name, root_dir=self.tmpdir.name)
        fig1, ax1 = tile.plot_map()
        fig2, ax2 = tile.plot_slope(want_log=False)
        fig3, ax3 = tile.plot_slope(want_log=True)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(fig3)

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

    def test_arctic_mosaic_masks_nodata(self):
        if not os.path.isdir("Arctic_Data"):
            self.skipTest("Arctic_Data directory not present")
        arctic = ComboTile("79 N / 91 W", root_dir="Arctic_Data", verbose=False)
        az_c, max_el, max_r, max_z, max_coords = arctic.max_elev_enu(
            bin_width=10.0,
            radius_max_meters=20000.0,
            lon0=-90.74778,
            lat0=79.41861,
        )
        self.assertEqual(len(az_c), 37)
        # Nodata (-9999) from missing tiles must not produce -10,000 m jumps
        valid_z = max_z[np.isfinite(max_z)]
        self.assertGreater(valid_z.min(), -5000.0)
        self.assertGreater(np.nanmax(max_el), 5.0)

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
        # Curvature lowers distant terrain relative to the local tangent plane
        self.assertTrue(np.all(max_curved <= max_flat + 1e-3))


if __name__ == "__main__":
    unittest.main()
