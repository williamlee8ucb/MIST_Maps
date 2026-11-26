# MIST_Maps

Terrain elevation, slope mapping, and ENU-frame horizon sky-view analysis for Earth (ALOS World 3D 30 m) and Lunar (USGS LRO WAC GLD100) Digital Elevation/Surface Models.

## Features

- **`ComboTile`**: Mosaics a grid (default $5 \times 5$) of 1-degree ALOS World 3D 30 m (`ALPSMLC30_*_DSM.tif`) GeoTIFF tiles, computes per-row latitude-scaled gradients and slopes, plots elevation/slope/horizon maps, exports HDF5/CSV outputs, computes sky-view fractions, and evaluates windowed azimuthal horizon profiles in the local East-North-Up (ENU) frame.
- **`JTile`**: Lightweight subclass of `ComboTile` for single-tile Earth DSM analysis.
- **`LunarTile`**: Subclass of `ComboTile` adapted for USGS LRO WAC GLD100 lunar GeoTIFFs, using the IAU Moon reference ellipsoid, lunar radius $R_{\text{moon}} = 1{,}737{,}400\,\text{m}$, bilinear + linear observer reference interpolation, and 3-D ENU curvature-aware horizon angles.
- **`TerrainData`**: Compressed HDF5 serialization/deserialization wrapper for 2-D terrain coordinate, elevation, gradient, and slope grids.
- **`JAXA_Scrape.py`**: Selenium helper to automate downloading and extracting ALOS AW3D30 GeoTIFF tiles after JAXA registration.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `ComboTile.py` | Core module (`ComboTile`, `JTile`, `LunarTile`) |
| `TerrainData.py` | HDF5 terrain data serialization wrapper (`TerrainData`) |
| `JAXA_Scrape.py` | Automated JAXA AW3D30 tile download & extraction helper |
| `ComboTile_Demo.ipynb` | Executed Jupyter notebook with Earth and Moon example cases |
| `documentation.tex` / `documentation.pdf` | Full LaTeX documentation (API reference, Earth vs. Moon appendix, data access) |
| `Arctic_Data/` | Sample Earth ALOS tiles covering Axel Heiberg Island, Nunavut (`79°N, 91°W`) |
| `Lunar_Test_Data/` | Sample USGS LRO WAC GLD100 clips (Equatorial Mare, Tycho, Copernicus) |
| `tests/test_combotile.py` | Unit and integration tests |

## Installation

```bash
git clone git@github.com:williamlee8ucb/MIST_Maps.git
cd MIST_Maps

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use this environment inside Jupyter / VS Code notebooks, register it as a kernel:

```bash
python -m ipykernel install --user --name mist_maps --display-name "Python 3 (MIST_Maps .venv)"
```

> **Note on NumPy 2.x**: If running inside an existing Anaconda/system environment whose `scipy`, `h5py`, or `astropy` wheels were compiled against NumPy 1.x, either use the `.venv` kernel above or pin `pip install "numpy<2"`.

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Quick Start

### Earth Mosaic (`ComboTile`)

```python
from ComboTile import ComboTile

site = ComboTile("79 N / 91 W", root_dir="Arctic_Data")
obs_lon, obs_lat = -90.74778, 79.41861

site.plot_map(observer=(obs_lon, obs_lat))
site.plot_slope(want_log=False, vmax=30.0, observer=(obs_lon, obs_lat))

az, max_el, max_r, max_z, max_coords = site.max_elev_enu(
    bin_width=5.0, radius_max_meters=20_000.0, lon0=obs_lon, lat0=obs_lat
)
site.plot_horizon(az, max_el, max_r, max_z)
print(ComboTile.sky_view_fraction(az, max_el))
ComboTile.save_horizon_csv("horizon.csv", az, max_el, max_r, max_z, max_coords)
```

### Lunar DEM (`LunarTile`)

```python
from ComboTile import LunarTile

moon = LunarTile("lunar_9.62_-20.08.tif", root_dir="Lunar_Test_Data")
el_px, az_px, az, max_el, max_r, max_z, max_coords = moon.max_elev_enu(
    bin_width=10.0,
    radius_max_meters=100_000.0,
    lon0=-20.08,
    lat0=9.62,
    include_curvature=True,
)
print(LunarTile.sky_view_fraction(az, max_el))
```

## Data Access

### Earth: ALOS World 3D 30 m (AW3D30)
1. Register for a free account at <https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm>.
2. Download the 1-degree `ALPSMLC30_{N|S}{LAT}{E|W}{LON}_DSM.tif` tiles covering your area (a $5 \times 5$ mosaic centered at `79 N / 91 W` spans `N077`–`N081` and `W089`–`W093`), or use `python JAXA_Scrape.py --output-dir ALOS_Data`.
3. Alternatively, fetch tiles from OpenTopography's S3 mirror (requester-pays):
   ```bash
   aws s3 cp s3://raster/AW3D30/N037/ALPSMLC30_N037E137_DSM.tif ./ALOS_Data/ \
     --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request
   ```

### Moon: USGS LRO WAC GLD100 (100 m/pixel)
Clip any $5^\circ \times 5^\circ$ (or custom) region directly via HTTP without downloading the full global mosaic:

```bash
gdalwarp \
  -te -22.58 7.12 -17.58 12.12 \
  -te_srs "+proj=longlat +a=1737400 +b=1737400 +no_defs" \
  -t_srs "+proj=longlat +a=1737400 +b=1737400 +no_defs" \
  /vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_WAC_GLD100_DTM_79S79N_100m_v1.1.tif \
  Lunar_Test_Data/lunar_9.62_-20.08.tif
```

`LunarTile` accepts any `.tif` filename (the `lunar_{LAT}_{LON}.tif` convention is recommended for clarity; bounds and center coordinates are always read from the GeoTIFF header).

## Documentation

See [`documentation.pdf`](documentation.pdf) (built from [`documentation.tex`](documentation.tex)) for detailed function descriptions, algorithms, data access instructions, and **Appendix A: Earth vs. Moon Computational Differences**.
