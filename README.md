# MIST_Maps

Terrain elevation, slope mapping, and ENU-frame horizon sky-view analysis for Earth (ALOS World 3D 30 m) and Lunar (USGS LRO WAC GLD100) Digital Elevation/Surface Models.

## Features

- **`ComboTile`**: Mosaics a grid (default $5 \times 5$) of 1-degree ALOS World 3D 30 m (`ALPSMLC30_*_DSM.tif`) GeoTIFF tiles, computes latitude-scaled gradients and slope angles, plots elevation/slope maps, exports to HDF5, and computes azimuthal horizon profiles in the local East-North-Up (ENU) frame.
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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
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

Name files `lunar_{LAT}_{LON}.tif` so `LunarTile` can validate the center coordinates.

## Documentation

See [`documentation.pdf`](documentation.pdf) (built from [`documentation.tex`](documentation.tex)) for detailed function descriptions, algorithms, data access instructions, and **Appendix A: Earth vs. Moon Computational Differences**.
