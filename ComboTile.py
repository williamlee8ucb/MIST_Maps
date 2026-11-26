import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pymap3d as pm
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.transform import rowcol, xy
import scipy.interpolate as spi


def _get_lunar_ellipsoid() -> pm.Ellipsoid:
    """Return the IAU Moon reference ellipsoid, compatible across pymap3d versions."""
    try:
        return pm.Ellipsoid.from_name("moon")
    except (AttributeError, KeyError, ValueError):
        return pm.Ellipsoid(
            semimajor_axis=1738100.0,
            semiminor_axis=1736000.0,
            name="Moon IAU 2015",
        )


class ComboTile:
    """Mosaic and analyze a grid of ALOS World 3D 30 m (AW3D30) GeoTIFF DSM tiles."""

    EARTH_MEAN_RADIUS_M = 6_371_000.0
    METERS_PER_DEG_EARTH = 111_320.0

    def __init__(
        self,
        line: str,
        root_dir: str = "",
        grid_size: int = 5,
        center: Optional[list] = None,
        verbose: bool = True,
    ):
        """
        Load and merge a grid of ALOS DSM tiles centered on a coordinate string.

        Parameters
        ----------
        line : str
            Center coordinate string in ``"LAT DIR / LON DIR"`` format,
            e.g. ``"79 N / 91 W"`` or ``"037 N/137 E"``.
        root_dir : str, optional
            Directory containing ``ALPSMLC30_*_DSM.tif`` files.
        grid_size : int, optional
            Side length (in 1-degree tiles) of the square mosaic grid (default 5).
        center : list, optional
            Deprecated and ignored; retained only for backwards compatibility.
            ``self.center`` is always derived from the merged raster bounds.
        verbose : bool, optional
            If True, print loaded/missing tile summaries and CRS info.
        """
        _ = center  # retained for backwards call compatibility
        try:
            lat_coord, lon_coord = line.split("/")
            lat_deg_str, lat_direction = lat_coord.strip().split()
            lon_deg_str, lon_direction = lon_coord.strip().split()
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinate string {line!r}. Expected format 'LAT DIR / LON DIR', e.g. '79 N / 91 W'."
            ) from exc

        lat_direction = lat_direction.upper()
        lon_direction = lon_direction.upper()
        degree_combinations = self.surrounding_tiles(
            float(lat_deg_str), float(lon_deg_str), grid_size=grid_size
        )

        file_paths = [
            os.path.join(
                root_dir,
                f"ALPSMLC30_{lat_direction}{lat_i:03d}{lon_direction}{lon_i:03d}_DSM.tif",
            )
            if root_dir
            else f"ALPSMLC30_{lat_direction}{lat_i:03d}{lon_direction}{lon_i:03d}_DSM.tif"
            for lat_i, lon_i in degree_combinations
        ]

        src_files_to_mosaic = []
        failed_files = []
        successful_files = []

        for file_path in file_paths:
            try:
                src = rasterio.open(file_path)
                src_files_to_mosaic.append(src)
                successful_files.append(file_path)
            except FileNotFoundError:
                failed_files.append((file_path, "File not found"))
            except Exception as exc:
                failed_files.append((file_path, str(exc)))

        if verbose:
            if successful_files:
                print(f"{len(successful_files)} file(s) successfully opened.")
            if failed_files:
                print(f"{len(failed_files)} file(s) missing or failed to open:")
                for file_path, error in failed_files:
                    print(f"  - {file_path}: {error}")

        if not src_files_to_mosaic:
            raise FileNotFoundError(
                f"No DSM tiles could be opened for {line!r} in {root_dir!r}."
            )

        try:
            mosaic, self.transform = merge(src_files_to_mosaic)
            self.crs = src_files_to_mosaic[0].crs
            self.nodata = src_files_to_mosaic[0].nodata
        finally:
            for src in src_files_to_mosaic:
                src.close()

        self.elevations = mosaic[0]
        self.shape = self.elevations.shape

        rows, cols = self.shape
        tl = xy(self.transform, 0, 0)
        br = xy(self.transform, rows - 1, cols - 1)
        self.center = ((tl[0] + br[0]) / 2.0, (tl[1] + br[1]) / 2.0)  # (lon, lat)

        if verbose:
            print(f"CRS: {self.crs}, Shape: {self.shape}, Center (lon, lat): {self.center}")

    def _masked_elevations(self) -> np.ndarray:
        """Return elevation array as float with nodata values replaced by NaN."""
        elev = self.elevations.astype(float, copy=True)
        if self.nodata is not None:
            elev[elev == self.nodata] = np.nan
        return elev

    def _valid_mask(self, elev_flat: np.ndarray) -> np.ndarray:
        """Return a boolean mask of finite, non-nodata elevation samples."""
        valid = np.isfinite(elev_flat)
        if self.nodata is not None:
            valid &= elev_flat != float(self.nodata)
        return valid

    def get_lon_lat(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return 2-D arrays ``(lon, lat)`` in degrees for every pixel center."""
        rows, cols = self.elevations.shape
        row_idx = np.arange(rows)
        col_idx = np.arange(cols)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")
        lon, lat = xy(self.transform, row_grid, col_grid)
        return np.asarray(lon), np.asarray(lat)

    def xy_to_lonlat(self, x: int, y: int) -> Tuple[float, float]:
        """Convert pixel coordinates ``(col, row)`` to geographic ``(lon, lat)``."""
        return xy(self.transform, y, x)

    def lonlat_to_xy(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert geographic coordinates ``(lon, lat)`` to pixel ``(col, row)``."""
        row, col = rowcol(self.transform, lon, lat)
        return col, row

    def surrounding_tiles(
        self,
        lat_center: float,
        lon_center: float,
        tile_size: float = 1.0,
        grid_size: int = 5,
    ):
        """Return a list of integer ``(lat, lon)`` tile pairs for a surrounding grid."""
        half = grid_size // 2
        offsets = np.arange(-half, half + 1) * tile_size
        latitudes = lat_center + offsets
        longitudes = lon_center + offsets
        latlon_grid = np.stack(np.meshgrid(latitudes, longitudes, indexing="ij"), axis=-1)
        return [tuple(pair) for pair in latlon_grid.astype(int).reshape(-1, 2)]

    def tile_bounds(self) -> Tuple[float, float, float, float]:
        """Return ``(min_lon, max_lon, min_lat, max_lat)`` for the raster."""
        rows, cols = self.shape
        tl = xy(self.transform, 0, 0)
        tr = xy(self.transform, 0, cols - 1)
        bl = xy(self.transform, rows - 1, 0)
        br = xy(self.transform, rows - 1, cols - 1)
        lons = [tl[0], tr[0], bl[0], br[0]]
        lats = [tl[1], tr[1], bl[1], br[1]]
        return min(lons), max(lons), min(lats), max(lats)

    def get_elevation(self, row: int, col: int) -> np.ndarray:
        """Return ``np.array([lat, lon, elev])`` for a single pixel ``(row, col)``."""
        lon, lat = xy(self.transform, row, col)
        elev = self.elevations[row, col]
        return np.array([float(lat), float(lon), float(elev)])

    def compute_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute terrain elevation gradients ``(grad_x, grad_y)`` in m/m.

        Uses the spherical-Earth degree-to-meter scaling at the tile center latitude.
        Nodata pixels are converted to NaN prior to differentiation.
        """
        dx_deg = abs(self.transform.a)
        dy_deg = abs(self.transform.e)
        mean_lat = self.center[1]
        lat_rad = np.radians(mean_lat)

        dx_m = self.METERS_PER_DEG_EARTH * np.cos(lat_rad) * dx_deg
        dy_m = self.METERS_PER_DEG_EARTH * dy_deg

        elev = self._masked_elevations()
        grad_y, grad_x = np.gradient(elev, dy_m, dx_m)
        return grad_x, grad_y

    def compute_slope(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(grad_x, grad_y, slope, slope_deg)`` over the raster."""
        grad_x, grad_y = self.compute_gradient()
        slope = np.hypot(grad_x, grad_y)
        slope_deg = np.degrees(np.arctan(slope))
        return grad_x, grad_y, slope, slope_deg

    def map_table(self, want_grad: bool = False) -> np.ndarray:
        """
        Return a flattened ``(N, 3)`` or ``(N, 7)`` array of per-pixel attributes.

        Columns are ``[lat, lon, elev]`` when ``want_grad=False``, or
        ``[lat, lon, elev, grad_x, grad_y, slope, slope_deg]`` when ``want_grad=True``.
        """
        lons, lats = self.get_lon_lat()
        elev = self._masked_elevations()

        if want_grad:
            grad_x, grad_y, slope, slope_deg = self.compute_slope()
            return np.stack(
                [
                    lats.ravel(),
                    lons.ravel(),
                    elev.ravel(),
                    grad_x.ravel(),
                    grad_y.ravel(),
                    slope.ravel(),
                    slope_deg.ravel(),
                ],
                axis=1,
            )

        return np.stack([lats.ravel(), lons.ravel(), elev.ravel()], axis=1)

    def plot_map(self, ax=None):
        """Plot the elevation raster in geographic coordinates with nodata masked."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        masked_elev = np.ma.masked_invalid(self._masked_elevations())
        show(masked_elev, transform=self.transform, cmap="plasma", ax=ax)

        cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.7)
        cbar.set_label("Elevation (m)")
        ax.set_title("Elevation Map")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        return fig, ax

    def plot_slope(self, want_log: bool = False, ax=None, vmax: float = 15.0):
        """Visualize terrain slope angles (in degrees) at each pixel."""
        _, _, _, slope_deg = self.compute_slope()
        masked_slope = np.ma.masked_invalid(slope_deg)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        if want_log:
            max_slope = float(np.nanmax(slope_deg))
            norm = LogNorm(vmin=0.1, vmax=max(max_slope, 0.2))
            show(masked_slope, transform=self.transform, cmap="inferno", norm=norm, ax=ax)
        else:
            show(
                masked_slope,
                transform=self.transform,
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
                ax=ax,
            )

        cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.7)
        cbar.set_label("Slope (degrees)")
        ax.set_title("Slope Map")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        return fig, ax

    def save_data(self, file_name: str, compute_gradients: bool = True):
        """Build a ``TerrainData`` object from this tile and save it to an HDF5 file."""
        from TerrainData import TerrainData

        lons, lats = self.get_lon_lat()
        if compute_gradients:
            grad_x, grad_y, slope, slope_deg = self.compute_slope()
        else:
            zeros = np.zeros_like(self.elevations, dtype=float)
            grad_x = grad_y = slope = slope_deg = zeros

        terrain = TerrainData(
            lats=lats,
            lons=lons,
            elev=self.elevations,
            grad_x=grad_x,
            grad_y=grad_y,
            slope=slope,
            slope_deg=slope_deg,
            shape=self.elevations.shape,
        )
        terrain.save_to_h5(file_name=file_name)

    @staticmethod
    def _estimate_reference_elevation(
        east: np.ndarray,
        north: np.ndarray,
        elev_flat: np.ndarray,
        valid_mask: np.ndarray,
        radius0_meters: float,
        method: str = "cubic",
        fallback_value: Optional[float] = None,
    ) -> float:
        """Interpolate ground elevation at the observer origin ``(E=0, N=0)``."""
        r = np.hypot(east, north)
        m0 = valid_mask & (r <= radius0_meters)
        min_pts = 4 if method == "cubic" else 3

        if m0.sum() >= min_pts:
            elref_raw = spi.griddata(
                np.column_stack((east[m0], north[m0])),
                elev_flat[m0],
                np.array([[0.0, 0.0]]),
                method=method,
            )
            elref = float(elref_raw[0])
            if not np.isnan(elref):
                return elref
            if fallback_value is not None:
                return float(fallback_value)
            return float(np.median(elev_flat[m0]))
        if m0.sum() > 0:
            if fallback_value is not None:
                return float(fallback_value)
            return float(np.median(elev_flat[m0]))
        if fallback_value is not None:
            return float(fallback_value)

        valid_r = np.where(valid_mask, r, np.inf)
        return float(elev_flat[np.argmin(valid_r)])

    def _bin_azimuth_max(
        self,
        az_deg: np.ndarray,
        el_deg: np.ndarray,
        rm: np.ndarray,
        dz: np.ndarray,
        flat_idx: np.ndarray,
        bin_width: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bin per-pixel horizon angles by azimuth with 0/360 wraparound handling."""
        az_ext = np.concatenate([az_deg, az_deg + 360.0])
        el_ext = np.concatenate([el_deg, el_deg])
        r_ext = np.concatenate([rm, rm])
        z_ext = np.concatenate([dz, dz])
        idx_ext = np.concatenate([flat_idx, flat_idx])

        az_centers = np.arange(0.0, 360.0 + bin_width / 2.0, bin_width)
        half = bin_width / 2.0
        nbins = len(az_centers)

        max_elev = np.full(nbins, np.nan)
        max_r = np.full(nbins, np.nan)
        max_z = np.full(nbins, np.nan)
        max_coords = np.full((nbins, 2), np.nan)

        ncols = self.elevations.shape[1]
        for i, azc in enumerate(az_centers):
            sel = (az_ext >= azc - half) & (az_ext < azc + half)
            if not np.any(sel):
                continue

            el_bin = el_ext[sel]
            best = int(np.argmax(el_bin))

            max_elev[i] = el_bin[best]
            max_r[i] = r_ext[sel][best]
            max_z[i] = z_ext[sel][best]

            orig_flat = int(idx_ext[sel][best])
            row_px = orig_flat // ncols
            col_px = orig_flat % ncols
            max_coords[i] = self.xy_to_lonlat(col_px, row_px)

        return az_centers, max_elev, max_r, max_z, max_coords

    def max_elev_enu(
        self,
        bin_width: float = 10.0,
        radius_max_meters: float = 20000.0,
        radius0_meters: float = 200.0,
        lon0: Optional[float] = None,
        lat0: Optional[float] = None,
        include_curvature: bool = False,
    ):
        """
        Compute the azimuthal horizon elevation profile in the local ENU frame.

        Parameters
        ----------
        bin_width : float, optional
            Azimuth bin width in degrees (default 10.0).
        radius_max_meters : float, optional
            Maximum horizontal radius from the observer in meters (default 20000).
        radius0_meters : float, optional
            Local neighborhood radius for observer elevation interpolation (default 200).
        lon0, lat0 : float
            Observer longitude and latitude in degrees (must be provided explicitly).
        include_curvature : bool, optional
            If True, use the 3-D ENU ``up`` coordinate relative to ``elref`` (which
            includes Earth ellipsoid curvature). If False (default), use flat elevation
            differences ``elev - elref`` for compatibility with short-baseline analyses.

        Returns
        -------
        az_centers, max_elev, max_r, max_z, max_coords
        """
        if lon0 is None or lat0 is None:
            raise ValueError(
                "lon0 and lat0 must be provided explicitly. "
                "self.center is the mosaic geometric midpoint and is unlikely "
                "to be the intended observation site."
            )

        lon, lat = self.get_lon_lat()
        lon_flat = np.asarray(lon).ravel()
        lat_flat = np.asarray(lat).ravel()
        elev_flat = self.elevations.ravel().astype(float)
        valid_mask = self._valid_mask(elev_flat)

        east, north, _ = pm.geodetic2enu(
            lat_flat, lon_flat, elev_flat, lat0, lon0, 0.0
        )
        r = np.hypot(east, north)
        elref = self._estimate_reference_elevation(
            east, north, elev_flat, valid_mask, radius0_meters, method="cubic"
        )

        if include_curvature:
            east, north, up = pm.geodetic2enu(
                lat_flat, lon_flat, elev_flat, lat0, lon0, elref
            )
            r = np.hypot(east, north)
            dz_all = up
        else:
            dz_all = elev_flat - elref

        theta = np.degrees(np.arctan2(east, north))  # -180..+180, N=0°, E=+90°
        mask = valid_mask & (r > 0) & (r <= radius_max_meters)
        flat_idx = np.where(mask)[0]
        dz = dz_all[mask]
        rm = r[mask]
        el_deg = np.degrees(np.arctan2(dz, rm))
        az_deg = theta[mask]

        return self._bin_azimuth_max(az_deg, el_deg, rm, dz, flat_idx, bin_width)


class JTile(ComboTile):
    """Load and analyze a single ALOS World 3D 30 m GeoTIFF DSM file without mosaicking."""

    def __init__(self, file_name: str, root_dir: str = "ALOS_Data"):
        """
        Initialize a JTile from a coordinate string or a DSM filename.

        Parameters
        ----------
        file_name : str
            Either ``"LAT DIR / LON DIR"`` (e.g. ``"079 N / 091 W"``) or a direct
            ``.tif`` filename (e.g. ``"ALPSMLC30_N079W091_DSM.tif"``).
        root_dir : str, optional
            Directory containing the DSM file (default ``"ALOS_Data"``).
        """
        if file_name.lower().endswith(".tif"):
            tif_name = file_name
        else:
            try:
                lat_coord, lon_coord = file_name.split("/")
                lat_deg, lat_direction = lat_coord.strip().split()
                lon_deg, lon_direction = lon_coord.strip().split()
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinate or filename {file_name!r}."
                ) from exc
            tif_name = (
                f"ALPSMLC30_{lat_direction.upper()}{int(float(lat_deg)):03d}"
                f"{lon_direction.upper()}{int(float(lon_deg)):03d}_DSM.tif"
            )

        file_path = os.path.join(root_dir, tif_name) if root_dir else tif_name

        with rasterio.open(file_path) as dsm:
            self.elevations = dsm.read(1)
            self.transform = dsm.transform
            self.nodata = dsm.nodata
            self.crs = dsm.crs
            self.shape = self.elevations.shape

        rows, cols = self.elevations.shape
        tl = xy(self.transform, 0, 0)
        br = xy(self.transform, rows - 1, cols - 1)
        self.center = ((tl[0] + br[0]) / 2.0, (tl[1] + br[1]) / 2.0)


class LunarTile(ComboTile):
    """Load and analyze a USGS LRO WAC GLD100 lunar GeoTIFF DEM tile."""

    LUNAR_RADIUS_M = 1_737_400.0

    def __init__(self, file_name: str, root_dir: str = "Lunar_Test_Data", verbose: bool = True):
        """
        Initialize a LunarTile from a clipped USGS LRO WAC GLD100 GeoTIFF.

        Expected filename format: ``lunar_LAT_LON.tif`` (e.g. ``lunar_9.62_-20.08.tif``).
        """
        base_name = os.path.basename(file_name).replace(".tif", "")
        parts = base_name.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid filename format {file_name!r}; expected 'lunar_LAT_LON.tif'."
            )
        # Validate numeric coordinates in filename
        float(parts[1])
        float(parts[2])

        file_path = os.path.join(root_dir, file_name) if root_dir else file_name

        with rasterio.open(file_path) as src:
            self.elevations = src.read(1)
            self.transform = src.transform
            self.nodata = src.nodata
            self.shape = self.elevations.shape
            self.crs = src.crs
            bounds = src.bounds

            self.geo_bounds = {
                "min_lon": bounds.left,
                "max_lon": bounds.right,
                "min_lat": bounds.bottom,
                "max_lat": bounds.top,
            }

            rows, cols = self.shape
            row_idx = np.arange(rows)
            col_idx = np.arange(cols)
            row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")
            lon_grid, lat_grid = xy(self.transform, row_grid, col_grid)
            self._lon_grid = np.asarray(lon_grid)
            self._lat_grid = np.asarray(lat_grid)

        center_lon = (self.geo_bounds["min_lon"] + self.geo_bounds["max_lon"]) / 2.0
        center_lat = (self.geo_bounds["min_lat"] + self.geo_bounds["max_lat"]) / 2.0
        self.center = (center_lon, center_lat)
        self.lunar_radius = self.LUNAR_RADIUS_M
        self.geo_resolution = {
            "lon": (self.geo_bounds["max_lon"] - self.geo_bounds["min_lon"]) / cols,
            "lat": (self.geo_bounds["max_lat"] - self.geo_bounds["min_lat"]) / rows,
        }

        if verbose:
            print(f"Loaded {file_name} | CRS: {self.crs}")
            print(f"  Center: ({center_lat:.4f}°, {center_lon:.4f}°), Shape: {self.shape}")
            print(
                f"  Bounds: Lat [{self.geo_bounds['min_lat']:.4f}, {self.geo_bounds['max_lat']:.4f}], "
                f"Lon [{self.geo_bounds['min_lon']:.4f}, {self.geo_bounds['max_lon']:.4f}]"
            )
            print(f"  Resolution: ~{self.geo_resolution['lat']:.6f}°/px")

    def get_lon_lat(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return pre-computed geographic ``(lon, lat)`` coordinate grids in degrees."""
        return self._lon_grid, self._lat_grid

    def plot_map(self, ax=None):
        """Plot the lunar elevation raster in geographic degrees with nodata masked."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        masked_elevations = np.ma.masked_invalid(self._masked_elevations())
        extent = [
            self.geo_bounds["min_lon"],
            self.geo_bounds["max_lon"],
            self.geo_bounds["min_lat"],
            self.geo_bounds["max_lat"],
        ]

        im = ax.imshow(
            masked_elevations,
            extent=extent,
            cmap="plasma",
            origin="upper",
            aspect="auto",
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label("Elevation (m)")
        ax.set_title("Lunar Elevation Map")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(True, alpha=0.3)
        return fig, ax

    def compute_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute lunar terrain gradients ``(grad_x, grad_y)`` in m/m."""
        dx_deg = abs(self.transform.a)
        dy_deg = abs(self.transform.e)
        mean_lat = self.center[1]

        meters_per_deg_lat = np.radians(1.0) * self.lunar_radius
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(mean_lat))

        dx_m = meters_per_deg_lon * dx_deg
        dy_m = meters_per_deg_lat * dy_deg

        elev = self._masked_elevations()
        grad_y, grad_x = np.gradient(elev, dy_m, dx_m)
        return grad_x, grad_y

    def max_elev_enu(
        self,
        bin_width: float = 10.0,
        radius_max_meters: float = 100000.0,
        radius0_meters: float = 1000.0,
        radius_min_meters: float = 500.0,
        lon0: Optional[float] = None,
        lat0: Optional[float] = None,
        include_curvature: bool = True,
        verbose: bool = False,
    ):
        """
        Compute the azimuthal lunar horizon elevation profile in the local ENU frame.

        Because ``pymap3d.geodetic2enu`` projects terrain onto the tangent plane of the
        IAU lunar ellipsoid, the ENU ``up`` coordinate natively incorporates the downward
        surface curvature drop (approximately ``-r^2 / (2 * R_moon)``).

        Parameters
        ----------
        bin_width : float, optional
            Azimuth bin width in degrees (default 10.0).
        radius_max_meters : float, optional
            Maximum horizontal radius from observer in meters (default 100000).
        radius0_meters : float, optional
            Local neighborhood radius for observer elevation refinement (default 1000).
        radius_min_meters : float, optional
            Minimum horizontal distance guard in meters (default 500).
        lon0, lat0 : float
            Observer longitude and latitude in degrees (must be provided explicitly).
        include_curvature : bool, optional
            If True (default), use the true 3-D ENU ``up`` coordinate directly (which
            includes lunar curvature). If False, use flat elevation differences
            ``elev - elref`` (ignoring curvature).
        verbose : bool, optional
            If True, print diagnostic reference elevation and extreme-pixel summaries.

        Returns
        -------
        el_deg, az_deg, az_centers, max_elev, max_r, max_z, max_coords
        """
        if lon0 is None or lat0 is None:
            raise ValueError(
                "lon0 and lat0 must be provided explicitly. "
                "self.center is the mosaic geometric midpoint and is unlikely "
                "to be the intended observation site."
            )

        clon, clat = lon0, lat0
        lunar_ellipsoid = _get_lunar_ellipsoid()

        lon, lat = self.get_lon_lat()
        lon_flat = np.asarray(lon).ravel()
        lat_flat = np.asarray(lat).ravel()
        elev_flat = self.elevations.ravel().astype(float)
        valid_mask = self._valid_mask(elev_flat)

        # Bilinear seed interpolation for observer elevation
        col_f = (clon - self.transform.c) / self.transform.a
        row_f = (clat - self.transform.f) / self.transform.e
        col0, row0 = int(np.floor(col_f)), int(np.floor(row_f))
        col1, row1 = col0 + 1, row0 + 1
        dc, dr = np.clip(col_f - col0, 0.0, 1.0), np.clip(row_f - row0, 0.0, 1.0)

        nrows, ncols = self.elevations.shape
        col0_c, col1_c = int(np.clip(col0, 0, ncols - 1)), int(np.clip(col1, 0, ncols - 1))
        row0_c, row1_c = int(np.clip(row0, 0, nrows - 1)), int(np.clip(row1, 0, nrows - 1))

        corners = np.array(
            [
                self.elevations[row0_c, col0_c],
                self.elevations[row0_c, col1_c],
                self.elevations[row1_c, col0_c],
                self.elevations[row1_c, col1_c],
            ],
            dtype=float,
        )
        if self.nodata is not None:
            corners[corners == float(self.nodata)] = np.nan

        if np.all(np.isfinite(corners)):
            elref_seed = float(
                corners[0] * (1.0 - dc) * (1.0 - dr)
                + corners[1] * dc * (1.0 - dr)
                + corners[2] * (1.0 - dc) * dr
                + corners[3] * dc * dr
            )
        else:
            elref_seed = float(np.nanmedian(elev_flat[valid_mask]))

        # Seed ENU frame to define local neighborhood around observer
        east_seed, north_seed, _ = pm.geodetic2enu(
            lat_flat, lon_flat, elev_flat, clat, clon, elref_seed, ell=lunar_ellipsoid
        )
        elref = self._estimate_reference_elevation(
            east_seed,
            north_seed,
            elev_flat,
            valid_mask,
            radius0_meters,
            method="linear",
            fallback_value=elref_seed,
        )

        # Final ENU frame centered at (clat, clon, elref)
        east, north, up = pm.geodetic2enu(
            lat_flat, lon_flat, elev_flat, clat, clon, elref, ell=lunar_ellipsoid
        )
        r = np.hypot(east, north)
        theta = np.degrees(np.arctan2(east, north))

        mask = valid_mask & (r > radius_min_meters) & (r <= radius_max_meters)
        flat_idx = np.where(mask)[0]
        rm = r[mask]
        dz = up[mask] if include_curvature else (elev_flat[mask] - elref)

        el_deg = np.degrees(np.arctan2(dz, rm))
        az_deg = theta[mask]

        if verbose and len(el_deg) > 0:
            obs_row = int(np.clip(round(row_f), 0, nrows - 1))
            obs_col = int(np.clip(round(col_f), 0, ncols - 1))
            print(f"elref: {elref:.2f} m (nearest pixel: {self.elevations[obs_row, obs_col]})")
            print(f"el_deg range: [{el_deg.min():.2f}°, {el_deg.max():.2f}°]")

        az_centers, max_elev, max_r, max_z, max_coords = self._bin_azimuth_max(
            az_deg, el_deg, rm, dz, flat_idx, bin_width
        )
        return el_deg, az_deg, az_centers, max_elev, max_r, max_z, max_coords
