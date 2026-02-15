import rasterio
import rasterio.features
import rasterio.transform
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import xy, rowcol
import pandas as pd
from rasterio.plot import show
from rasterio.merge import merge 
import h5py
from matplotlib.colors import LogNorm

class ComboTile():
    def __init__(self, line, root_dir = '', center = []):
        # TODO: integrate functionality for just typing with a print statement; add error handling for if tile does not exist
        #coords = input("Enter a coordinate center with convention 000.000 S/000.000 W: ")
        lat_coord, lon_coord = line.split('/')
        lat_deg, lat_direction = lat_coord.split()
        lon_deg, lon_direction = lon_coord.split()
        #self.center = (lat_deg, lat_direction, lon_deg, lon_direction)
        
        if center:
            self.center = (center[0], center[1])
        else:
            self.center = (float(lon_deg), float(lat_deg))

        degree_combinations = self.surrounding_tiles(float(lat_deg), float(lon_deg))
        print(degree_combinations)
        file_paths = [f"{root_dir}/ALPSMLC30_{lat_direction}{combo[0]:03d}{lon_direction}{combo[1]:03d}_DSM.tif" for combo in degree_combinations]

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
                print(f"Warning: Could not find {file_path}")
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"Warning: Failed to open {file_path}: {e}")

        if successful_files:
            print(f"\n{len(successful_files)} file(s) successfully opened:")
            for file_path in successful_files:
                print(f"  - {file_path}")

        if failed_files:
            print(f"\n{len(failed_files)} file(s) failed to open:")
            for file_path, error in failed_files:
                print(f"  - {file_path}: {error}")

        self.mosaic, self.transform = merge(src_files_to_mosaic)
        self.elevations = self.mosaic[0]
        self.shape = self.mosaic[0].shape

    def get_lon_lat(self):
        '''lat = y, lon = x'''
        rows, cols = self.elevations.shape

        # Arrays of row indices and column indices
        row_idx = np.arange(rows)
        col_idx = np.arange(cols)
        
        # Create 2D meshgrid
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")
        
        # Use rasterio.transform.xy to convert all pixel centers
        lon, lat = xy(self.transform, row_grid, col_grid)
        
        return lon, lat
    
    def xy_to_lonlat(self, x, y):  # x=col, y=row
        return xy(self.transform, y, x)  
    
    def lonlat_to_xy(self, lon, lat):
        row, col = rowcol(self.transform, lon, lat)
        return col, row  



    def surrounding_tiles(self, lat_center, lon_center, tile_size=1.0, grid_size=5):
        """Return an array of (lat, lon) pairs for a grid of surrounding tiles."""
        half = grid_size // 2
        offsets = np.arange(-half, half + 1) * tile_size

        # Each combination of offset lat/lon
        latitudes = lat_center + offsets
        longitudes = lon_center + offsets

        latlon_grid = np.stack(np.meshgrid(latitudes, longitudes, indexing='ij'), axis=-1)
        # shape: (grid_size, grid_size, 2), where [:,:,0] = lat, [:,:,1] = lon

        combos = [tuple(pair) for pair in latlon_grid.astype(int).reshape(-1, 2)]

        return combos
        
    # returns the minimum and maximum bounds for the latitude and longitude
    def tile_bounds(tile):
        rows, cols = tile.shape

        tl = xy(tile.transform, 0, 0)
        tr = xy(tile.transform, 0, cols-1)
        bl = xy(tile.transform, rows-1, 0)
        br = xy(tile.transform, rows-1, cols-1)
    
        lons = [tl[0], tr[0], bl[0], br[0]]
        lats = [tl[1], tr[1], bl[1], br[1]]
        return min(lons), max(lons), min(lats), max(lats) 
    
    def get_elevation(self, row, col):
        '''Returns the longitude, latitude, and elevation as a 3-element array for a specific tile object'''
        lon, lat = xy(self.transform, row, col)
        elev = self.elevations[row, col]
        return np.array([float(lat), float(lon), float(elev)])
    
    def map_table(self, want_grad: bool):
        '''Uses vectorization to create a list of the latitude, longitude, and elevation for each pixel in the tile'''
        rows, cols = np.indices(self.shape)
        lons, lats = xy(self.transform, rows, cols)

        if want_grad:
            grad_x, grad_y, slope, slope_deg = self.compute_gradient()
            return np.stack([
                lats.flatten(),
                lons.flatten(),
                self.elevations.flatten(),
                grad_x.flatten(),
                grad_y.flatten(),
                slope.flatten(),
                slope_deg.flatten()
            ], axis=1)

        return np.stack([lats.flatten(), lons.flatten(), self.elevations.flatten()], axis = 1)
        
    def plot_map(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Show the raster on this axes
        im = show(self.elevations, transform=self.transform, cmap='plasma', ax=ax)
        # ADD: option for log_scal

        # Grab the image object created by show
        im = ax.images[0]  # THIS is the AxesImage

        # Now attach the colorbar to the figure
        cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.7)
        cbar.set_label("Elevation (m)")

        ax.set_title("Elevation Map")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
    def compute_gradient(self):
        dx_deg = self.transform.a
        dy_deg = -self.transform.e  # make positive
    
        # Approximate pixel size in meters
        # Use mean latitude of the tile for lon->meter scaling
        nrows, ncols = self.elevations.shape
        y0 = self.transform[3]
        mean_lat = y0 - (nrows / 2) * dy_deg
        lat_rad = np.radians(mean_lat)

        dx_m = 111_320 * np.cos(lat_rad) * dx_deg
        dy_m = 111_320 * dy_deg

        # Compute gradients in meters
        grad_y, grad_x = np.gradient(self.elevations, dy_m, dx_m)

        # Slope magnitude and slope angle in degrees
        #slope = np.sqrt(grad_x**2 + grad_y**2)
        #slope_deg = np.degrees(np.arctan(slope))

        return grad_x, grad_y
        # should be 3D -> at a given elevation point, what is the tangent plane? 
        # get quantities necessary to define a tangent plane at a given point (center and normal vector)
        # NEED NORMAL VECTOR

    def plot_slope(self, want_log: bool, ax=None):
        """
        Visualize slope angles (in degrees) at each pixel.
        """
        grad_x, grad_y = self.compute_gradient()
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope_deg = np.degrees(np.arctan(slope))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
    
        # Use rasterio.plot.show to display slope with transform
        if want_log:
            im = show(
                slope_deg,
                transform=self.transform,
                cmap='inferno',
                norm=LogNorm(vmin=0.1, vmax=np.nanmax(slope_deg)),
                ax=ax,
                title="Slope Angle Map (degrees)"
            )
        else:
            im = show(
                slope_deg,
                transform=self.transform,
                cmap='inferno',
                vmin = 0, vmax = np.nanmax(15),
                ax=ax,
                title="Slope Angle Map (degrees)"
            )            

        cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.7)
        cbar.set_label("Slope (degrees)")
        ax.set_title("Slope Map")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def save_data(self, file_name, compute_gradients=True):
        """
        Build a TerrainData object from this tile and save to an HDF5 file.
        """
        rows, cols = np.indices(self.shape)
        lons, lats = xy(self.transform, rows, cols)

        if compute_gradients:
            grad_x, grad_y, slope, slope_deg = self.compute_gradient()
        else:
            grad_x = grad_y = slope = slope_deg = np.zeros_like(self.elevations)

        terrain = TerrainData(
            lats=lats,
            lons=lons,
            elev=self.elevations,
            grad_x=grad_x,
            grad_y=grad_y,
            slope=slope,
            slope_deg=slope_deg,
            shape=self.elevations.shape
        )

        terrain.save_to_h5(file_name=file_name)
       
    def compute_azimuth(self):
         
         lon, lat = self.get_lon_lat() # x, y
         clon, clat = self.center

         meters_per_deg_lat = 111_320
         meters_per_deg_lon = 111_320 * np.cos(np.radians(clat))
        
            
         dlat = (lat - clat) * meters_per_deg_lat
         dlon = (lon - clon) * meters_per_deg_lon # convert back to non-negative for JTile array
            
         theta = np.degrees(np.arctan2(dlon, dlat))  # arctan2(y, x)
         theta = (theta + 360) % 360
         r = np.sqrt(dlat**2 + dlon**2)
            
         return r, theta
        
    def max_per_bin_fast(self, bin_width = 0.01):
        clon, clat = self.center
        meters_per_deg_lat = 111_320
        meters_per_deg_lon = 111_320 * np.cos(np.radians(clat))
        
        r, theta = self.compute_azimuth()
        theta = theta % 360
        center_x, center_y = self.lonlat_to_xy(lon = clon, lat = clat)
        
        print(center_x, center_y)
        r = r.flatten()
        theta = theta.flatten()
        z = self.elevations.flatten() - self.elevations[int(center_y), int(center_x)]
        print(z)
        # elevations data

        bins = np.arange(0, 360 + bin_width/2, bin_width)
        nbins = len(bins) - 1
        
        # Bin assignment
        bin_idx = (theta // bin_width).astype(int)

        
        bin_idx[bin_idx == nbins] = nbins - 1
        print("Bin Index:", bin_idx)
    
        # validity
        valid = (r > 0) & (bin_idx >= 0) & (bin_idx < nbins)
        print("Validation:", valid)
    
        elev_angle = np.degrees(np.arctan2(z[valid], r[valid]))
        print("Elevations angle: ", elev_angle)
    
        order = np.lexsort((-elev_angle, bin_idx[valid]))
        # returns a sorted array of indices, sorted by angle in descending order (-) and by bins in increasing ordr
        # data product looks like elevation angles sorted per bin, and bins are sorted
        print("ordering index mask", order)

        
        bin_sorted  = bin_idx[valid][order]
        elev_sorted = elev_angle[order]
        r_sorted = r[valid][order]
        z_sorted = z[valid][order]
        print('sorted elevations', elev_angle)
    
        # output
        max_elev = np.full(nbins, np.nan)
        max_r = np.full(nbins, np.nan)
        max_z = np.full(nbins, np.nan)
        print('max elev length',len(max_elev))
    
        # first occurrence per bin = max elevation
        _, first = np.unique(bin_sorted, return_index=True)
        print('first', first)
        max_elev[bin_sorted[first]] = elev_sorted[first]
        max_r[bin_sorted[first]] = r_sorted[first]
        max_z[bin_sorted[first]] = z_sorted[first]
        print('elevation maxes',max_elev)

        print(f"len(bin_sorted): {len(bin_sorted)}")
        print(f"len(elev_sorted): {len(elev_sorted)}")
        print(f"unique bins: {len(np.unique(bin_sorted))}")

        # Get the original flattened indices of the max elevations
        valid_indices = np.where(valid)[0]
        max_indices = valid_indices[order[first]]

        # Convert flattened indices back to 2D pixel coordinates (row, col)
        rows = max_indices // self.elevations.shape[1]
        cols = max_indices % self.elevations.shape[1]

        # Convert pixel coordinates to lon/lat
        max_coords = np.array([self.xy_to_lonlat(col, row) for col, row in zip(cols, rows)])

    
        return max_elev, max_r, max_z, max_coords

    def plot_max_per_bin(self, max_elev = [], max_r=[], max_z=[], bin_width=0.01):
        """
        Plot the outputs from max_per_bin_fast as subplots.
        
        Parameters:
        - max_elev: elevation angles (degrees)
        - max_r: radial distances (meters)
        - max_z: normalized elevations (meters)
        - bin_width: bin width in degrees (for x-axis)
        """
        if not max_elev or not max_r or not max_z:
            max_elev, max_r, max_z, _ = self.max_per_bin_fast(bin_width)

        nbins = len(max_elev)
        azimuths = np.arange(nbins) * bin_width
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Elevation Angle
        axes[0].plot(azimuths, max_elev, linewidth=1.5)
        axes[0].set_ylabel("Elevation Angle (degrees)")
        axes[0].set_title("Maximum Elevation Angle per Azimuth")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 360)
        
        # Radial Distance
        axes[1].plot(azimuths, max_r, linewidth=1.5, color='orange')
        axes[1].set_ylabel("Distance (meters)")
        axes[1].set_title("Distance to Maximum Elevation Point per Azimuth")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 360)
        
        # Normalized Elevation (Z)
        axes[2].plot(azimuths, max_z, linewidth=1.5, color='green')
        axes[2].set_ylabel("Elevation Difference (meters)")
        axes[2].set_xlabel("Azimuth (degrees)")
        axes[2].set_title("Normalized Elevation (above center) per Azimuth")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, 360)
        
        plt.tight_layout()
        return fig, axes


class JTile(ComboTile):
    def __init__(self, file_name, root_dir="ALOS_Data"):
        """
        Initialize JTile from a single DSM file.
        """
        print(file_name)
        lat_coord, lon_coord = file_name.split('/')
        
        lat_deg, lat_direction = lat_coord.split()
        lon_deg, lon_direction = lon_coord.split()

        self.center = (float(lat_deg), lat_direction, float(lon_deg), lon_direction)

        file_path = f"{root_dir}/ALPSMLC30_{lat_direction}{int(float(lat_deg)):03d}{lon_direction}{int(float(lon_deg)):03d}_DSM.tif"
        
        # Open the raster file
        with rasterio.open(file_path) as dsm:
            self.elevations = dsm.read(1)      # elevation array
            self.transform = dsm.transform     # affine transform
            self.nodata = dsm.nodata           # nodata value
            self.shape = self.elevations.shape # (rows, cols)

        # If needed, extract center coordinates from the raster bounds
        # e.g., for convenience:
        rows, cols = self.elevations.shape
        tl = xy(self.transform, 0, 0) # top left
        br = xy(self.transform, rows-1, cols-1) # bottom right
        center_lat = (tl[1] + br[1]) / 2
        center_lon = (tl[0] + br[0]) / 2
        self.center = (center_lon, center_lat)

   