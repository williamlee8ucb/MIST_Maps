import rasterio
import rasterio.features
import rasterio.transform
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import xy
import pandas as pd
from rasterio.plot import show
from rasterio.merge import merge 
import h5py
from matplotlib.colors import LogNorm

class ComboTile():
    
    def __init__(self, line):
        
        # TODO: integrate functionality for just typing with a print statement; add error handling for if tile does not exist
        #coords = input("Enter a coordinate center with convention 000.000 S/000.000 W: ")
        lat_coord, lon_coord = line.split('/')
        
        lat_deg, lat_direction = lat_coord.split()
        lon_deg, lon_direction = lon_coord.split()

        #self.center = (lat_deg, lat_direction, lon_deg, lon_direction)
        self.center = (float(lon_deg), float(lat_deg))
        
        degree_combinations = self.surrounding_tiles(float(lat_deg), float(lon_deg))
        print(degree_combinations)
        
        file_paths = [f"ALPSMLC30_{lat_direction}{combo[0]:03d}{lon_direction}{combo[1]:03d}_DSM.tif" for combo in degree_combinations]
            
        src_files_to_mosaic = [rasterio.open(file_path) for file_path in file_paths[:2]] ## remove indexing for full coverage
        
        
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
            
         theta = np.degrees(np.arctan2(dlat, dlon))  # arctan2(y, x)
         theta = (theta + 360) % 360
         r = np.sqrt(dlat**2 + dlon**2)
            
         return r, theta
        
    def max_per_bin_fast(self, bin_width = 0.01):
        clon, clat = self.center
        meters_per_deg_lat = 111_320
        meters_per_deg_lon = 111_320 * np.cos(np.radians(clat))
        
        r, theta = self.compute_azimuth()
        theta = theta % 360
        center_x = int(len(self.elevations)/2)
        center_y = int(len(self.elevations[0])/2)
        
        print(center_x, center_y)
        r = r.flatten()
        theta = theta.flatten()
        z = self.elevations.flatten() - self.elevations[center_x][center_y] # FIX TO GENERALIZE
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
        print('sorted elevations', elev_angle)
    
        # output
        max_elev = np.full(nbins, np.nan)
        print('max elev length',len(max_elev))
    
        # first occurrence per bin = max elevation
        _, first = np.unique(bin_sorted, return_index=True)
        print('first', first)
        max_elev[bin_sorted[first]] = elev_sorted[first]
        print('elevation maxes',max_elev)

        print(f"len(bin_sorted): {len(bin_sorted)}")
        print(f"len(elev_sorted): {len(elev_sorted)}")
        print(f"unique bins: {len(np.unique(bin_sorted))}")
    
        return max_elev, elev_angle
        

    def max_per_bin_iterative(self, bin_width=0.01):
        clon, clat = self.center
        meters_per_deg_lat = 111_320
        meters_per_deg_lon = 111_320 * np.cos(np.radians(clat))
        
        r, theta = self.compute_azimuth()
        theta = theta % 360.0
        center_x = int(len(self.elevations)/2)
        center_y = int(len(self.elevations[0])/2)
        
        r = r.flatten()
        z = self.elevations.flatten() - self.elevations[center_x][center_y]
    
        bins = np.arange(0, 360 + bin_width/2, bin_width)
        nbins = len(bins) - 1
    
        max_angle = np.full(nbins, -np.inf)
        has_point = np.zeros(nbins, dtype=bool)
    
        # --- manual binning ---
        for i in range(len(theta)):
            if r[i] <= 0:
                continue  # avoid division/angle pathologies
    
            b = int(theta[i] // bin_width)
    
            # circular edge case: 360° == 0°
            if b == nbins:
                b = 0
    
            angle = np.degrees(np.arctan2(z[i], r[i]))
    
            if angle > max_angle[b]:
                max_angle[b] = angle
                has_point[b] = True
    
        elev_angle = np.full(nbins, np.nan)
        elev_angle[has_point] = max_angle[has_point]
    
        return elev_angle

    # Change 'theta' parameter to 'az' to denote azimuth 
    # adjust bin "tracking" to be the bin center rather than the bin edges 

# get angle created with the max_angle
# elev_angle = np.degrees(np.arctan2(max_elev, r)) -> get indices of maxes



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

   