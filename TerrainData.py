import h5py 
import numpy as np

class TerrainData():
    '''Designed to be a wrapper for processing h5 files for terrain data '''
    def __init__(self, lats, lons, elev, grad_x, grad_y, slope, slope_deg, shape=None):
        self.lats = lats
        self.lons = lons
        self.elev = elev
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.slope = slope
        self.slope_deg = slope_deg
        
        if shape is None:
            self.shape = self.lats.shape
        else:
            self.shape = shape

    def save_to_h5(self, file_name, dtype=np.float32, compression="gzip", compression_opts=6):
        """
        Save the TerrainData object to an HDF5 file. 
        """
        rows, cols = self.shape

        # Flatten arrays for storage
        lats_flat      = self.lats.flatten().astype(dtype)
        lons_flat      = self.lons.flatten().astype(dtype)
        elev_flat      = self.elev.flatten().astype(dtype)
        grad_x_flat    = self.grad_x.flatten().astype(dtype)
        grad_y_flat    = self.grad_y.flatten().astype(dtype)
        slope_flat     = self.slope.flatten().astype(dtype)
        slope_deg_flat = self.slope_deg.flatten().astype(dtype)

        # Stack as (rows*cols, 7)
        data_tile = np.stack([
            lats_flat, lons_flat, elev_flat,
            grad_x_flat, grad_y_flat, slope_flat, slope_deg_flat
        ], axis=1)

        # Save to HDF5
        with h5py.File(file_name, "w") as f:
            dset = f.create_dataset(
                "terrain_data",
                data=data_tile,
                dtype=dtype,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True
            )
            dset.attrs["headers"] = np.string_([
                "latitude", "longitude", "elevation",
                "gradient_x", "gradient_y", "slope", "slope_deg"
            ])

            # Save shape for reconstruction
            f.attrs["rows"] = rows
            f.attrs["cols"] = cols
            
    @staticmethod
    def load_terrain_data(file_name):
        """
        Load TerrainData from an HDF5 file.
        Returns a TerrainData object with 2D NumPy arrays.
        """
        with h5py.File(file_name, "r") as f:
            if "terrain_data" not in f:
                raise KeyError("Dataset 'terrain_data' not found in file.")
        
            # Load the flattened data
            data = f["terrain_data"][:]  # shape: (rows*cols, 7)
        
            # Read the original 2D shape from attributes
            rows, cols = f.attrs["rows"], f.attrs["cols"]
            shape = (rows, cols)

        # Reconstruct 2D arrays from flattened data
        return TerrainData(
            lats      = data[:, 0].reshape(shape),
            lons      = data[:, 1].reshape(shape),
            elev      = data[:, 2].reshape(shape),
            grad_x    = data[:, 3].reshape(shape),
            grad_y    = data[:, 4].reshape(shape),
            slope     = data[:, 5].reshape(shape),
            slope_deg = data[:, 6].reshape(shape),
            shape     = shape  # pass explicit shape
        )

    # ---- Getter Methods ----
    
    def get_latitudes(self):
        return self.lats

    def get_longitudes(self):
        return self.lons

    def get_elevation(self):
        return self.elev

    def get_gradient_x(self):
        return self.grad_x

    def get_gradient_y(self):
        return self.grad_y

    def get_slope(self):
        return self.slope

    def get_slope_deg(self):
        return self.slope_deg
    
