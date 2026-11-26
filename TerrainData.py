import h5py
import numpy as np


class TerrainData:
    """Wrapper for saving and loading 2-D terrain grids and derivatives in HDF5 format."""

    HEADERS = [
        "latitude",
        "longitude",
        "elevation",
        "gradient_x",
        "gradient_y",
        "slope",
        "slope_deg",
    ]

    def __init__(self, lats, lons, elev, grad_x, grad_y, slope, slope_deg, shape=None):
        """Store 2-D terrain coordinate, elevation, gradient, and slope arrays."""
        self.lats = lats
        self.lons = lons
        self.elev = elev
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.slope = slope
        self.slope_deg = slope_deg
        self.shape = self.lats.shape if shape is None else shape

    def save_to_h5(
        self,
        file_name: str,
        dtype=np.float32,
        compression: str = "gzip",
        compression_opts: int = 6,
    ):
        """Serialize all terrain arrays to a compressed HDF5 dataset named ``terrain_data``."""
        rows, cols = self.shape
        arrays = [
            self.lats,
            self.lons,
            self.elev,
            self.grad_x,
            self.grad_y,
            self.slope,
            self.slope_deg,
        ]
        data_tile = np.stack([arr.ravel().astype(dtype) for arr in arrays], axis=1)

        with h5py.File(file_name, "w") as f:
            dset = f.create_dataset(
                "terrain_data",
                data=data_tile,
                dtype=dtype,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True,
            )
            dset.attrs["headers"] = np.bytes_([h.encode("utf-8") for h in self.HEADERS])
            f.attrs["rows"] = rows
            f.attrs["cols"] = cols

    @staticmethod
    def load_terrain_data(file_name: str) -> "TerrainData":
        """Load a ``TerrainData`` instance from an HDF5 file created by ``save_to_h5``."""
        with h5py.File(file_name, "r") as f:
            if "terrain_data" not in f:
                raise KeyError("Dataset 'terrain_data' not found in file.")
            data = f["terrain_data"][:]
            rows, cols = int(f.attrs["rows"]), int(f.attrs["cols"])
            shape = (rows, cols)

        return TerrainData(
            lats=data[:, 0].reshape(shape),
            lons=data[:, 1].reshape(shape),
            elev=data[:, 2].reshape(shape),
            grad_x=data[:, 3].reshape(shape),
            grad_y=data[:, 4].reshape(shape),
            slope=data[:, 5].reshape(shape),
            slope_deg=data[:, 6].reshape(shape),
            shape=shape,
        )

    def get_latitudes(self):
        """Return the 2-D latitude array (degrees)."""
        return self.lats

    def get_longitudes(self):
        """Return the 2-D longitude array (degrees)."""
        return self.lons

    def get_elevation(self):
        """Return the 2-D elevation array (meters)."""
        return self.elev

    def get_gradient_x(self):
        """Return the zonal terrain gradient array (m/m)."""
        return self.grad_x

    def get_gradient_y(self):
        """Return the meridional terrain gradient array (m/m)."""
        return self.grad_y

    def get_slope(self):
        """Return the terrain slope magnitude array (m/m)."""
        return self.slope

    def get_slope_deg(self):
        """Return the terrain slope angle array (degrees)."""
        return self.slope_deg
