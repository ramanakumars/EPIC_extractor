import os

import netCDF4 as nc
import numpy as np

from .extractor import Extractor


class Extractor5(Extractor):
    """
    Reads extract.nc files from EPIC simulation and converts into a
    dictionary of variables

    Inputs
    ------
    imgfolder : String
        folder containing extract.nc files
    pltfolder : String
        folder to put plots in (not used)

    """

    def getextractmatch(self, folder):
        """
        Loops through the folder and finds extract files. Called by
         default, unles getextractmatch=False

        Raises
        ------
        AssertionError : when no files were found
        """

        file = os.path.join(folder, "extract.nc")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"extract.nc not found in {folder}")

        return np.asarray([file])

    def get_coordinates(self):
        """
        Get basic coordinates of the model and cache them for easy access
        """
        # vertical coordinates
        try:
            self.sigmatheta = self.get_variable_at_time("hybrid_sigmatheta_h", 0)
            self.sigmatheta_u = self.get_variable_at_time("hybrid_sigmatheta_u", 0)
            self.sigmatheta_v = self.get_variable_at_time("hybrid_sigmatheta_v", 0)
            self.sigmatheta_pv = self.get_variable_at_time("hybrid_sigmatheta_pv2", 0)
        except KeyError:
            self.p = self.get_variable_at_time("p_h", 0)
            self.p_h = self.get_variable_at_time("p_h", 0)
            self.p_u = self.get_variable_at_time("p_u", 0)
            self.p_pv = self.get_variable_at_time("p_pv2", 0)

        # Lat/lon grids for different variable types
        self.lat_h = self.get_variable_at_time("lat_h", 0)
        self.lon_h = self.get_variable_at_time("lon_h", 0)

        self.lat_u = self.get_variable_at_time("lat_u", 0)
        self.lon_u = self.get_variable_at_time("lon_u", 0)

        self.lat_v = self.get_variable_at_time("lat_v", 0)
        self.lon_v = self.get_variable_at_time("lon_v", 0)

        try:
            self.lat_pv = self.get_variable_at_time("lat_pv", 0)
            self.lon_pv = self.get_variable_at_time("lon_pv", 0)
        except KeyError:
            self.lat_pv = self.get_variable_at_time("lat_pv2", 0)
            self.lon_pv = self.get_variable_at_time("lon_pv2", 0)

    def setup_time(self) -> None:
        '''
        Get all the timestamps for the outputs. Also finds the index/file correspondence
        when having restart simulations
        '''
        # set up sizes
        self.time = []
        self.tarr_file = []
        self.tarr_index_in_file = []
        for i, ti in enumerate(self.files):
            fname = self.files[i]
            with nc.Dataset(fname, 'r') as dset:
                self.time.extend(dset.variables['time'][:].tolist())
                # EPIC 5 outputs have multiple timesteps per output file
                self.tarr_file.extend([fname] * len(dset.variables['time'][:]))
                self.tarr_index_in_file.extend(
                    list(range(len(dset.variables['time'][:])))
                )
