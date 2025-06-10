import numpy as np
import fnmatch
import netCDF4 as nc
import os
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

        return [file]

    def setup_extract(self):
        '''
            Initialize the bookkeeping and get basic properties of the
            outputs (e.g., thermo variables, grid sizes, extents etc.)
            Should be called immediately after initialization of the class
        '''

        # Open the first dataset
        fname = self.iarr[0]
        with nc.Dataset(fname, 'r') as dset:
            # read in useful variables
            self.Ratmo = dset.planet_rgas
            self.Cp = dset.planet_cp
            self.xh2 = dset.planet_x_h2
            self.xhe = dset.planet_x_he
            self.x3 = dset.planet_x_3
            self.cpr = self.Cp / self.Ratmo
            try:
                self.p0 = dset.grid_press0
            except AttributeError:
                self.p0 = dset.planet_p0

            # Get info about gridbox
            self.gridni = dset.grid_ni  # nLongitudes
            self.gridnj = dset.grid_nj  # nLatituteds
            self.gridnk = dset.grid_nk  # nHeight

            # Grid extents
            self.gridlatbot = dset.grid_globe_latbot
            self.gridlattop = dset.grid_globe_lattop
            self.gridlonbot = dset.grid_globe_lonbot
            self.gridlontop = dset.grid_globe_lontop

            # vertical coordinates
            try:
                self.sigmatheta = dset.variables['hybrid_sigmatheta_h'][:]
                self.sigmatheta_u = dset.variables['hybrid_sigmatheta_u'][:]
                self.sigmatheta_v = dset.variables['hybrid_sigmatheta_v'][:]
                self.sigmatheta_pv = dset.variables['hybrid_sigmatheta_pv2'][:]
            except KeyError:
                self.p = dset.variables['p_h'][:]
                self.p_h = dset.variables['p_h'][:]
                self.p_u = dset.variables['p_u'][:]
                self.p_pv = dset.variables['p_pv2'][:]

            # useful for recalculating Ertel's PV
            self.omega = dset.planet_omega_sidereal
            self.grid_re = dset.grid_re
            self.grid_rp = dset.grid_rp

            self.dln = np.radians(dset.grid_dln)
            self.dlt = np.radians(dset.grid_dlt)

            # Lat/lon grids for different variable types
            self.lat_h = dset.variables["lat_h"][:]
            self.lon_h = dset.variables["lon_h"][:]

            self.lat_u = dset.variables["lat_u"][:]
            self.lon_u = dset.variables["lon_u"][:]

            self.lat_v = dset.variables["lat_v"][:]
            self.lon_v = dset.variables["lon_v"][:]

            try:
                self.lat_pv = dset.variables["lat_pv"][:]
                self.lon_pv = dset.variables["lon_pv"][:]
            except KeyError:
                self.lat_pv = dset.variables["lat_pv2"][:]
                self.lon_pv = dset.variables["lon_pv2"][:]

            try:
                self.gravity = np.array(dset.variables['gravity'])
            except KeyError:
                self.gravity = np.array(dset.variables['gravity2'])
            self.gave = self.gravity.mean()


        self.set_shape_factors()

        self.setup_time()

    def setup_time(self):
        '''
            Get all the timestamps for the outputs. This is slow
            if you have a lot of outputs. If this is slow, you can turn off
            `get_time` in the `setup_extract` function
        '''
        # set up sizes
        self.tarr = []
        self.tarr_file = []
        self.tarr_index_in_file = []
        for i, ti in enumerate(self.iarr):
            fname = self.iarr[i]
            with nc.Dataset(fname, 'r') as dset:
                self.tarr.extend(dset.variables['time'][:].tolist())
                self.tarr_file.extend([fname] * len(dset.variables['time'][:]))
                self.tarr_index_in_file.extend(list(range(len(dset.variables['time'][:]))))

    def get_variable_at_time(self, var, time):
        fname = self.tarr_file[time]

        with nc.Dataset(fname, 'r') as dset:
            if var == 'ertel_pv':
                return self.get_ertel_pv(time)

            if var not in dset.variables:
                raise KeyError(f'Dataset does not contain {var} at time {time} => {self.tarr[time]}')

            ind = self.tarr_index_in_file[time]

            if dset.variables[var].dimensions[0] == 'time':
                variable = dset.variables[var][ind, :]
            else:
                variable = dset.variables[var][:]

        return variable

    def get_attrs(self, time, attrs=None):
        '''
            Gets a specific attribute (or all attributes from a given extract
        '''
        fname = self.tarr_file[time]
        with nc.Dataset(fname, 'r') as dset:
            if attrs is None:
                attrs = dset.ncattrs()
            elif isinstance(attrs, str):
                attrs = [attr for attr in dset.ncattrs() if fnmatch.fnmatch(attr, attrs)]

            return {attr: getattr(dset, attr) for attr in attrs}
