import numpy as np
import netCDF4 as nc
import os
import glob
from collections.abc import Iterable


class Extractor():
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

    def __init__(self, imgfolder=".", validate_files=True):
        """
        Initializes the extractor
        """
        self.imgfolder = os.path.abspath(imgfolder)

        if validate_files:
            self.getextractmatch()

        self.setup_extract()

    def getextractmatch(self):
        """
        Loops through the folder and finds extract files. Called by
         default, unles getextractmatch=False

        Raises
        ------
        AssertionError : when no files were found
        """

        files = sorted(glob.glob(self.imgfolder + "/extract*.nc"))
        iarr = []
        for file in files:
            iarr.append(file)

        assert len(iarr) > 0., "No files found!"

        self.iarr = np.asarray(iarr)
        self.iarrset = True

    def add_restarts(self, resfolder, start=-1):
        """
        adds extract files from a restart folder
        """
        if not os.path.exists(resfolder):
            return

        self.iarr = self.iarr[:start].tolist()
        files = sorted(glob.glob(resfolder + "/extract*.nc"))
        for file in files:
            self.iarr.append(file)

        self.iarr = np.asarray(self.iarr)

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
                self.sigmatheta = dset.variables['sigmatheta_h'][:]
                self.sigmatheta_u = dset.variables['sigmatheta_u'][:]
                self.sigmatheta_v = dset.variables['sigmatheta_v'][:]
                self.sigmatheta_pv = dset.variables['sigmatheta_pv'][:]
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

            # set up sizes
            self.nt = self.iarr.shape[0]

        self.set_shape_factors()

        self.setup_time()

    def setup_time(self):
        '''
            Get all the timestamps for the outputs. This is slow
            if you have a lot of outputs. If this is slow, you can turn off
            `get_time` in the `setup_extract` function
        '''
        self.tarr = np.zeros(self.nt)
        for i, ti in enumerate(self.iarr):
            fname = self.iarr[i]
            with nc.Dataset(fname, 'r') as dset:
                self.tarr[i] = dset.variables['time'][0]

    def set_shape_factors(self):
        '''
            Set these for calculating Ertel's PV on isobaric surfaces
            Should be called right after setup_extract
        '''
        self.m_h = np.zeros((self.gridnk + 1, self.gridnj + 1))
        self.n_h = np.zeros((self.gridnk + 1, self.gridnj + 1))
        self.m_pv = np.zeros((self.gridnk + 1, self.gridnj + 1))
        self.n_pv = np.zeros((self.gridnk + 1, self.gridnj + 1))

        f_pv = np.zeros((self.gridnk + 1, self.gridnj + 1, self.gridni))
        f_h = np.zeros((self.gridnk + 1, self.gridnj + 1, self.gridni))

        for j in range(self.gridnj + 1):
            lat_pv = np.radians(self.lat_pv[j])
            lat_h = np.radians(self.lat_h[j])

            rln_pv = self.grid_re / \
                np.sqrt(1. +
                        (self.grid_rp / self.grid_re * np.tan(lat_pv))**2.)
            rlt_pv = rln_pv /\
                (np.cos(lat_pv) * (
                    np.sin(lat_pv)**2. + (
                        self.grid_re /
                        self.grid_rp * np.cos(lat_pv))**2.))

            self.m_pv[:, j] = 1. / (rln_pv * self.dln)
            self.n_pv[:, j] = 1. / (rlt_pv * self.dlt)

            rln_h = self.grid_re / \
                np.sqrt(1. + (self.grid_rp / self.grid_re * np.tan(lat_h))**2.)
            rlt_h = rln_h /\
                (np.cos(lat_h) * (
                    np.sin(lat_h)**2. +
                    (self.grid_re / self.grid_rp *
                     np.cos(lat_h))**2.))

            self.m_h[:, j] = 1. / (rln_h * self.dln)
            self.n_h[:, j] = 1. / (rlt_h * self.dlt)

            f_pv[:, j, :] = 2. * self.omega * np.sin(lat_pv)
            f_h[:, j, :] = 2. * self.omega * np.sin(lat_h)

    def get_variable_at_time(self, var, time):
        fname = self.iarr[time]

        with nc.Dataset(fname, 'r') as dset:
            if var == 'ertel_pv':
                return self.get_ertel_pv(time)

            if var not in dset.variables:
                raise KeyError(f'Dataset does not contain {var} at time {time} => {self.tarr[time]}')

            if dset.variables[var].dimensions[0] == 'time':
                variable = dset.variables[var][0, :]
            else:
                variable = dset.variables[:]

        return variable

    def get_variable(self, var, time=None):
        if time is None:
            time = range(len(self.iarr))
        if time is not None and isinstance(time, int):
            return self.get_variable_at_time(var, time)
        elif isinstance(time, Iterable):
            data = []
            for ix in time:
                data.append(self.get_variable_at_time(var, ix))
            return np.asarray(data)
        else:
            raise ValueError(f"time must be None, integer or a list of time values. Got {time}")

    def get_attrs(self, time, attrs=None):
        '''
            Gets a specific attribute (or all attributes from a given extract
        '''
        fname = self.iarr[time]
        with nc.Dataset(fname, 'r') as dset:
            if attrs is None:
                attrs = dset.ncattrs()
            elif isinstance(attrs, str):
                attrs = [attrs]

            return {attr: getattr(dset, attr) for attr in attrs}

    def get_ertel_pv(self, time):
        '''
            Get Ertel's PV in the sigma portion of the zeta coordinate.
            Calculates the (d theta/d zeta) term and multiplies the
            PV from EPIC to get the actual Ertel's PV on isobaric surfaces
        '''
        pv = self.get_variable_at_time('pv', time)
        theta = self.get_variable_at_time('theta', time)
        sigth = self.sigmatheta_pv

        ertel_pv = pv.copy()
        dsgth = (sigth[2:] - sigth[:-2])
        dsgth = dsgth.reshape(sigth.size - 2, 1).repeat(pv.shape[1], 1)
        dsgth = dsgth.reshape(
            sigth.size - 2, pv.shape[1], 1).repeat(pv.shape[2], 2)

        # get the shape factors for all horizontal grid points
        mm1 = 1. / self.m_pv[:, :-1]
        nm1 = 1. / self.n_pv[:, :-1]
        mp1 = 1. / self.m_pv[:, 1:]
        np1 = 1. / self.n_pv[:, 1:]

        # get MN on the edges (where PV is defined)
        mn_pv = 0.5 / (mm1 * nm1 + mp1 * np1)

        theta_av = theta.copy()

        for ii in range(self.gridni):
            # get the value of theta on the edges by interpolating
            theta_av[:, 1:, ii] = ((theta[:, :-1, ii - 1] +
                                    theta[:, :-1, ii]) * (mm1 * nm1)
                                   + (theta[:, 1:, ii] +
                                   theta[:, 1:, ii - 1]) *
                                   (mp1 * np1)) * mn_pv

        dth = (theta_av[2:, :, :] - theta_av[:-2, :, :])

        # calculated d(theta)/d(zeta)
        dthetadsgth = dth[:, 1:, :] / dsgth[:, 1:, :]
        ertel_pv[1:-1, 1:, :] = pv[1:-1, 1:, :] * dthetadsgth

        return ertel_pv
