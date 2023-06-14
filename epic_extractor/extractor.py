import numpy as np
import netCDF4 as ncdf
import os
import re
import glob


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

    def __init__(self, imgfolder=".", plotfolder="plots/"):
        """
        Initializes the extractor
        """
        self.imgfolder = os.path.abspath(imgfolder)
        self.pltfolder = os.path.abspath(plotfolder)
        self.getextractmatch()

        self.checkfolders()

    def checkfolders(self):
        """
        Makes sure the folder exists. Called by default.
        """
        if not os.path.exists(self.pltfolder):
            os.makedirs(self.pltfolder)
            print("Folder was created: ", self.pltfolder)

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

        self.iarr = self.iarr[:start]
        files = sorted(glob.glob(resfolder + "/extract*.nc"))
        for file in files:
            self.iarr.append(file)

        self.iarr = np.asarray(self.iarr)

    def setup_extract(self, auto_vars=True, get_time=True,
                      varlist=['t', 'p', 'pdry', 'rho'],
                      species=[],
                      phase=['liquid', 'rain', 'vapor', 'snow', 'solid']):
        '''
            Initialize the bookkeeping and get basic properties of the
            outputs (e.g., thermo variables, grid sizes, extents etc.)
            Should be called immediately after initialization of the class
        '''

        # Open the first dataset
        fname = self.iarr[0]
        with ncdf.Dataset(fname, 'r') as dset:
            # read in useful variables
            self.Ratmo = dset.planet_rgas
            self.Cp = dset.planet_cp
            self.xh2 = dset.planet_x_h2
            self.xhe = dset.planet_x_he
            self.x3 = dset.planet_x_3
            self.cpr = self.Cp / self.Ratmo
            self.p0 = dset.grid_press0

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
            self.sigmatheta = dset.variables['sigmatheta_h'][:]
            self.sigmatheta_u = dset.variables['sigmatheta_u'][:]
            self.sigmatheta_v = dset.variables['sigmatheta_v'][:]
            self.sigmatheta_pv = dset.variables['sigmatheta_pv'][:]

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
            self.lon_v = dset.variables["lon_u"][:]

            self.lat_v = dset.variables["lat_v"][:]
            self.lon_v = dset.variables["lon_v"][:]

            self.lat_pv = dset.variables["lat_pv"][:]
            self.lon_pv = dset.variables["lon_pv"][:]

            self.gravity = np.array(dset.variables['gravity'])
            self.gave = self.gravity.mean()

            # set up sizes
            self.nt = self.iarr.shape[0]

        # Automatically retrieve variables
        # This is slow if there are a lot of variables.
        # Can be sped up if you only need a few variables for analysis
        # by specifying
        if auto_vars:
            self.varlist, self.species = self.get_variables(fname)
        else:
            self.varlist = varlist
            self.species = species

        if get_time:
            self.setup_time()
        else:
            self.tarr = (-1.) * np.ones(self.nt)

    def get_variables(self, fname):
        # Create the REGEX phrase for matching species names
        matchphrase = "(.*)_(solid|liquid|rain|snow|vapor)([_tendency]*)"

        # Initialize empty varlist and species arrays
        species = []
        varlist = []

        # loop through all the variables and check if
        # it matches the shape
        with ncdf.Dataset(fname, 'r') as dset:
            for var in dset.variables.keys():
                recheck = re.match(matchphrase, var)

                # if it matches the species name, add it to species var
                if recheck:
                    spec = recheck.group(1)
                    if spec not in species:
                        species.append(spec)
                else:
                    # if it's a 4-D variable (time, z, y, x), then add it
                    if len(dset.variables[var].shape) == 4:
                        varlist.append(var)

        return varlist, species

    def setup_time(self):
        '''
            Get all the timestamps for the outputs. This is slow
            if you have a lot of outputs. If this is slow, you can turn off
            `get_time` in the `setup_extract` function
        '''
        self.tarr = np.zeros(self.nt)
        for i, ti in enumerate(self.iarr):
            fname = self.iarr[i]
            with ncdf.Dataset(fname, 'r') as dset:
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

    def get_vars(self, i):
        '''
            Get the variables for a given output i
        '''
        fname = self.iarr[i]
        with ncdf.Dataset(fname, 'r') as dset:
            if self.tarr[i] == -1:
                self.tarr[i] = dset.variables['time'][0]

            vars = {}
            # base EPIC dynamics variables
            for j, var in enumerate(self.varlist):
                vars[var] = np.array(dset.variables[var][0, :, :, :])

            # cloud variables
            for i, spec in enumerate(self.species):
                for phase in ['solid', 'liquid', 'rain', 'vapor', 'snow']:
                    var = spec + "_" + phase
                    vars[var] = np.array(dset.variables[var][0, :, :, :])

                    # moist convection variables
                    try:
                        var = var + "_tendency"
                        vars[var] = np.array(dset.variables[var][:, :, :])
                    except BaseException:
                        pass

                # moist convection variables
                try:
                    var = spec + "_pbase"
                    vars[var] = np.array(dset.variables[var][:, :])
                    var = spec + "_cwf"
                    vars[var] = np.array(dset.variables[var][0, :, :, :])
                    var = spec + "_lambda_mc"
                    vars[var] = np.array(dset.variables[var][0, :, :, :])
                    var = spec + "_mb_mc"
                    vars[var] = np.array(dset.variables[var][0, :, :, :])
                    var = spec + "_dadt"
                    vars[var] = np.array(dset.variables[var][0, :, :, :])
                except BaseException:
                    pass
        return vars

    def get_ertel_pv(self, var):
        '''
            Get Ertel's PV in the sigma portion of the zeta coordinate.
            Calculates the (d theta/d zeta) term and multiplies the
            PV from EPIC to get the actual Ertel's PV on isobaric surfaces
        '''
        pv = var["pv"][:]
        theta = var["theta"][:]
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

        var["ertel_pv"] = ertel_pv

        return var
