import fnmatch
import glob
import os
from collections.abc import Iterable

import netCDF4 as nc
import numpy as np


class Extractor:
    """
    Reads extract.nc files from EPIC simulation
    """

    def __init__(self, imgfolder: str = "."):
        """
        Initializes the extractor and sets up the main parameters
        :param input_folder: folder containing extract.nc files
        """
        self.imgfolder = os.path.abspath(imgfolder)
        self.files = self.getextractmatch(self.imgfolder)
        self.setup_extract()

    def getextractmatch(self, folder: str) -> np.array:
        """
        Loops through the folder and finds extract files. Called by
         default, unles getextractmatch=False

        :param folder: main folder containing the .nc files

        :raises FileNotFoundError: when no files were found
        """

        files = sorted(glob.glob(os.path.join(folder, "extract*.nc")))
        iarr = []
        for file in files:
            iarr.append(file)

        if len(iarr) < 1:
            raise FileNotFoundError(f"No files found in path {folder}!")

        return np.asarray(iarr)

    def add_restarts(self, restart_folder: str, start: int = -1) -> None:
        """
        adds extract files from a restart folder

        :param resfolder: folder containing the restart .nc files
        :param start: the timestep index at which to concatenate the restart files
        """
        if not os.path.exists(restart_folder):
            # soft break out and don't add any new files
            return

        self.files = self.files[:start].tolist()
        try:
            self.files.extend(self.getextractmatch(restart_folder))
        except FileNotFoundError:
            pass

        self.files = np.asarray(self.files)

    def setup_extract(self) -> None:
        '''
        Initialize the bookkeeping and get basic properties of the
        outputs (e.g., thermo variables, grid sizes, extents etc.)
        Should be called immediately after initialization of the class
        '''
        self.setup_time()

        # Open the first dataset
        self.get_coordinates()

        # set the shape factors to calculate Ertel PV
        self.set_shape_factors()

    def get_coordinates(self) -> None:
        """
        Get basic coordinates of the model and cache them for easy access
        """
        self.nt = len(self.files)
        # vertical coordinates
        self.sigmatheta = self.get_variable_at_time("sigmatheta_h", 0)
        self.sigmatheta_u = self.get_variable_at_time("sigmatheta_u", 0)
        self.sigmatheta_v = self.get_variable_at_time("sigmatheta_v", 0)
        self.sigmatheta_pv = self.get_variable_at_time("sigmatheta_pv", 0)

        # Lat/lon grids for different variable types
        self.lat_h = self.get_variable_at_time("lat_h", 0)
        self.lon_h = self.get_variable_at_time("lon_h", 0)

        self.lat_u = self.get_variable_at_time("lat_u", 0)
        self.lon_u = self.get_variable_at_time("lon_u", 0)

        self.lat_v = self.get_variable_at_time("lat_v", 0)
        self.lon_v = self.get_variable_at_time("lon_v", 0)

        self.lat_pv = self.get_variable_at_time("lat_pv", 0)
        self.lon_pv = self.get_variable_at_time("lon_pv", 0)

    def setup_time(self) -> None:
        '''
        Get all the timestamps for the outputs. Also finds the index/file correspondence
        when having restart simulations
        '''
        # set up sizes
        self.tarr = []
        self.tarr_file = []
        self.tarr_index_in_file = []
        for i, ti in enumerate(self.files):
            fname = self.files[i]
            with nc.Dataset(fname, 'r') as dset:
                self.tarr.append(dset.variables['time'][0])
                # the time dimension is only the first index in each file
                # since EPIC 4 staggers one timestep per output file
                self.tarr_file.append(fname)
                self.tarr_index_in_file.append(0)

    def set_shape_factors(self) -> None:
        '''
        Set these for calculating Ertel's PV on isobaric surfaces
        Should be called right after setup_extract
        '''
        # useful for recalculating Ertel's PV
        omega = self.get_attrs(0, "planet_omega_sidereal")["planet_omega_sidereal"]
        grid_re = self.get_attrs(0, "grid_re")["grid_re"]
        grid_rp = self.get_attrs(0, "grid_rp")["grid_rp"]
        dln = np.radians(self.get_attrs(0, "grid_dln")["grid_dln"])
        dlt = np.radians(self.get_attrs(0, "grid_dlt")["grid_dlt"])

        gridnj = self.get_attrs(0, "grid_nj")["grid_nj"]
        gridnk = self.get_attrs(0, "grid_nk")["grid_nk"]
        gridni = self.get_attrs(0, "grid_ni")["grid_ni"]

        self.m_h = np.zeros((gridnk + 1, gridnj + 1))
        self.n_h = np.zeros((gridnk + 1, gridnj + 1))
        self.m_pv = np.zeros((gridnk + 1, gridnj + 1))
        self.n_pv = np.zeros((gridnk + 1, gridnj + 1))

        try:
            self.gravity = self.get_variable_at_time("gravity", 0)
        except KeyError:
            self.gravity = self.get_variable_at_time("gravity2", 0)
        self.gave = self.gravity.mean()

        f_pv = np.zeros((gridnk + 1, gridnj + 1, gridni))
        f_h = np.zeros((gridnk + 1, gridnj + 1, gridni))

        for j in range(gridnj + 1):
            lat_pv = np.radians(self.lat_pv[j])
            lat_h = np.radians(self.lat_h[j])

            rln_pv = grid_re / np.sqrt(
                1.0 + (grid_rp / grid_re * np.tan(lat_pv)) ** 2.0
            )
            rlt_pv = rln_pv / (
                np.cos(lat_pv)
                * (np.sin(lat_pv) ** 2.0 + (grid_re / grid_rp * np.cos(lat_pv)) ** 2.0)
            )

            self.m_pv[:, j] = 1.0 / (rln_pv * dln)
            self.n_pv[:, j] = 1.0 / (rlt_pv * dlt)

            rln_h = grid_re / np.sqrt(1.0 + (grid_rp / grid_re * np.tan(lat_h)) ** 2.0)
            rlt_h = rln_h / (
                np.cos(lat_h)
                * (np.sin(lat_h) ** 2.0 + (grid_re / grid_rp * np.cos(lat_h)) ** 2.0)
            )

            self.m_h[:, j] = 1.0 / (rln_h * dln)
            self.n_h[:, j] = 1.0 / (rlt_h * dlt)

            f_pv[:, j, :] = 2.0 * omega * np.sin(lat_pv)
            f_h[:, j, :] = 2.0 * omega * np.sin(lat_h)

    def get_variable_at_time(self, var: str, time: int) -> np.array:
        """
        Get a given variable for a given index

        :param var: variable name
        :param time: index of the output

        :returns: a numpy array of the variable for requested time

        :raises KeyError: if the dataset does not contain `var`
        """
        fname = self.tarr_file[time]

        with nc.Dataset(fname, 'r') as dset:
            if var == 'ertel_pv':
                return self.get_ertel_pv(time)

            if var not in dset.variables:
                raise KeyError(
                    f'Dataset does not contain {var} at time {time} => {self.tarr[time]}'
                )

            ind = self.tarr_index_in_file[time]

            if dset.variables[var].dimensions[0] == 'time':
                variable = dset.variables[var][ind, :]
            else:
                variable = dset.variables[var][:]

        return variable

    def get_variable(self, var: str, time: list[int] | int | None = None) -> np.array:
        """
        Wrapper function to get a variable for a range of times

        :param var: name of the variable
        :param time: either a list of indices, a single index or None, in which case all the extracts are used

        :returns: a numpy array of the variable for the range of requested times

        :raises KeyError: if the dataset does not contain `var`
        :raises ValueError: if the input time format is not correct
        """
        if time is None:
            time = range(len(self.files))
        if time is not None and isinstance(time, int):
            return self.get_variable_at_time(var, time)
        elif isinstance(time, Iterable):
            data = []
            for ix in time:
                data.append(self.get_variable_at_time(var, ix))
            return np.asarray(data)
        else:
            raise ValueError(
                f"time must be None, integer or a list of time values. Got {time}"
            )

    def get_attrs(self, time, attrs=None):
        '''
        Gets a specific attribute (or all attributes from a given extract
        '''
        fname = self.tarr_file[time]
        with nc.Dataset(fname, 'r') as dset:
            all_attrs = dset.ncattrs()
            if attrs is None:
                output_attrs = all_attrs
            elif isinstance(attrs, str):
                output_attrs = [
                    attr for attr in all_attrs if fnmatch.fnmatch(attr, attrs)
                ]
            elif isinstance(attrs, Iterable):
                # if it is an iterable, loop through all
                # and do the regex match to see which attributes are
                # requested
                output_attrs = []
                for in_attr in attrs:
                    output_attrs.extend(
                        attr for attr in all_attrs if fnmatch.fnmatch(attr, in_attr)
                    )
                output_attrs = list(set(output_attrs))

            return {attr: getattr(dset, attr) for attr in output_attrs}

    def get_ertel_pv(self, time: int) -> np.array:
        '''
        Get Ertel's PV in the sigma portion of the zeta coordinate.
        Calculates the (d theta/d zeta) term and multiplies the
        PV from EPIC to get the actual Ertel's PV on isobaric surfaces

        :param time: the output index to extract the PV

        :returns: the Ertel's PV at `time`
        '''
        pv = self.get_variable_at_time('pv', time)
        theta = self.get_variable_at_time('theta', time)
        sigth = self.sigmatheta_pv

        ertel_pv = pv.copy()
        dsgth = sigth[2:] - sigth[:-2]
        dsgth = dsgth.reshape(sigth.size - 2, 1).repeat(pv.shape[1], 1)
        dsgth = dsgth.reshape(sigth.size - 2, pv.shape[1], 1).repeat(pv.shape[2], 2)

        # get the shape factors for all horizontal grid points
        mm1 = 1.0 / self.m_pv[:, :-1]
        nm1 = 1.0 / self.n_pv[:, :-1]
        mp1 = 1.0 / self.m_pv[:, 1:]
        np1 = 1.0 / self.n_pv[:, 1:]

        # get MN on the edges (where PV is defined)
        mn_pv = 0.5 / (mm1 * nm1 + mp1 * np1)

        theta_av = theta.copy()

        for ii in range(self.gridni):
            # get the value of theta on the edges by interpolating
            theta_av[:, 1:, ii] = (
                (theta[:, :-1, ii - 1] + theta[:, :-1, ii]) * (mm1 * nm1)
                + (theta[:, 1:, ii] + theta[:, 1:, ii - 1]) * (mp1 * np1)
            ) * mn_pv

        dth = theta_av[2:, :, :] - theta_av[:-2, :, :]

        # calculated d(theta)/d(zeta)
        dthetadsgth = dth[:, 1:, :] / dsgth[:, 1:, :]
        ertel_pv[1:-1, 1:, :] = pv[1:-1, 1:, :] * dthetadsgth

        return ertel_pv
