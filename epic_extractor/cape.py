import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fixed_point
from scipy.integrate import odeint
from dataclasses import dataclass, field
from .extractor import Extractor
from .thermo import Planet
import multiprocessing
import signal
import tqdm

Cpw = 4218.


@dataclass
class Constants:
    Ratmo: float
    Rw: float
    g: float
    Cp: float
    Lv: float
    epsilon: float = field(init=False)
    kappa: float = field(init=False)

    def __post_init__(self):
        self.epsilon = self.Ratmo / self.Rw
        self.kappa = self.Ratmo / self.Cp


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class EPIC_CAPE:
    def __init__(self, extractor: Extractor):
        self.extractor = extractor
        self.planet = Planet.from_extract(extractor)
        self.CAPE = np.zeros((self.extractor.tarr.size, self.extractor.lat_h.size, self.extractor.lon_h.size))
        self.CIN = np.zeros((self.extractor.tarr.size, self.extractor.lat_h.size, self.extractor.lon_h.size))
        self.pLCL = np.zeros((self.extractor.tarr.size, self.extractor.lat_h.size, self.extractor.lon_h.size))
        self.pLFC = np.zeros((self.extractor.tarr.size, self.extractor.lat_h.size, self.extractor.lon_h.size))
        self.pEL = np.zeros((self.extractor.tarr.size, self.extractor.lat_h.size, self.extractor.lon_h.size))

    def get_CAPE_CIN(self, num_procs=1, pbase=6000.e2, species='H_2O'):
        if species == 'H_2O':
            Rw = 8314. / 18.
            Lv = 2.501e6 + 333.55e3
        else:
            raise ValueError(f"{species} not implemented")

        for n, t in enumerate(self.extractor.tarr):
            var = self.extractor.get_vars(n)
            pfull = var["p"]
            T = var["t"]
            vapor = var[f"{species}_vapor"]

            inpargs = []
            for j, latj in enumerate(self.extractor.lat_h):
                for i, loni in enumerate(self.extractor.lon_h):
                    pij = pfull[:, j, i][::-1]
                    Tij = T[:, j, i][::-1]
                    qij = vapor[:, j, i][::-1]
                    k0 = np.argmin((pij - pbase)**2.)
                    cp_k0 = self.planet.return_cp(pij[k0], Tij[k0])
                    inpargs.append([pij, Tij, qij, k0, cp_k0, self.extractor.Ratmo, self.extractor.gave, Rw, Lv])

            if num_procs == 1:
                for jj, args in enumerate(tqdm.tqdm(inpargs, desc='Calculating CAPE')):
                    Tparcel, Tvparcel, CAPEi, CINi, bi, plcli, plfci, peli = do_CAPE(*args)
                    j, i = np.unravel_index(jj, (self.extractor.lat_h.size, self.extractor.lon_h.size))
                    self.CAPE[n, j, i] = CAPEi
                    self.CIN[n, j, i] = CINi
                    self.pLCL[n, j, i] = plcli
                    self.pLFC[n, j, i] = plfci
                    self.pEL[n, j, i] = peli
            else:
                with multiprocessing.Pool(processes=num_procs, initializer=initializer) as pool:
                    try:
                        for jj, result in enumerate(pool.starmap(do_CAPE, tqdm.tqdm(inpargs, desc='Calculating CAPE'))):
                            j, i = np.unravel_index(jj, (self.extractor.lat_h.size, self.extractor.lon_h.size))
                            self.CAPE[n, j, i] = result[2]
                            self.CIN[n, j, i] = result[3]
                            self.pLCL[n, j, i] = result[5]
                            self.pLFC[n, j, i] = result[6]
                            self.pEL[n, j, i] = result[7]

                        pool.close()
                    except KeyboardInterrupt:
                        pool.terminate()
                        pool.join()
                        return
                    except Exception:
                        raise

                    pool.join()


def esat(T):
    if isinstance(T, float):
        es = 610.78 * np.exp(17.26939 * (T - 273.16) / (T - 35.86)) if T > 35.86 else 0
    else:
        es = np.zeros_like(T)
        es[T > 35.86] = 610.78 * np.exp(17.26939 * (T[T > 35.86] - 273.16) / (T[T > 35.86] - 35.86))

    return es


def dewpoint(p, q, epsilon):
    T0 = 273.16
    T1 = 38.86
    e0 = 610.78
    e = q * p / (epsilon + q)

    if (e > 0.):
        a = T0 - T1 * (np.log(e / e0) / 17.26939)
        b = 1. - (np.log(e / e0) / 17.26939)
        dp = a / b
    else:
        dp = 0.

    return dp


def get_lcl(p, T, qH2O, k0, constants):
    p0 = p[k0]
    t0 = T[k0]

    q0 = qH2O[k0]
    dewpt0 = dewpoint(p0, q0, constants.epsilon)

    esdew0 = esat(dewpt0)
    q0 = esdew0 / (p0 - esdew0) * constants.epsilon

    # def dewpoint(p, q):
    #     e = q*p/(epsilon + q)
    #     Td = 2681.18/(12.610 - np.log10(e))
    #     return Td

    def lcl_min(p):
        Td = dewpoint(p, q0, constants.epsilon)
        # print(p, Td)
        return p0 * (Td / t0)**(constants.Cp / constants.Ratmo)

    # interpolate the Tp profile so we can call
    # it as a function of pressure
    Tp = interp1d(np.log10(p), T, kind='cubic')
    # zp = interp1d(np.log10(p), z, kind='cubic')

    # this is the minimizer where we are trying to solve
    # for the location where q0 = qsat(p)
    # where qsat is defined as the saturation mixing ratio

    plcl = fixed_point(lcl_min, p0, maxiter=50)
    # zlcl = zp(np.log10(plcl))
    tlcl = Tp(np.log10(plcl))
    # get the index of k corresponding to the LCL
    klcl = np.argmin((p - plcl)**2.)

    return (plcl, tlcl, klcl)


def get_parcel_temp(p, T, k0, q0, plcl, klcl, constants):
    p0 = p[k0]
    t0 = T[k0]

    # integrate the parcel temperature
    # with the dry lapse rate

    # integrate upto the point before
    Tparcel = (T[k0]) * np.ones_like(p)
    Xparcel = np.zeros_like(p)
    for k in range(0, klcl + 1):
        # dry lapse rate
        Tparcel[k] = t0 * (p[k] / p0)**(constants.Ratmo / (constants.Cp))
        Xparcel[k] = q0 / (constants.epsilon + q0)

    # we then correct the remaining amount based on
    # how much more of the atmosphere is dry
    Tparcel[klcl + 1] = t0 * (plcl / p0)**(constants.Ratmo / constants.Cp)

    # calculate the moist adiabat wrt pressure
    def dTdp_moist(Ti, p):
        esi = esat(Ti)
        qi = esi / (p - esi) * constants.epsilon
        # This equation comes from [Bakhshaii2013]_.
        dTdp = (1. / p) * ((constants.Ratmo * Ti + constants.Lv * qi) /
                           ((constants.Cp + qi * Cpw) / (1. + qi) +
                           (constants.Lv * constants.Lv * qi * constants.epsilon /
                            (constants.Ratmo * Ti * Ti))))
        return dTdp

    # integrate it for all points from klcl+1 to the end
    pmoist = p[klcl:]

    T0 = Tparcel[klcl]

    Tmoist = odeint(dTdp_moist, T0, pmoist)

    # set everything above the klcl to be the newly
    # integrated temperature
    Tparcel[klcl:] = Tmoist[:, 0]

    # get the virtual temperature of the parcel
    emoist = esat(Tmoist[:, 0])
    Xparcel[klcl:] = emoist / (pmoist)
    Tvparcel = Tparcel / (1. - Xparcel * (1. - constants.epsilon))

    return (Tparcel, Tvparcel)


def get_CAPE_CIN(p, k0, plcl, T, Tvparcel, q, constants):

    # get the virtual temperature of the atmosphere
    Xatmo = q / (q + constants.epsilon)
    Tv = T / (1. - Xatmo * (1. - constants.epsilon))

    # find the buoyancy:
    # b = Rdry*(Tvatmo - Tvparcel)
    # CAPE = integral b
    b = constants.Ratmo * (Tvparcel - Tv)
    blogp = interp1d(np.log(p), b, kind='cubic')

    # we need to find the kLFC and kEL
    # from MetPY:
    # The LFC could:
    # 1) Not exist
    # 2) Exist but be equal to the LCL
    # 3) Exist and be above the LCL

    # let's loop up from klcl to the top
    # and find the LFC
    # use a high resolution b to find the root
    ptemp = np.linspace(np.log(plcl), np.log(plcl / 50.), 75)
    btemp = blogp(ptemp)

    plfc = -10
    pel = -10
    b_crit = 15.  # critical buoyancy to avoid finding local zeros
    for xi, lpi in enumerate(ptemp):
        if (xi > 74):
            break
        # get the EL
        if (btemp[xi] < -b_crit):
            if ((plfc != -10) & (pel == -10)):
                pel = np.exp(lpi)
                break
        # get the LFC
        if (btemp[xi] > b_crit):
            if (plfc == -10):
                plfc = np.exp(lpi)

    # integrate b to get the CAPE
    if (plfc == -10):
        # if there is no LFC, CAPE=0
        CAPE = 0.
        CIN = 0.
    else:
        # integrate from zlfc to zel
        # to get the CAPE
        pCAPE = np.linspace(np.log(plfc), np.log(pel), 50)
        bCAPE = blogp(pCAPE)
        bCAPE[bCAPE < 0.] = 0.
        CAPE = -np.trapz(bCAPE, pCAPE)

        pCIN = np.linspace(np.log(p[k0]), np.log(plfc), 50)
        bCIN = blogp(pCIN)
        bCIN[bCIN > 0.] = 0.
        CIN = - np.trapz(bCIN, pCIN)

    return CAPE, CIN, b, plfc, pel


def do_CAPE(p, T, qH2O, k0, Cp, Ratmo, g, Rw, Lv):
    constants = Constants(Ratmo, g, Cp, Rw, Lv)
    plcl, tlcl, klcl = get_lcl(p, T, qH2O, k0, constants)
    if np.isfinite(plcl) and (plcl > 100.e2):
        Tparcel, Tvparcel = get_parcel_temp(p, T, k0, qH2O[k0], plcl, klcl, constants)
        CAPE, CIN, b, plfc, pel = get_CAPE_CIN(p, k0, plcl, T, Tvparcel, qH2O, constants)
    else:
        Tparcel = Tvparcel = b = np.zeros_like(p)
        CAPE = CIN = 0
        plfc = pel = -10

    return Tparcel, Tvparcel, CAPE, CIN, b, plcl, plfc, pel
