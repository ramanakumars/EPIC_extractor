import numpy as np
from ..thermo import Planet
from ..utils import get_brunt2
from .planet_global_properties import PLANETS
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
import argparse


mu_water = 18.

aw = 12.610
bw = -2681.18
SOLAR_WATER_VMR = 1.433e-3


def get_planet_from_name(planet_name: str) -> Planet:
    planet_properties = PLANETS[planet_name]
    return Planet(**planet_properties, p0=1000e2)


def sat_vapor_pressure_water(T):
    return np.power(10, aw + bw / T)


def get_water_vapor_profile(p, T, deep_value, planet):
    vmr = np.zeros_like(p)

    vmr[0] = deep_value * SOLAR_WATER_VMR

    mu_dry = 8314. / planet.rgas

    vmr_sat = sat_vapor_pressure_water(T[::-1]) / p[::-1]
    for i, (pi, Ti) in enumerate(zip(p, T)):
        if i < 1:
            continue
        vmr[i] = min([vmr_sat[i], vmr[i - 1]])

    q = 1 / (mu_dry * (1 - vmr)) * mu_water * vmr

    return q[::-1]


def get_q(T, p, deep_value, planet: Planet):
    vmr = np.min([sat_vapor_pressure_water(T) / p, deep_value * SOLAR_WATER_VMR], axis=0)
    mu_dry = 8314. / planet.rgas
    return (1 / (mu_dry * (1 - vmr))) * mu_water * vmr


def get_R(T: np.ndarray, logp: np.ndarray, deep_value: float, planet: Planet) -> np.ndarray:
    """
    get the specific gas constant for a given (p, T) assuming a deep abundance of water
    this assumes that the water is at most saturated at this location

    :param T: the input temperature [K]
    :param logp: the input log pressure value [Pa]
    :param deep_value: the deep water abundance [solar]
    :param planet: the object holding the planet specific properties

    :return: the specific gas constant [J/kg K]
    """
    qH2O = get_q(T, np.exp(logp), deep_value, planet)
    mu_dry = 8314 / planet.rgas
    mu = (1 + qH2O) / (1. / mu_dry + qH2O / mu_water)
    return 8314. / mu


def dTdlogp(logp: np.ndarray, T: np.ndarray, N2: callable, planet: Planet, deep_value: float) -> np.ndarray:
    """
    Integration function. This returns the value of dT/d(log p) = f(p, T, N^2 ...)

    The full function is from Dowling et al. 2006 (Appendix A), where we expand
    d(rho)/dp env into dT/d(log p) and ignore the dR/dT term in this expansion:
        d(rho)/dp = d(p / RT)/dp = 1/(RT) - 1/(R^2 T) dR/d(log p) - 1 / (R T^2) dT / d(log p)

    :param logp: the natural log of the pressure [Pa]
    :param T: the temperature at this location [K]

    :return: dT/d(log p) at this location [K]
    """
    p = np.exp(logp)

    cp = np.asarray([planet.return_cp(p, ti) for ti in T])

    dlnp = 0.1

    R = get_R(T, logp, deep_value, planet)

    x1 = 1. / (cp * T)
    x2 = N2(p) / (planet.g**2.)
    x3 = (get_R(T, logp + dlnp / 2, deep_value, planet) - get_R(T, logp - dlnp / 2, deep_value, planet)) / (dlnp) / (R**2. * T)
    denom = 1 / (R * (T**2.))

    return (x1 - x2 - x3) / denom


def get_new_Tp(p: np.ndarray, T: np.ndarray, planet: Planet, Nsq_deep: float, p_knee: float, p_interp: float, p_end: float, deep_water: float) -> tuple[np.ndarray]:
    """
    Integrate the N^2 equation to calculate T as a function of p

    :param p: input reference pressure [mbar]
    :param T: input reference temperature [K]
    :param planet: planet object that contains atmospheric properties
    :param Nsq_deep: deep value of N^2 to use for integration
    :param p_knee: the pressure value of the start (top) of the smoothing region where the reference N^2 is smoothed to the deep N^2 [mbar]
    :param p_interp: the pressure value of the start of the integration region [mbar]
    :param p_end: the pressure value of the end (bottom) of the integration region [mbar]
    :param deep_water: the deep water abundance [solar]

    :returns:   - the new pressure layers [mbar]
                - the new temperature at the corresponding pressure layers [K]
    """
    g = planet.g

    def fun(logp, T):
        return dTdlogp(logp, T, Nsq_smooth, planet, deep_water)

    # get the initial N^2 from the T-p profile and assumed water abundance
    # we will use this to smoothen the transition region
    mu_dry = 8314 / planet.rgas
    q = get_water_vapor_profile(p, T, deep_water, planet)
    mu = (1 + q) / (1 / mu_dry + q / mu_water)  # / 2.26
    Nsq_initial = get_brunt2(planet, p, T, mu, g)

    # get the knee (top of the smoothing region) and the interpolation start (bottom of the smoothing region)
    ind0 = ind_knee = np.argmin((p - p_knee)**2.)
    ind_interp = np.argmin((p - p_interp)**2.)
    T0 = T[ind_knee]
    pstart = p[ind_knee] * 100  # 680mb

    # we'll use an spline interpolator to smoothly transition between the top N^2 and the deep constant N^2
    x = np.log([p[ind_knee - 1], p[ind_knee], p[ind_interp], p[ind_interp + 1]])
    y = [Nsq_initial[ind_knee - 1], Nsq_initial[ind_knee], Nsq_deep, Nsq_deep]
    Nsq_spline = PchipInterpolator(x + np.log(100), y)

    # blanket function to just return the spline in the transition region or the original/deep values elsewhere as needed
    Nsq_smooth = np.vectorize(lambda pi: Nsq_spline(np.log(pi)) if (pi < (p_interp * 100)) & (pi > (p_knee * 100)) else (Nsq_deep if pi > (p_interp * 100) else Nsq_initial[np.argmin((pi - p * 100)**2.)]))

    # the pressure values where the intepolated T-p will be evaluated
    peval = np.logspace(np.log10(pstart) + 0.01, np.log10(p_end) - 0.01, 101)

    # solve the IVP where dT/dlog(p) = f(p, T, N^2_deep) with T=T_0 at p=pstart
    sol = solve_ivp(fun, [np.log(pstart), np.log(p_end)], [T0], t_eval=np.log(peval), method='LSODA')
    print(sol.success, sol.message)

    # then take the solution and stitch it back together to get the new T-p
    Tnew = np.zeros(len(T[:ind0]) + len(peval))
    pnew = np.zeros(len(T[:ind0]) + len(peval))

    # use the reference T-p above pstart
    Tnew[:ind0] = T[:ind0]
    pnew[:ind0] = p[:ind0]

    # and the new values below
    pnew[ind0:] = np.exp(sol.t) / 100.
    Tnew[ind0:] = sol.y.flatten()

    # in the transition, smoothen the T-p to avoid strange jumps in N^2
    Tnew[ind0 - 1] = (T[ind0] * np.sqrt(pnew[ind0]) + T[ind0 - 2] * np.sqrt(pnew[ind0 - 2])) / (np.sqrt(pnew[ind0]) + np.sqrt(pnew[ind0 - 2]))

    return pnew, Tnew


def extrapolate_Tp():
    parser = argparse.ArgumentParser(description="Interpolate a T-p profile for a planet based on a deep constant N^2")

    parser.add_argument("input", help="Input reference T-p profile (EPIC formatted)", type=str)
    parser.add_argument("output", help="Output file to store new T-p profile (EPIC formatted)", type=str)
    parser.add_argument("--planet", help="Name of the planet", type=str, choices=list(PLANETS.keys()))
    parser.add_argument("--water_abundance", help="Deep water abundance (in solar units)", type=float)
    parser.add_argument("--N2_deep", help="Deep value of the N^2 to use for extrapolation", type=float)
    parser.add_argument("--p_knee", help="Pressure level from the reference T-p profile to use as smoothing for N^2 (mbar)", type=float, default=200)
    parser.add_argument("--p_interp", help="Pressure level to start the interpolation (mbar)", default=1000, type=float)
    parser.add_argument("--p_end", help="Pressure value for the end of the extrapolation (mbar)", default=60000, type=float)
    args = parser.parse_args()

    input_Tp = args.input
    outfile = args.output
    planet_name = args.planet
    water_abundance = args.water_abundance
    Nsq_deep = args.N2_deep
    p_knee = args.p_knee
    p_interp = args.p_interp
    p_end = args.p_end

    t_vs_p = np.loadtxt(input_Tp, skiprows=7)

    p = t_vs_p[:, 0]
    T = t_vs_p[:, 1]

    planet = get_planet_from_name(planet_name)

    pnew, Tnew = get_new_Tp(p, T, planet, Nsq_deep, p_knee, p_interp, p_end * 100, water_abundance)
    with open(outfile, 'w') as out:
        out.write(f"Temperature versus pressure for {planet_name}, extrapolated from {input_Tp}\n\n\n\n")
        out.write(f"Extended past {int(p_interp):d} mb using Nsq = {Nsq_deep:1.0e} and water abundance = {int(water_abundance):d}x solar\n")
        out.write("#     p[hPa]     T[K]       dT[K]\n")
        out.write(f"{len(Tnew)}\n")

        for (Ti, pi) in zip(Tnew, pnew):
            out.write(f"     {pi:.3e} {Ti:.3e}    0.\n")
