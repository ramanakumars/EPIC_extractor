# 1-D lightning simulation
# EPIC model, including extractor


import os
import sys

# For plotting
import matplotlib.pyplot as plt
import netCDF4

# For scientific computing
import numpy as np
from numba import njit

from ...extractor import Extractor


# Particle charging, for multiple particle species
@njit
def dQdt6(Ns, binbounds, velocities, mus, Qcoeffs):
    # velocities is particle fall velocities, rhosadj removed
    r0s = np.zeros(len(binbounds) - 1)
    dQidt = np.zeros((len(mus), len(binbounds) - 1))
    # calculate r0 as bin center radius
    for s in range(len(binbounds) - 1):
        r0s[s] = (binbounds[s + 1] + binbounds[s]) / 2.0
    # charge transfer
    for au in range(len(mus)):
        for bu in range(len(binbounds) - 1):
            dQitot = 0.0
            for cu in range(len(binbounds) - 1):
                for du in range(len(mus)):
                    rG = min(r0s[bu], r0s[cu])
                    # transferred charge dependent on molar mass
                    # if unknown species, don't transfer any charge
                    # subject to change later
                    if mus[au] > 0.0175 and mus[au] < 0.0185:
                        if mus[du] > 0.0175 and mus[du] < 0.0185:
                            # Water-water charging
                            if rG <= 0.000111:
                                Gr = 0.0000271 * ((1000000.0 * rG) ** 2.7)
                            else:
                                Gr = 0.0988 * ((1000000.0 * rG) ** 0.98)
                            velocity = abs(velocities[au, bu] - velocities[du, cu])
                            delQ = ((velocity / 3.0) ** 2.5) * Gr * (10**-15)
                            dQi = (
                                delQ
                                * Ns[du, cu]
                                * np.pi
                                * (r0s[bu] ** 2 + r0s[cu] ** 2)
                                * min(Qcoeffs[au], Qcoeffs[du])
                            )
                            if bu < cu:
                                dQitot = dQitot + dQi
                            else:
                                dQitot = dQitot - dQi
                        elif mus[du] > 0.0165 and mus[du] < 0.0175:
                            # Ammonia-water charging
                            # Triboelectric: water gets positive charge
                            # Per Lee et al. 2018 and a lot of dubious extrapolation
                            # Only goes as v^1.5 based on Lesprit et al. 2020
                            if rG <= 0.000111:
                                Gr = 0.0000271 * ((1000000.0 * rG) ** 2.7)
                            else:
                                Gr = 0.0988 * ((1000000.0 * rG) ** 0.98)
                            velocity = abs(velocities[au, bu] - velocities[du, cu])
                            delQ = (
                                ((velocity / 3.0) ** 2.5)
                                * (20.68 / velocity)
                                * Gr
                                * (10**-15)
                            )  # or 0.482/velocity if pessimistic
                            dQi = (
                                delQ
                                * Ns[du, cu]
                                * np.pi
                                * (r0s[bu] ** 2 + r0s[cu] ** 2)
                                * min(Qcoeffs[au], Qcoeffs[du])
                            )
                            dQitot = dQitot + abs(dQi)
                    elif mus[au] > 0.0165 and mus[au] < 0.0175:
                        if mus[du] > 0.0165 and mus[du] < 0.0175:
                            # Ammonia-ammonia charging
                            # Use same equation as for water, multiplied by smaller constant (ratio of dielectric constants)
                            if rG <= 0.000111:
                                Gr = 0.0000271 * ((1000000.0 * rG) ** 2.7)
                            else:
                                Gr = 0.0988 * ((1000000.0 * rG) ** 0.98)
                            velocity = abs(velocities[au, bu] - velocities[du, cu])
                            delQ = (
                                (25 / 80.0) * ((velocity / 3.0) ** 2.5) * Gr * (10**-15)
                            )
                            dQi = (
                                delQ
                                * Ns[du, cu]
                                * np.pi
                                * (r0s[bu] ** 2 + r0s[cu] ** 2)
                                * min(Qcoeffs[au], Qcoeffs[du])
                            )
                            if bu < cu:
                                dQitot = dQitot + dQi
                            else:
                                dQitot = dQitot - dQi
                        elif mus[du] > 0.0175 and mus[du] < 0.0185:
                            # Water-ammonia charging
                            # Triboelectric: ammonia gets negative charge
                            # Per Lee et al. 2018 and a lot of dubious extrapolation
                            # Only goes as v^1.5 based on Lesprit et al. 2020
                            if rG <= 0.000111:
                                Gr = 0.0000271 * ((1000000.0 * rG) ** 2.7)
                            else:
                                Gr = 0.0988 * ((1000000.0 * rG) ** 0.98)
                            velocity = abs(velocities[au, bu] - velocities[du, cu])
                            delQ = (
                                ((velocity / 3.0) ** 2.5)
                                * (20.68 / velocity)
                                * Gr
                                * (10**-15)
                            )  # or 0.482/velocity if pessimistic
                            dQi = (
                                delQ
                                * Ns[du, cu]
                                * np.pi
                                * (r0s[bu] ** 2 + r0s[cu] ** 2)
                                * min(Qcoeffs[au], Qcoeffs[du])
                            )
                            dQitot = dQitot - abs(dQi)
                    print(mus[au], mus[du], bu, cu, rG, dQitot, dQi, Ns[du, cu])
            dQidt[au, bu] = dQitot + 0
    return dQidt

@njit
def dEdt6(Ns, binbounds, velocities, mus, Qcoeffs, Ebreakdown):
    # as above, with Ebreakdown as breakdown electric field, for the average
    # also returns 1/(time to lightning in seconds)
    Q1s = dQdt6(Ns, binbounds, velocities, mus, Qcoeffs)
    Jc = 0.0
    for g in range(len(binbounds) - 1):
        for h in range(len(mus)):
            Jcg = -Ns[h, g] * velocities[h, g] * Q1s[h, g]
            Jc = Jc + Jcg
    AvdE = -np.sign(Jc) * np.sqrt(0.5 * abs(Jc / (8.854 * (10**-12))) * Ebreakdown)
    invtc = abs(AvdE / Ebreakdown)
    return AvdE, invtc, Q1s


def dEtotal(
    qH2Oicecloud,
    qH2Oliquidcloud,
    qH2Osnow,
    qH2Orain,
    qNH3icecloud,
    qNH3liquidcloud,
    qNH3snow,
    qNH3rain,
    P,
    T,
    mu_dry=0.0022,
):
    # Mass fractions, P in pascals, T in Kelvin
    binbounds = np.geomspace(0.00001, 10.48576, 41)
    Ebreakdown = 5.0 * P  # Rough approximation of the electric field for hydrogen
    r0s = np.zeros(len(binbounds) - 1)
    # calculate r0 as bin center radius
    for s in range(len(binbounds) - 1):
        r0s[s] = (binbounds[s + 1] + binbounds[s]) / 2.0
    # Species: H2O solid, H2O liquid, H2O snow, NH3 solid, NH3 liquid, NH3 snow
    ## mu in kg/mol
    mus = np.array([0.018015, 0.018015, 0.018015, 0.017031, 0.017031, 0.017031])
    Qcoeffs = np.array([1.0, 0.2, 0.5, 1.0, 0.2, 0.5])
    Ns = np.zeros([6, len(binbounds) - 1])
    velocities = np.zeros([6, len(binbounds) - 1])
    # Dry gas density
    rho_dry = P * mu_dry / (8.314 * T)
    # Ns, water and ammonia ice

    # m_ice = alpha * D^beta
    # N_ice = c * (rho q) ^ d

    # m_ice = (rho q) / N_ice =  1 / c * (rho q) ^ (1 - d)
    # = (rho * q) ** (- d) / c
    D_H2Oice = np.sqrt(((rho_dry * qH2Oicecloud) ** 0.25) / (5.38 * 70616.5))
    H2Oicebin = 0
    for bbcheck in range(len(binbounds) - 1):
        if D_H2Oice >= (2 * binbounds[bbcheck + 1]):
            H2Oicebin = H2Oicebin + 1
    if qH2Oicecloud > 0:
        Ns[0, H2Oicebin] = 53800000.0 * ((rho_dry * qH2Oicecloud) ** 0.75)
    D_NH3ice = np.sqrt(
        ((rho_dry * qNH3icecloud) ** 0.25) / (5.38 * 70616.5 * (786.8 / 917.0))
    )
    NH3icebin = 0
    for bbcheck in range(len(binbounds) - 1):
        if D_NH3ice >= (2 * binbounds[bbcheck + 1]):
            NH3icebin = NH3icebin + 1
    if qNH3icecloud > 0:
        Ns[3, NH3icebin] = 53800000.0 * ((rho_dry * qNH3icecloud) ** 0.75)
    # For liquid cloud particles, EPIC doesn't assume any particular size distribution
    # That said, here we'll assume the cloud particles are, as for ice, all the same size with the mass matching the mass for ice
    # TODO: is this correct?
    D_H2Oliquid = (np.cbrt(917.0 / 1000.0)) * np.sqrt(
        ((rho_dry * qH2Oliquidcloud) ** 0.25) / (5.38 * 70616.5)
    )
    H2Oliquidbin = 0
    for bbcheck in range(len(binbounds) - 1):
        if D_H2Oliquid >= (2 * binbounds[bbcheck + 1]):
            H2Oliquidbin = H2Oliquidbin + 1
    if qH2Oliquidcloud > 0:
        Ns[1, H2Oliquidbin] = 0# 53800000.0 * ((rho_dry * qH2Oliquidcloud) ** 0.75)
    D_NH3liquid = (np.cbrt(786.8 / 733.0)) * np.sqrt(
        ((rho_dry * qNH3liquidcloud) ** 0.25) / (5.38 * 70616.5 * (786.8 / 917.0))
    )
    NH3liquidbin = 0
    for bbcheck in range(len(binbounds) - 1):
        if D_NH3liquid >= (2 * binbounds[bbcheck + 1]):
            NH3liquidbin = NH3liquidbin + 1
    if qNH3liquidcloud > 0:
        Ns[4, NH3liquidbin] = 0.#53800000.0 * ((rho_dry * qNH3liquidcloud) ** 0.75)
    # Okay, now precipitation Ns
    if qH2Orain > 0:
        lambda_H2Orain = (8000000.0 * np.pi * 1000.0 / (rho_dry * qH2Orain)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[1, s] = Ns[1, s] + (8000000.0 / lambda_H2Orain) * (
                np.exp(-lambda_H2Orain * binbounds[s])
                - np.exp(-lambda_H2Orain * binbounds[s + 1])
            )
    if qNH3rain > 0:
        lambda_NH3rain = (8000000.0 * np.pi * 1000.0 / (rho_dry * qNH3rain)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[4, s] = Ns[4, s] + (8000000.0 / lambda_NH3rain) * (
                np.exp(-lambda_NH3rain * binbounds[s])
                - np.exp(-lambda_NH3rain * binbounds[s + 1])
            )
    if qH2Osnow > 0:
        N0snowH2O = 200000000.0 * min(1.0, 0.01 * np.exp(-0.12 * (T - 273.16)))
        lambda_H2Osnow = (N0snowH2O * np.pi * 1000.0 / (rho_dry * qH2Osnow)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[2, s] = (N0snowH2O / lambda_H2Osnow) * (
                np.exp(-lambda_H2Osnow * binbounds[s])
                - np.exp(-lambda_H2Osnow * binbounds[s + 1])
            )
    if qNH3snow > 0:
        N0snowNH3 = 200000000.0 * min(1.0, 0.01 * np.exp(-0.12 * (T - 195.40)))
        lambda_NH3snow = (N0snowNH3 * np.pi * 1000.0 / (rho_dry * qNH3snow)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[5, s] = (N0snowNH3 / lambda_NH3snow) * (
                np.exp(-lambda_NH3snow * binbounds[s])
                - np.exp(-lambda_NH3snow * binbounds[s + 1])
            )
    # for s in [0,10,20,30]:
    # Test: print
    # print('Ns',Ns[0,s],Ns[1,s],Ns[2,s],Ns[3,s],Ns[4,s],Ns[5,s])
    # Velocities
    for s in range(len(binbounds) - 1):
        # Liquid
        velocities[1, s] = 2376.36 * ((2 * r0s[s]) ** 0.7308) * ((100000.0 / P) ** 0.3568)
        velocities[4, s] = 2479.0 * ((2 * r0s[s]) ** 0.76217) * ((100000.0 / P) ** 0.33)
        # Snow
        velocities[2, s] = 42.83 * ((2 * r0s[s]) ** 0.5271) * ((100000.0 / P) ** 0.3043)
        velocities[5, s] = 41.74 * ((2 * r0s[s]) ** 0.5386) * ((100000.0 / P) ** 0.3008)
        # Ice
        velocities[0, s] = 535.41 * ((2 * r0s[s]) ** 0.8746) * ((100000.0 / P) ** 0.1264)
        velocities[3, s] = 495.22 * ((2 * r0s[s]) ** 0.8807) * ((100000.0 / P) ** 0.1248)
        # velocities[0,s] = 2615.1*((r0s[s])**0.74245)*((100000./P)**0.33)*np.sqrt(917/1000.)
        # velocities[3,s] = 2479.0*((r0s[s])**0.76217)*((100000./P)**0.33)*np.sqrt(786.8/733.)
    # for s in [0,10,20,30]:
    # Test: print
    # print('velocities',velocities[0,s],velocities[1,s],velocities[2,s],velocities[3,s],velocities[4,s],velocities[5,s])
    # Now, calculate dE/dt
    dEdt, invtc, dQidt = dEdt6(Ns, binbounds, velocities, mus, Qcoeffs, Ebreakdown)
    return dEdt, invtc, dQidt


def SVPs(P, T, mus_cond):
    SVPs = np.zeros(len(mus_cond))
    for a in range(len(mus_cond)):
        if mus_cond[a] > 0.0175 and mus_cond[a] < 0.0185:
            # Water vapor pressure, from Lowe 1977
            a0 = 6984.505294
            a1 = -188.9039310
            a2 = 2.133357675
            a3 = -0.01288580973
            a4 = (4.393587233) * (10**-5)
            a5 = (-8.023923082) * (10**-8)
            a6 = (6.136820929) * (10**-11)
            mb = a0 + T * (a1 + T * (a2 + T * (a3 + T * (a4 + T * (a5 + T * a6)))))
            pasc = max(mb * 100.0, 0.0)
            SVPs[a] = pasc
        elif mus_cond[a] < 0.0175 and mus_cond[a] > 0.0165:
            # Ammonia vapor pressure
            # Source: Lange's Handbook of Chemistry
            larth = 6.67956 - 1002.711 / (T - 25.215)
            larth = min(larth, 12.0)
            pasc = max(10 ** (3 + larth), 0.0)
            SVPs[a] = pasc
        elif mus_cond[a] < 0.0515 and mus_cond[a] > 0.0505:
            # NH4SH vapor pressure
            # Source: Sanchez-Lavega (2004)
            # An effective 'NH4SH vapor pressure', although the vapor is really H2S with an equilibrium constant
            # That is, really it's a chemical reaction
            Ctteh = (
                120.678 + (-2915.7 / T - 1.760 * np.log(T) + 0.00078 * T) / 0.167
            )  # From Zuchowski et al. (2009), previously 75.678
            pasc = max(100000 * np.exp(Ctteh), 0.0)
            SVPs[a] = pasc
        else:
            # unknown
            SVPs[a] = 0
    return SVPs


def get_lightning(extract: Extractor, time: int):
    p = extract.get_variable('p', time)
    T = extract.get_variable('t', time)
    H2Osolid = extract.get_variable("H_2O_solid", time)
    H2Oliquid = extract.get_variable("H_2O_liquid", time)
    H2Orain = extract.get_variable("H_2O_rain", time)
    H2Osnow = extract.get_variable("H_2O_snow", time)
    NH3solid = extract.get_variable("NH_3_solid", time)
    NH3liquid = extract.get_variable("NH_3_liquid", time)
    NH3rain = extract.get_variable("NH_3_rain", time)
    NH3snow = extract.get_variable("NH_3_snow", time)

    get_dEdt = np.vectorize(dEtotal, excluded={-1, "mu_dry"})
    p1bar = np.argmin((p[:, 0, 0] / 100 - 1000) ** 2.0)

    dEdt, invt = get_dEdt(
        H2Osolid[p1bar, :, :],
        H2Oliquid[p1bar, :, :],
        H2Osnow[p1bar, :, :],
        H2Orain[p1bar, :, :],
        NH3solid[p1bar, :, :],
        NH3liquid[p1bar, :, :],
        NH3snow[p1bar, :, :],
        NH3rain[p1bar, :, :],
        p[p1bar, :, :],
        T[p1bar, :, :],
        mu_dry=8314 / extract.get_attrs(time, "planet_rgas")["planet_rgas"],
    )

    print(dEdt.shape, invt.shape)


# Example model setup
def main():
    stdout_fileno = sys.stdout
    ndesc = '240hrs'  # parameters
    sys.stdout = open(os.path.join('Epic1', ndesc + ' log.txt'), 'w')
    print(ndesc)
    e00 = netCDF4.Dataset("lightning_outputs/extract240.nc", "r", format="NETCDF4")
    P00 = e00["/p"]
    T00 = e00["/t"]
    Wi00 = e00["/H_2O_solid"]
    Wl00 = e00["/H_2O_liquid"]
    Ws00 = e00["/H_2O_snow"]
    Wr00 = e00["/H_2O_rain"]
    Ai00 = e00["/NH_3_solid"]
    Al00 = e00["/NH_3_liquid"]
    As00 = e00["/NH_3_snow"]
    Ar00 = e00["/NH_3_rain"]
    Ps = P00[0, 21, :, :]  # 21 for deep (5.6 bar) lightning
    Ts = T00[0, 21, :, :]
    Wis = Wi00[0, 21, :, :]
    Wls = Wl00[0, 21, :, :]
    Wss = Ws00[0, 21, :, :]
    Wrs = Wr00[0, 21, :, :]
    Ais = Ai00[0, 21, :, :]
    Als = Al00[0, 21, :, :]
    Ass = As00[0, 21, :, :]
    Ars = Ar00[0, 21, :, :]
    dEdts = np.zeros([201, 256])
    invts = np.zeros([201, 256])
    fcheck = np.zeros([201, 256])
    Psbar = Ps / 100000.0
    for a in range(201):
        for b in range(256):
            ade, ait = dEtotal(
                Wis[a, b],
                Wls[a, b],
                Wss[a, b],
                Wrs[a, b],
                Ais[a, b],
                Als[a, b],
                Ass[a, b],
                Ars[a, b],
                Ps[a, b],
                Ts[a, b],
            )
            dEdts[a, b] = ade
            invts[a, b] = ait * 3600.0  # flashes per hour
            if invts[a, b] > 0.4:
                fcheck[a, b] = 1
    plt.figure(figsize=[11, 8])
    plt.imshow(dEdts, origin='lower')
    plt.title('Electric field growth (N/Cs), 5.6 bar')
    plt.colorbar()
    ncurrn = ndesc + ' field growth.png'
    plt.savefig(os.path.join('Epic1', ncurrn))
    plt.figure(figsize=[11, 8])
    plt.imshow(invts, origin='lower')
    plt.title('Flashes per hour, 5.6 bar')
    plt.colorbar()
    ncurrn = ndesc + ' frequency.png'
    plt.savefig(os.path.join('Epic1', ncurrn))
    plt.figure(figsize=[11, 8])
    plt.imshow(fcheck, origin='lower')
    plt.title('Lightning at 5.6 bar')
    plt.colorbar()
    ncurrn = ndesc + ' boolean.png'
    plt.savefig(os.path.join('Epic1', ncurrn))

    Psh = P00[0, 13, :, :]
    Tsh = T00[0, 13, :, :]
    Wish = Wi00[0, 13, :, :]
    Wlsh = Wl00[0, 13, :, :]
    Wssh = Ws00[0, 13, :, :]
    Wrsh = Wr00[0, 13, :, :]
    Aish = Ai00[0, 13, :, :]
    Alsh = Al00[0, 13, :, :]
    Assh = As00[0, 13, :, :]
    Arsh = Ar00[0, 13, :, :]
    dEdtsh = np.zeros([201, 256])
    invtsh = np.zeros([201, 256])
    fcheckh = np.zeros([201, 256])
    Pshbar = Psh / 100000.0
    for a in range(201):
        for b in range(256):
            adeh, aith = dEtotal(
                Wish[a, b],
                Wlsh[a, b],
                Wssh[a, b],
                Wrsh[a, b],
                Aish[a, b],
                Alsh[a, b],
                Assh[a, b],
                Arsh[a, b],
                Psh[a, b],
                Tsh[a, b],
            )
            dEdtsh[a, b] = adeh
            invtsh[a, b] = aith * 3600.0  # flashes per hour
            if invtsh[a, b] > 0.4:
                fcheckh[a, b] = 1
    plt.figure(figsize=[11, 8])
    plt.imshow(dEdtsh, origin='lower')
    plt.title('Electric field growth (N/Cs), 0.54 bar')
    plt.colorbar()
    ncurrn = ndesc + ' shallow field growth.png'
    plt.savefig(os.path.join('Epic1', ncurrn))
    plt.figure(figsize=[11, 8])
    plt.imshow(invtsh, origin='lower')
    plt.title('Flashes per hour, 0.54 bar')
    plt.colorbar()
    ncurrn = ndesc + ' shallow frequency.png'
    plt.savefig(os.path.join('Epic1', ncurrn))
    plt.figure(figsize=[11, 8])
    plt.imshow(fcheckh, origin='lower')
    plt.title('Lightning at 0.54 bar')
    plt.colorbar()
    ncurrn = ndesc + ' shallow boolean.png'
    plt.savefig(os.path.join('Epic1', ncurrn))

    sys.stdout.close()
    sys.stdout = stdout_fileno


if __name__ == '__main__':
    main()

