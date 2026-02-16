import numpy as np
from numba import njit
from .lightning_old import dEdt6

@njit
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
    Ns = np.zeros((6, len(binbounds) - 1))
    velocities = np.zeros((6, len(binbounds) - 1))
    # Dry gas density
    rho_dry = P * mu_dry / (8.314 * T)
    # Ns, water and ammonia ice
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
        Ns[1, H2Oliquidbin] = 0. ##53800000.0 * ((rho_dry * qH2Oliquidcloud) ** 0.75)
    D_NH3liquid = (np.cbrt(786.8 / 733.0)) * np.sqrt(
        ((rho_dry * qNH3liquidcloud) ** 0.25) / (5.38 * 70616.5 * (786.8 / 917.0))
    )
    NH3liquidbin = 0
    for bbcheck in range(len(binbounds) - 1):
        if D_NH3liquid >= (2 * binbounds[bbcheck + 1]):
            NH3liquidbin = NH3liquidbin + 1
    if qNH3liquidcloud > 0:
        Ns[4, NH3liquidbin] = 0. ## 53800000.0 * ((rho_dry * qNH3liquidcloud) ** 0.75)
    # Okay, now precipitation Ns
    if qH2Orain > 0:
        lambda_H2Orain = (8000000.0 * np.pi * 1000.0 / (rho_dry * qH2Orain)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[1, s] = Ns[1, s] + 0.5 * (8000000.0 / lambda_H2Orain) * (
                np.exp(-2 * lambda_H2Orain * binbounds[s])
                - np.exp(-2 * lambda_H2Orain * binbounds[s + 1])
            )
    if qNH3rain > 0:
        lambda_NH3rain = (8000000.0 * np.pi * 1000.0 / (rho_dry * qNH3rain)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[4, s] = Ns[4, s] +  0.5 * (8000000.0 / lambda_NH3rain) * (
                np.exp(-2 * lambda_NH3rain * binbounds[s])
                - np.exp(-2 * lambda_NH3rain * binbounds[s + 1])
            )
    if qH2Osnow > 0:
        N0snowH2O = 200000000.0 * min(1.0, 0.01 * np.exp(-0.12 * (T - 273.16)))
        lambda_H2Osnow = (N0snowH2O * np.pi * 1000.0 / (rho_dry * qH2Osnow)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[2, s] = 0.5 * (N0snowH2O / lambda_H2Osnow) * (
                np.exp(-2 * lambda_H2Osnow * binbounds[s])
                - np.exp(-2 * lambda_H2Osnow * binbounds[s + 1])
            )
    if qNH3snow > 0:
        N0snowNH3 = 200000000.0 * min(1.0, 0.01 * np.exp(-0.12 * (T - 195.40)))
        lambda_NH3snow = (N0snowNH3 * np.pi * 1000.0 / (rho_dry * qNH3snow)) ** 0.25
        for s in range(len(binbounds) - 1):
            Ns[5, s] = 0.5 * (N0snowNH3 / lambda_NH3snow) * (
                np.exp(-2 * lambda_NH3snow * binbounds[s])
                - np.exp(-2 * lambda_NH3snow * binbounds[s + 1])
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
    # return Ns, velocities, mus, Qcoeffs, Ebreakdown
    dEdt, invtc, dQidt = dEdt6(Ns, binbounds, velocities, mus, Qcoeffs, Ebreakdown)
    return Ns, velocities, Qcoeffs, dEdt, invtc, dQidt

