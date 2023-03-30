import numpy as np


R_GAS = 8.314472e+3


def fit_ellipse(v):
    '''
        from NUMERICALLY  STABLE  DIRECT  LEAST  SQUARESFITTING  OF  ELLIPSES
        (Halir and Flusser)

        Used to fit ellipses to a set of 2-D points. Used in Hadland
        et al. (2020) and Sankar et al. (2021)
    '''

    x = v[:, 0]
    y = v[:, 1]
    nx = x.shape[0]

    # python version
    x = x.reshape((nx, 1))
    y = y.reshape((nx, 1))

    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]

    A, B, C, D, F, G = a
    B = B / 2.
    D = D / 2.
    F = F / 2.

    disc = B**2. - A * C
    x0 = (C * D - B * F) / disc
    y0 = (A * F - B * D) / disc

    a = np.sqrt((2 * (A * F**2. + C * D**2. + G * B**2. -
                2 * B * D * F - A * C * G)) /
                (disc * (np.sqrt((A - C)**2. + 4 * B**2) - (A + C))))
    b = np.sqrt((2 * (A * F**2. + C * D**2. + G * B**2. -
                2 * B * D * F - A * C * G)) /
                (disc * (-np.sqrt((A - C)**2. + 4 * B**2) - (A + C))))

    if (B == 0):
        if (A < C):
            alpha = 0.
        else:
            alpha = np.pi / 2.
    else:
        alpha = np.arctan2((C - A - np.sqrt((A - C)**2. + B**2.)), B)

    return (x0, y0, a, b, alpha)


def get_density(planet, p, t, mu):
    # temp = planet.return_temp(p, theta, mu)
    density = p * mu / (R_GAS * t)

    return density


def get_brunt2(planet, pressure, temp, mu, g=22.67):
    brunt2 = np.zeros_like(temp)
    rho = np.zeros_like(temp)

    for k, (p, t) in enumerate(zip(pressure, temp)):
        rho[k] = get_density(planet, p, t, mu)

    Drho_Dp = np.gradient(rho) / np.gradient(pressure)

    for k, (p, t) in enumerate(zip(pressure, temp)):
        cp = planet.return_cp(p, t)
        dp = 0.001 * p
        dT = 0.001 * t

        drho_dp_T = (get_density(planet, p + dp, t, mu) -
                     get_density(planet, p - dp, t, mu)) / (2 * dp)

        drho_dT_p = (get_density(planet, p, t + dT, mu) -
                     get_density(planet, p, t - dT, mu)) / (2 * dT)

        brunt2[k] = g * g * (Drho_Dp[k] - drho_dp_T +
                             (t / (cp * rho[k] * rho[k])) * drho_dT_p * drho_dT_p)
    return brunt2
