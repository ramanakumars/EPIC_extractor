import numpy as np


''' adapted from EPIC
    see thermo_setup() and return_cp() in epic_funcs_diag.c
'''

# constants
MDIM_THERMO = 128
NDIM_THERMO = 16
THLO_THERMO = 20.
THHI_THERMO = 600.
CCOLN_THERMO = -2.105769
CCPLN_THERMO = -1.666421
CPRH2 = 2.5
CPRHE = 2.5
CPR3 = 3.5

M_PROTON = 1.67262158e-27
H_PLANCK = 6.62606876e-34
K_B = 1.3806503e-23
N_A = 6.02214199e23


class Planet():
    def __init__(self, xh2, xhe, x3, cpr, rgas, p0):
        self.xh2 = xh2
        self.xhe = xhe
        self.x3 = x3
        self.cpr = cpr
        self.rgas = rgas
        self.p0 = p0
        self.kappa = 1. / self.cpr

        self.thermo_setup()

    @classmethod
    def from_extract(cls, extract):
        cls.__init__(extract.xh2, extract.xhe, extract.x3, extract.cpr, extract.Ratmo, extract.p0)

        return cls

    def thermo_setup(self):
        self.t_grid = np.zeros(MDIM_THERMO)
        self.theta_grid = np.zeros(MDIM_THERMO)
        self.tharray = np.zeros((5, MDIM_THERMO))  # thermo.array in EPIC
        self.theta_array = np.zeros((2, MDIM_THERMO))

        a = np.zeros(8)
        z = np.zeros((3, 2))
        jmax = 50
        jn = np.zeros(2, dtype=np.int)

        temp = np.zeros(MDIM_THERMO)
        tho = np.zeros(MDIM_THERMO)
        thp = np.zeros(MDIM_THERMO)

        ndegeneracy = np.array([3., 1.])

        # xh2 = self.xh2
        # xhe = self.xhe
        # x3 = self.x3

        # if(xh2 > 0.):
        #     cpr_out = (CPRH2*xh2 + CPRHE*xhe + CPR3*x3)/(xh2+xhe+x3)
        # else:
        #     cpr_out = self.cpr

        '''
            calculate hydrogen (H2) properties
            subscript o refers to ortho- and p for para-hydrogen
        '''

        # reference temp and pressure used to define
        # the mean potential temperature
        t0 = 1.
        p0 = 1.e5

        c1 = np.log(K_B * t0 / p0 * (2. * np.pi *
                                     (M_PROTON / H_PLANCK) *
                                     (K_B * t0 / H_PLANCK))**1.5)
        c2 = np.log(9.)
        p = p0
        theta = 87.567

        for i in range(MDIM_THERMO):
            temperature = 500. * (float(i + 1) / float(MDIM_THERMO))
            if (temperature < 10.):
                ho = CPRH2 * temperature
                hp = CPRH2 * temperature
                pottempo = temperature
                pottempp = temperature
                a[0] = 1.
                a[1] = 175.1340
                a[2] = 0.
                a[3] = 0.
                a[4] = 0.
                a[5] = 0.
                a[6] = 0.
                a[7] = 0.
                ff = 0.
            else:
                y = theta / temperature
                y = np.min([y, 30.])
                z = np.zeros((3, 2))

                for j in range(1, jmax + 1):
                    jn[0] = 2 * j - 1
                    jn[1] = jn[0] - 1
                    for n in range(2):
                        term = ndegeneracy[n] * (2 * jn[n] + 1) *\
                            np.exp(-jn[n] * (jn[n] + 1) * y)
                        for m in range(3):
                            z[m, n] += term
                            if (m < 2):
                                term *= jn[n] * (jn[n] + 1)

                    if ((j > 1) & (term < 1.e20)):
                        break

                den = z[0, 0] + z[0, 1]

                a[0] = z[0, 1] / den

                for n in range(2):
                    a[n + 1] = theta * z[1, n] / z[0, n]
                    a[n + 3] = y * y * (z[0, n] * z[2, n] - z[1, n] *
                                        z[1, n]) / (z[0, n] * z[0, n])
                    a[n + 5] = np.log(z[0, n])

                a[7] = (1. - a[0]) * a[3] + a[0] * a[4] + \
                    (a[2] - a[1]) * y / temperature * \
                    (z[1][1] * z[0][0] - z[0][1] * z[1][0]) / (den * den)

                ho = a[1] + CPRH2 * temperature - 2. * theta
                hp = a[2] + CPRH2 * temperature

                so = -np.log(p / p0) + 2.5 * np.log(temperature)\
                    + 1.5 * np.log(2.0) + c1 + (ho + 2. *
                                                theta) / temperature + a[5]
                sp = -np.log(p / p0) + 2.5 * np.log(temperature)\
                    + 1.5 * np.log(2.0) + c1 + hp / temperature + a[6]

                pottempo = np.exp(
                    0.4 * (so - c2 - 1.5 * np.log(2.0) - c1 - 2.5))
                pottempp = np.exp(0.4 * (sp - 1.5 * np.log(2.0) - c1 - 2.5))

                ff = -(so - c2 - 1.5 * np.log(2.0) -
                       c1 - 2.5 -
                       ho / temperature)\
                    + (sp - 1.5 * np.log(2.0) - c1 - 2.5 - hp / temperature)
                ff *= temperature
                temp[i] = temperature
                tho[i] = pottempo
                thp[i] = pottempp

                self.tharray[0][i] = ho
                self.tharray[1][i] = hp
                self.tharray[2][i] = ff
                self.tharray[3][i] = a[0]
                self.tharray[4][i] = a[1] - a[2]
                self.t_grid[i] = temperature
                self.theta_array[0][i] = pottempo
                self.theta_array[1][i] = pottempp

    def return_enthalpy(self, temperature, pressure):
        '''
            assume that we are dealing with H/He atmosphere
            and fpara is off
        '''
        fp = 0.25

        if (temperature < 20.):
            enthalpy = self.cpr * temperature
            fgibb = 0.
            uoup = 175.1340

        elif (temperature > 500.):
            ho = 1545.3790 + 3.5 * (temperature - 500.)
            hp = 1720.3776 + 3.5 * (temperature - 500.)

            enthalpy = (self.xh2) * ((1. - fp) * ho + fp * hp)\
                + (self.xhe * 2.5 + self.x3 * 3.5) * temperature
            fgibb = self.xh2 * 2.5 * (CCPLN_THERMO - CCOLN_THERMO) *\
                temperature - self.xh2 * (hp - ho)
            uoup = 0.
        else:
            em = float(MDIM_THERMO - 1) *\
                (temperature - self.t_grid[0]) /\
                (self.t_grid[MDIM_THERMO - 1] - self.t_grid[0])
            m = int(em)

            if (m == MDIM_THERMO - 1):
                m -= 1
                fract = 1.
            else:
                fract = np.fmod(em, 1.)

            thermo_vector = np.zeros(5)
            for j in range(5):
                thermo_vector[j] = (1. - fract) * self.tharray[j][m]\
                    + (fract) * self.tharray[j][m + 1]

            enthalpy = (self.xh2) * ((1. - fp) * thermo_vector[0]
                                     + (fp) * thermo_vector[1])\
                + (self.xhe * 2.5 + self.x3 * 3.5) * temperature
            fgibb = self.xh2 * thermo_vector[2]
            uoup = thermo_vector[4]

        enthalpy *= self.rgas
        fgibb *= self.rgas
        uoup *= self.rgas

        return (enthalpy, fgibb, uoup)

    def return_cp(self, p, t):
        epsilon = 1.e-6
        deltaT = t * epsilon

        h2, _, _ = self.return_enthalpy(t + deltaT, p)
        h1, _, _ = self.return_enthalpy(t - deltaT, p)

        cp = (h2 - h1) / (2. * deltaT)

        return cp

    def return_theta(self, p, t, fp=0.25):
        if self.xh2 == 0.:
            # No hydrogen, so use standard definition of theta.
            theta = t * np.power(self.p0 / p, self.kappa)
        else:
            # Use mean theta as defined in Dowling et al (1998), to handle ortho/para hydrogen.
            if t <= 20.:
                theta = t
            elif t > 500.:
                cc = self.xh2 * 2.5 * ((1. - fp) * CCOLN_THERMO + fp * CCPLN_THERMO)
                theta = np.exp(cc / self.cpr) * \
                    np.power(t, ((3.5 * self.xh2 + 2.5 * self.xhe + 3.5 * self.x3) / self.cpr))
            else:
                # 0 < em < MDIM_THERMO-1
                em = (MDIM_THERMO - 1) * (t - self.t_grid[0]) / (self.t_grid[MDIM_THERMO - 1] - self.t_grid[0])
                m = int(em)
                #  0 < m < MDIM_THERMO-2
                if (m == MDIM_THERMO - 1):
                    m -= 1
                    fract = 1.
                else:
                    fract = np.fmod(em, 1.)

                thermo_vector = np.zeros(2)
                for j in range(2):
                    thermo_vector[j] = (1. - fract) *\
                        self.theta_array[j][m] +\
                        fract * self.theta_array[j][m + 1]

                thetaln = (self.xh2) * \
                    ((1. - fp) * np.log(thermo_vector[0]) + ((fp) * np.log(thermo_vector[1]))) +\
                    (self.xhe * 2.5 + self.x3 * 3.5) * np.log(t) / self.cpr

                theta = np.exp(thetaln)
            pp = pow(self.p0 / p, self.kappa)
            theta *= pp

        return theta
