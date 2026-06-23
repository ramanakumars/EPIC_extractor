import numpy as np
import tqdm
from numba import njit

from .constants import SPECIES_ID, SPECIES_PARAMETERS

RAIN_INDEX = 0
SNOW_INDEX = 1
ICE_INDEX = 2
P0 = 1000e2  # reference pressure is 1bar

H_2O_ID: int = SPECIES_ID["H_2O"]
NH_3_ID: int = SPECIES_ID["NH_3"]

pbar = None


@njit
def dQdt6_vec(number_density, velocities, species_id, Qcoeffs, binbounds):
    nbins = len(binbounds) - 1
    nspecies = len(species_id)
    r0s = (binbounds[:-1] + binbounds[1:]) / 2.0  # bin center radius

    dQidt = np.zeros((nspecies, nbins))

    r0s_sq = r0s**2
    r0s_mat_bu_cu = r0s_sq[:, None] + r0s_sq[None, :]  # shape (nbins, nbins)
    rG_mat = np.zeros((nbins, nbins))

    for j in range(nbins):
        for i in range(nbins):
            rG_mat[j, i] = np.minimum(r0s[j], r0s[i])
    # rG_mat = np.minimum.outer(r0s, r0s)  # shape (nbins, nbins)

    Gr_mat = np.where(
        rG_mat <= 0.000111,
        0.0000271 * (1e6 * rG_mat) ** 2.7,
        0.0988 * (1e6 * rG_mat) ** 0.98,
    )

    for au in range(nspecies):
        for du in range(nspecies):
            # Filter based on species type
            spec_a, spec_d = species_id[au], species_id[du]
            qmin = min(Qcoeffs[au], Qcoeffs[du])

            water_a = spec_a == H_2O_ID
            water_d = spec_d == H_2O_ID
            ammonia_a = spec_a == NH_3_ID
            ammonia_d = spec_d == NH_3_ID

            # Shape: (nbins, nbins)
            vel_diff = np.abs(velocities[au, :, None] - velocities[du, None, :])
            vel_diff_pow = (vel_diff / 3.0) ** 2.5

            if water_a and water_d:
                # Water-water charging
                coef = 1.0
                scale = np.ones_like(vel_diff)
            elif water_a and ammonia_d:
                # Ammonia-water charging
                # Triboelectric: water gets positive charge
                # Per Lee et al. 2018 and a lot of dubious extrapolation
                # Only goes as v^1.5 based on Lesprit et al. 2020
                coef = 1.0
                scale = 20.68 / vel_diff
            elif ammonia_a and ammonia_d:
                # Ammonia-ammonia charging
                # Use same equation as for water,
                # multiplied by smaller constant (ratio of dielectric constants)
                coef = 25.0 / 80.0
                scale = np.ones_like(vel_diff)
            elif ammonia_a and water_d:
                # Water-ammonia charging
                # Triboelectric: ammonia gets negative charge
                # Per Lee et al. 2018 and a lot of dubious extrapolation
                # Only goes as v^1.5 based on Lesprit et al. 2020
                coef = -1.0  # flipped sign
                scale = 20.68 / vel_diff
            else:
                continue  # unsupported combination

            # scale_term = scale / vel_diff
            scale_term = np.where(vel_diff == 0, 0, scale / vel_diff)
            # scale_term[vel_diff == 0] = 0.0  # avoid div-by-zero

            delQ = coef * vel_diff_pow * scale_term * Gr_mat * 1e-15
            dQi = delQ * number_density[du, None, :] * np.pi * r0s_mat_bu_cu * qmin

            # antisymmetrize
            add_mask = np.arange(nbins)[:, None] < np.arange(nbins)[None, :]
            sub_mask = np.arange(nbins)[:, None] > np.arange(nbins)[None, :]

            if (water_a and water_d) or (ammonia_a and ammonia_d):
                # add when bu < cu
                dQidt[au] += np.sum(np.where(add_mask, dQi, 0), axis=1)
                # subtract when bu > cu
                dQidt[au] -= np.sum(np.where(sub_mask, dQi, 0), axis=1)
                # No change when bu == cu (diagonal)
            elif water_a and ammonia_d:
                # symmetric, triboelectric: water gets positive charge
                dQidt[au] += np.sum(np.abs(dQi), axis=1)
            elif ammonia_a and water_d:
                # symmetric, triboelectric: ammonia gets negative charge
                dQidt[au] -= np.sum(np.abs(dQi), axis=1)

    Jc = np.sum(-number_density * velocities * dQidt)
    return Jc


@njit
def dQdt6(Ns, velocities, species, Qcoeffs, binbounds):
    # velocities is particle fall velocities, rhosadj removed
    r0s = 0.5 * (binbounds[1:] + binbounds[:-1])
    dQidt = np.zeros((len(species), len(binbounds) - 1))
    # calculate r0 as bin center radius
    # charge transfer
    for au in range(len(species)):
        for bu in range(len(binbounds) - 1):
            dQitot = 0.0
            for cu in range(len(binbounds) - 1):
                for du in range(len(species)):
                    rG = min(r0s[bu], r0s[cu])
                    # transferred charge dependent on molar mass
                    # if unknown species, don't transfer any charge
                    # subject to change later
                    if species[au] == H_2O_ID:
                        if species[du] == H_2O_ID:
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
                        elif species[du] == NH_3_ID:
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
                    elif species[au] == NH_3_ID:
                        if species[du] == NH_3_ID:
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
                        elif species[du] == H_2O_ID:
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
            dQidt[au, bu] = dQitot
    Jc = np.sum(-Ns * velocities * dQidt)
    return Jc


def dEdt6(Ns, velocities, species, Qcoeffs, Ebreakdown, binbounds):
    # as above, with Ebreakdown as breakdown electric field, for the average
    # also returns 1/(time to lightning in seconds)
    # global pbar
    # calc_dQdt = np.vectorize(
    #     dQdt6_vec,
    #     excluded={-1, "binbounds"},
    #     signature='(k, m), (k, m), (k), (k) -> ()',
    # )
    # with tqdm.tqdm(total=len(Ns)) as pbar:
    #     Jc = calc_dQdt(Ns, velocities, mus, Qcoeffs, binbounds=binbounds)
    Jc = np.zeros(len(Ns))
    for i in tqdm.tqdm(range(Jc.size)):
        # Jc[i] = dQdt6_vec(Ns[i], velocities[i], species[i], Qcoeffs[i], binbounds)
        Jc[i] = dQdt6(
            Ns[i], velocities[i], species[i], Qcoeffs[i], binbounds
        )
    # Jc = 0.0
    # for g in range(len(binbounds) - 1):
    #     for h in range(len(mus)):
    #         Jcg = -Ns[h, g] * velocities[h, g] * Q1s[h, g]
    #         Jc = Jc + Jcg
    AvdE = -np.sign(Jc) * np.sqrt(0.5 * abs(Jc / (8.854 * (10**-12))) * Ebreakdown)
    invtc = abs(AvdE / Ebreakdown)
    return AvdE, invtc


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


def get_number_density_velocity(
    qice: np.ndarray,
    qsnow: np.ndarray,
    qrain: np.ndarray,
    P: np.ndarray,
    T: np.ndarray,
    species: list[str],
    Rdry: float = 3637.0,
):
    bin_edges = np.geomspace(0.00001, 10.48576, 41)
    bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2

    nspecies, nk, nj, ni = qice.shape

    qice = qice.reshape(qice.shape[0], -1)
    qsnow = qsnow.reshape(qice.shape[0], -1)
    qrain = qrain.reshape(qice.shape[0], -1)
    P = P.flatten()
    T = T.flatten()

    Ebreakdown = 5.0 * P  # Rough approximation of the electric field for hydrogen
    rho_dry = P / (Rdry * T)

    number_density = np.zeros((3 * nspecies, nk * nj * ni, len(bin_center)))
    velocity = np.zeros_like(number_density)
    Q_coefficients = np.zeros((3 * nspecies, nk * nj * ni))
    species_id = np.zeros((3 * nspecies, nk * nj * ni), dtype=int)

    for n, spec in enumerate(species):
        if spec not in SPECIES_PARAMETERS:
            raise KeyError(f"Constants for {spec} is not defined")
        species_parameters = SPECIES_PARAMETERS[spec]

        spec_id = SPECIES_ID[spec]

        species_id[3 * n + ICE_INDEX] = spec_id
        species_id[3 * n + RAIN_INDEX] = spec_id
        species_id[3 * n + SNOW_INDEX] = spec_id

        # ========= ICE ======================== #
        rho_ice = rho_dry * qice[n]
        N_ice = (
            species_parameters['ice']['c'] * (rho_ice) ** species_parameters['ice']['d']
        )
        m_ice = rho_ice / N_ice
        D_ice = (m_ice / species_parameters['ice']['alpha']) ** (
            1 / species_parameters['ice']['beta']
        )

        # get the bin index corresponding to each (k, j, i) grid
        diff = np.linalg.outer(2.0 / D_ice, bin_center) - 1
        bin_idx = np.argmin(diff**2.0, axis=1)
        xx, yy = np.unravel_index(
            bin_idx + np.arange(nk * nj * ni) * len(bin_center),
            (nk * nj * ni, len(bin_center)),
        )

        # set the number density at these indices to the number density of ice
        number_density[3 * n + ICE_INDEX, xx, yy] = N_ice

        Q_coefficients[3 * n + ICE_INDEX, :] = species_parameters['ice']['Q']

        # ============== RAIN ==================== #
        rho_rain = rho_dry * qrain[n]
        # lambda is the intercept parameter for the log-normal size distribution
        lambda_rain = np.power(
            species_parameters['rain']['N0']
            * np.pi
            * species_parameters['liquid_density']
            / rho_rain,
            0.25,
        )
        lambda_rain = np.repeat(lambda_rain[:, np.newaxis], len(bin_edges), axis=-1)
        # integrate the log normal size function between bin_edges[k] to bin_edges[k + 1]
        number_density_rain = np.asarray(
            0.5
            * (species_parameters['rain']['N0'] / lambda_rain)
            * np.exp(-2 * lambda_rain * bin_edges)
        )
        number_density_rain[~np.isfinite(number_density_rain)] = 0.0

        # get the difference between the integrand at the bin edges
        number_density[3 * n + RAIN_INDEX, :, :] = (
            number_density_rain[:, :-1] - number_density_rain[:, 1:]
        )

        Q_coefficients[3 * n + RAIN_INDEX, :] = species_parameters['rain']['Q']

        # ============== SNOW ==================== #
        rho_snow = rho_dry * qsnow[n]
        N0_snow = species_parameters['snow']['N0'] * np.clip(
            0.01 * np.exp(-0.12 * (T - species_parameters["T_tp"])), a_min=None, a_max=1
        )

        # lambda is the intercept parameter for the log-normal size distribution
        lambda_snow = (
            N0_snow * np.pi * 0.5 * species_parameters['bulk_ice_density'] / rho_snow
        ) ** 0.25
        N0_snow = np.repeat(N0_snow[:, np.newaxis], len(bin_edges), axis=-1)
        lambda_snow = np.repeat(lambda_snow[:, np.newaxis], len(bin_edges), axis=-1)
        # integrate the log normal size function between bin_edges[k] to bin_edges[k + 1]
        number_density_snow = np.asarray(
            0.5 * (N0_snow / lambda_snow) * np.exp(-lambda_snow * 2 * bin_edges)
        )
        # get the difference between the integrand at the bin edges
        number_density[3 * n + SNOW_INDEX, :, :] = (
            number_density_snow[:, :-1] - number_density_snow[:, 1:]
        )
        number_density_snow[~np.isfinite(number_density_snow)] = 0.0

        Q_coefficients[3 * n + SNOW_INDEX, :] = species_parameters['snow']['Q']

        for i, phase in enumerate(['rain', 'snow', 'ice']):
            vi = (
                species_parameters[phase]["x"]
                * (2 * bin_center) ** species_parameters[phase]["y"],
            )
            velocity[3 * n + i, :, :] = np.outer(
                (P0 / P) ** species_parameters[phase]["gamma"], vi
            )

    return number_density, velocity, species_id, Q_coefficients, Ebreakdown, bin_edges


def get_dEtotal(
    number_density: np.ndarray,
    velocity: np.ndarray,
    species_id: np.ndarray,
    Q_coefficients: np.ndarray,
    Ebreakdown: np.ndarray,
    bin_edges: np.ndarray,
):
    dEdt, invtc =  dEdt6(
        np.transpose(number_density, (1, 0, 2)),
        np.transpose(velocity, (1, 0, 2)),
        species_id.T,
        Q_coefficients.T,
        Ebreakdown,
        binbounds=bin_edges,
    )
    return dEdt, invtc 
