import numpy as np

from ...extractor import Extractor
from .utils import get_dEtotal, get_number_density_velocity


def get_lightning(extract: Extractor, time: int, subsample: int = 1):
    p = extract.get_variable('p', time)[:, ::subsample, ::subsample]
    T = extract.get_variable('t', time)[:, ::subsample, ::subsample]
    H2Osolid = extract.get_variable("H_2O_solid", time)[:, ::subsample, ::subsample]
    H2Orain = extract.get_variable("H_2O_rain", time)[:, ::subsample, ::subsample]
    H2Osnow = extract.get_variable("H_2O_snow", time)[:, ::subsample, ::subsample]
    NH3solid = extract.get_variable("NH_3_solid", time)[:, ::subsample, ::subsample]
    NH3rain = extract.get_variable("NH_3_rain", time)[:, ::subsample, ::subsample]
    NH3snow = extract.get_variable("NH_3_snow", time)[:, ::subsample, ::subsample]

    qrain = np.asarray([H2Orain, NH3rain])
    qsnow = np.asarray([H2Osnow, NH3snow])
    qsolid = np.asarray([H2Osolid, NH3solid])

    Ns, velocity, species_ids, Q_coefficients, Ebreakdown, sizes = (
        get_number_density_velocity(
            qsolid,
            qsnow,
            qrain,
            p,
            T,
            ['H_2O', 'NH_3'],
            extract.get_attrs("planet_rgas", time),
        )
    )

    dEdt, invt = get_dEtotal(
        Ns, velocity, species_ids, Q_coefficients, Ebreakdown, sizes
    )

    return (
        dEdt.reshape(qrain[0].shape),
        invt.reshape(qrain[0].shape),
        Ns.reshape((qrain.shape[0] * 3, *qrain[0].shape, sizes.size - 1)),
        velocity.reshape((qrain.shape[0] * 3, *qrain[0].shape, sizes.size - 1)),
        sizes,
    )
