import numpy as np

from ...extractor import Extractor
from .utils import dEtotal


def get_lightning(extract: Extractor, time: int, subsample: int = 1):
    p = extract.get_variable('p', time)[:, ::subsample, ::subsample]
    T = extract.get_variable('t', time)[:, ::subsample, ::subsample]
    H2Osolid = extract.get_variable("H_2O_solid", time)[:, ::subsample, ::subsample]
    H2Oliquid = extract.get_variable("H_2O_liquid", time)[:, ::subsample, ::subsample]
    H2Orain = extract.get_variable("H_2O_rain", time)[:, ::subsample, ::subsample]
    H2Osnow = extract.get_variable("H_2O_snow", time)[:, ::subsample, ::subsample]
    NH3solid = extract.get_variable("NH_3_solid", time)[:, ::subsample, ::subsample]
    NH3liquid = extract.get_variable("NH_3_liquid", time)[:, ::subsample, ::subsample]
    NH3rain = extract.get_variable("NH_3_rain", time)[:, ::subsample, ::subsample]
    NH3snow = extract.get_variable("NH_3_snow", time)[:, ::subsample, ::subsample]

    qrain = np.asarray([H2Orain, NH3rain])
    qsnow = np.asarray([H2Osnow, NH3snow])
    qsolid = np.asarray([H2Osolid, NH3solid])
    qliquid = np.asarray([H2Oliquid, NH3liquid])

    print(qsnow.min(), qsnow.max())

    dEdt, invt, Ns, velocity, sizes = dEtotal(
        qsolid,
        qliquid,
        qsnow,
        qrain,
        p,
        T,
        ['H_2O', 'NH_3'],
        extract.get_attrs("planet_rgas", time),
    )

    return (
        dEdt.reshape(qrain[0].shape),
        invt.reshape(qrain[0].shape),
        Ns.reshape((qrain.shape[0] * 4, *qrain[0].shape, sizes.size - 1)),
        velocity.reshape((qrain.shape[0] * 4, *qrain[0].shape, sizes.size - 1)),
        sizes,
    )
