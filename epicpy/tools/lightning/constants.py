SPECIES_PARAMETERS = {
    "H_2O": {
        "mean_molecular_weight": 18.015 / 1000,
        "T_tp": 273.16,
        "bulk_ice_density": 917.0,
        "liquid_density": 1000.0,
        "ice": {
            "c": 5.38e7,
            "d": 0.75,
            "alpha": 7.06165e-3,
            "beta": 2.0,
            "x": 535.41,
            "y": 0.8746,
            "gamma": 0.1264,
            "Q": 1.0,
        },
        "snow": {"N0": 2.0e8, "x": 42.83, "y": 0.5271, "Q": 0.5, "gamma": 0.3043},
        "rain": {"N0": 8.0e6, "x": 2376.36, "y": 0.7308, "Q": 0.2, "gamma": 0.3568},
        "liquid": {"N0": 8.0e6, "x": 2376.36, "y": 0.7308, "Q": 0.2, "gamma": 0.3568},
    },
    "NH_3": {
        "mean_molecular_weight": 17.031 / 1000,
        "T_tp": 195.5,
        "bulk_ice_density": 768.0,
        "liquid_density": 733.0,
        "ice": {
            "c": 5.38e7,
            "d": 0.75,
            "alpha": 7.06165e-3 * (768 / 917),
            "beta": 2.0,
            "x": 495.22,
            "y": 0.8807,
            "gamma": 0.1248,
            "Q": 1.0,
        },
        "snow": {"N0": 2.0e8, "x": 41.74, "y": 0.5386, "Q": 0.5, "gamma": 0.3008},
        "rain": {"N0": 8.0e6, "x": 2479.0, "y": 0.76217, "Q": 0.2, "gamma": 0.3568},
        "liquid": {"N0": 8.0e6, "x": 2479.0, "y": 0.76217, "Q": 0.2, "gamma": 0.3568},
    },
}

SPECIES_ID = {
    key: list(SPECIES_PARAMETERS.keys()).index(key) for key in SPECIES_PARAMETERS
}
