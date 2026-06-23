from .planet import Planet
from .planet_properties import PLANETS


def get_planet_from_name(planet_name: str) -> Planet:
    planet_properties = PLANETS[planet_name]
    return Planet(**planet_properties, p0=1000e2)
