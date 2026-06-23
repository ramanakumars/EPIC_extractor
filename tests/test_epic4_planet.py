import os


from epicpy.extractor import Extractor
from epicpy.planet import Planet
from epicpy.planet.utils import get_planet_from_name

root_folder = os.path.dirname(__file__)


class TestClass:
    def test_planet_initialization(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))

        planet = Planet.from_extract(extract)

        assert planet.return_cp(1000e2, 150) == 10727.778081006061

    def test_default_initialization(self):
        planet = get_planet_from_name("JUPITER")

        assert planet.return_cp(1000e2, 150) == 10727.778081006061
