import os


from epic_extractor.extractor import Extractor
from epic_extractor.thermo import Planet

root_folder = os.path.dirname(__file__)


class TestClass:
    def test_planet_initialization(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))

        planet = Planet.from_extract(extract)

        assert planet.return_cp(1000e2, 150) == 10727.778081006061
