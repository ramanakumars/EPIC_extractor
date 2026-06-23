import os

import netCDF4 as nc
import numpy as np
import pytest

from epicpy.extractor import Extractor

root_folder = os.path.dirname(__file__)


class TestClass:
    def test_number_of_extracts(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))
        extract.add_restarts(os.path.join(root_folder, 'epic4_test/restart1/'))

        assert len(extract.time) == 21

    def test_number_of_extracts_without_restart(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))

        assert len(extract.time) == 11

    def test_missing_folder(self):
        with pytest.raises(FileNotFoundError):
            Extractor(os.path.join(root_folder, 'epic4_bad_folder/'))

    def test_missing_restart_folder(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))
        extract.add_restarts(os.path.join(root_folder, 'epic4_test/restart_missing/'))

        assert len(extract.time) == 11

    def test_variable(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))

        with nc.Dataset(
            os.path.join(root_folder, "epic4_test/extract100.nc")
        ) as indset:
            variable = indset.variables["H_2O_vapor"][0, :]

        assert np.all(variable == extract.get_variables("H_2O_vapor", 0)["H_2O_vapor"])

    def test_missing_variable(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))
        with pytest.raises(KeyError):
            extract.get_variables("missing")

    def test_variable_multiple_times(self):
        extract = Extractor(os.path.join(root_folder, 'epic4_test/'))

        variable = []
        for i in range(5):
            with nc.Dataset(
                os.path.join(root_folder, f"epic4_test/extract10{i}.nc")
            ) as indset:
                variable.append(indset.variables["H_2O_vapor"][0, :])

        assert np.all(
            np.asarray(variable)
            == extract.get_variables("H_2O_vapor", range(5))["H_2O_vapor"]
        )
