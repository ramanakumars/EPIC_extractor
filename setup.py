from distutils.core import setup
from epic_extractor import __version__

setup(name='EPIC_Extractor', 
    version=__version__,
    description='Python scripts to extract EPIC outputs',
    author='Ramanakumar Sankar',
    url='https://github.com/ramanakumars/EPIC_extractor',
    packages=['epic_extractor'],
    install_requires=[
        'numpy',
        'netCDF4',
    ]
)