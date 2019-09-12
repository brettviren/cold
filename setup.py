
from setuptools import setup, find_packages

setup(name = 'cold',
      version = '0.0',
      description = 'CUDA Operations for Liquid argon Detector is a contrived acronym',
      author = 'Brett Viren',
      author_email = 'brett.viren@gmail.com',
      license = 'GPLv2',
      url = 'http://github.com/brettviren/cold',
      packages = find_packages(),
      install_requires=[l.strip() for l in open("requirements.txt").readlines()],
      dependency_links = [
      ],
      entry_points = {
          'console_scripts': [
              'cold = cold.main:main',
          ]
      }
)
