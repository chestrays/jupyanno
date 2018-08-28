try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='jupyanno',
      version='0.0.1',
      description='Easy for doing radiology in python',
      long_description=open('README.md').read(),
      url='https://www.github.com/chestrays/jupyanno',
      license='Apache',
      author='ChestRays Team',
      packages=['jupyanno'],
      install_requires=['numpy', 'pandas', 'notebook', 'ipywidgets'],
      extras_require={
          'plots': ["matplotlib", "plotly"]
      }
      )