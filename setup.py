import setuptools

with open("Readme.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="magcolloid",
    version="0.3.0",
    author="Antonio Ortiz-Ambriz",
    author_email="aortiza@gmail.com",
    description="A set of routines to interface with lammps and setup simulations of magnetic colloidal particles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aortiza/magcolloid",
    packages=setuptools.find_packages(),
    install_requires = ['numpy','scipy','pandas','matplotlib','ureg','jsonpickle']
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows, MacOS X(intel), Linux",
    ),
)