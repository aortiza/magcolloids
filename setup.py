import setuptools
with open("Readme.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="magcolloids",
    version="0.3.10",
    author="Antonio Ortiz-Ambriz",
    author_email="aortiza@gmail.com",
    description="A set of routines to interface with lammps and setup simulations of magnetic colloidal particles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aortiza/magcolloid",
    packages=setuptools.find_packages(),
    install_requires = ['numpy','scipy','pandas','matplotlib','pint'],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
    )
)