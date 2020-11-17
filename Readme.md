# confined-colloids

This is a set of routines to start and read and process simulations in lammps of colloidal paramagnetic particles confined to WCA walls. It's mostly a wrapper for lammps, although that might change in the future. It is made to work on windows, and it therefore interfaces with lammps through input scripts and dump files, instead of using the python interface. 
Documentation can be found in https://aortiza.github.io/magcolloids/.

### Prerequisites
Most things are tested using numpy 1.13.1, scipy 1.0.0 and pandas 0.22.0. A lammps executable is needed, which includes the dipole package, and a superparamagnetic atom type. The source code for this can be found in https://github.com/aortiza/lammps which is a branch of the original lammps http://lammps.sandia.gov/. Executables for Windows, Intel Mac and Ubuntu can be found in the folder lammps_executables/


## Authors

* **Antonio Ortiz-Ambriz** 

## License

This project is licensed under the GPL License. In fact, most of the code is licenced under MIT licence, but LAMMPS is GPL. 
