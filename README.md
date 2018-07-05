# tensorlibrary
Python Tensor Library 

Tensor Resolution Algorithms for Various Decomposition

As of today, the following decompositions and resolutions are available:
- DEDICOM, ALS and nn-ALS resolution
- PARATUCK2, ALS and nn-ALS resolution

## Dependencies
The library uses the following modules:
- numpy (pip install numpy)
- numba (pip install numba)

It is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

## Citation
Please cite the following article if you use the library for your work and/or publications:

J. Charlier, R. State & J. Hilger (2018, January). Non-negative Paratuck2 Tensor Decomposition Combined to LSTM Network for Smart Contracts Profiling. In Big Data and Smart Computing (BigComp), 2018 IEEE International Conference on (pp. 74-81). IEEE.
