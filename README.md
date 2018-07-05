# tensorlibrary
Python Tensor Library 

Tensor Resolution Algorithms for Various Decomposition

As of today, the following decompositions and resolutions are available:
- DEDICOM, ALS and nn-ALS resolution
- PARATUCK2, ALS and nn-ALS resolution


## Dependencies
The library uses **Python 3** and the following modules:
- numpy (pip install numpy)
- numba (pip install numba)

It is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

## Citing
If you use tensorlibrary in an academic paper, please cite::

```bibtex
@inproceedings{charlier2018non,
  title={Non-negative Paratuck2 Tensor Decomposition Combined to LSTM Network for Smart Contracts Profiling},
  author={Charlier, Jeremy and State, Radu and others},
  booktitle={Big Data and Smart Computing (BigComp), 2018 IEEE International Conference on},
  pages={74--81},
  year={2018},
  organization={IEEE}
}
```
