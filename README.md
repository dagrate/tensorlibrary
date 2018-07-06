
# tensorlibrary
tensorlibrary is a Python library that focuses on less commonly known tensor decompositions such as DEDICOM and PARATUCK2. CP/PARAFAC is also available since it is the most popular. The library allows to perform tensor decomposition and tensor algebra in a simple way.

As of today, the following decompositions and resolutions are available:
- CP/PARAFAC, nn-ALS resolution
- DEDICOM, ALS and nn-ALS resolution
- PARATUCK2, ALS and nn-ALS resolution

----------------------------

## Dependencies

The library uses **Python 3** and the following modules:
- numpy (pip install numpy)
- numba (pip install numba)

It is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

----------------------------

## Quick Start

We create a tensor of size 3x4x5 and we perform PARATUCK2 decomposition using non-negative Alternating-Least-Square resolution for latent factors (P=6,Q=2).

```python
from PARATUCK2_Decomposition import paratuck2
from tnsr_utils import tenseur_init

tnsr_size = (3, 4, 5)
latfact = (6,2)
X = tenseur_init((tnsr_size))
A, DA, R, DB, B = paratuck2(X, latfact, 
                                decompType="PARATUCK2",  
                                maxiter=1000, eps=1.0E-8, 
                                eps_objfnct = 1.0E-3, NonNgtv="Y")
```

We can rebuild the tensor inherited from the decomposition using 
```python
print("\nX_hat = ", opt.build_PRTCK2(tnsr_size, A, DA, R, DB, B.T))
```

----------------------------

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
