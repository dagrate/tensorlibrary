"""
    Description
    ----------
    Supporting functions to be used by the tensor resolution files. 
    Functions might use numba library to save computation time.
    
    References
    ----------
    .. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and Applications", 
       2009, DOI 10.1137/0707111X\n
    .. [2] Tamara G. Kolda, "Multilinear Operators for High-Order Decompositions", 
       April 2006, SAND2006-2081\n       
    .. [3] Rasmus Bro, "Multi-Way Analysis In The Food Industry, Models, Algorithms & Applications", 
       Ph.D. thesis, University of Amsterdam, 1998\n

    Version
    ----------
    1.0 | 01 Jan 2017 | Jeremy CHARLIER | Initial Creation\n
    2.0 | 03 Jun 2018 | Jeremy CHARLIER | Optimization using numba
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def tnsr_norm(X):
    return np.sqrt(np.sum(X**2))

    
@jit(nopython=True)
def m_norm(M):
    return np.linalg.norm(M)
    
    
@jit(nopython=True)
def m_size(C):
    res=np.zeros((2))
    res[0]=len(C)
    res[1]=len(C[1])
    return res
    
    
@jit(nopython=True)
def kr_size(C_dim, D_dim):
    res=np.zeros((2))
    res[0]=C_dim[0]*D_dim[0]
    res[1]=C_dim[1]
    return res

    
@jit(nopython=True)   
def kr_prod(Xkr, C, D):
    Clin=C.shape[0];Ccol=C.shape[1]
    Dlin=D.shape[0];Dcol=D.shape[1]
    if Ccol != Dcol:
        raise ValueError("All matrices must have the same number of columns.")
    for jtr in range(Clin):
        Xkr[Dlin*jtr:Dlin*(jtr+1)] = np.multiply(C[jtr].T, D)
    return Xkr

@jit(nopython=True)
def nmb_krao(C, D, clin, ccol, dlin, arr):
    for jtr in range(clin):
        arr[dlin*jtr:dlin*(jtr+1)] = C[jtr].T * D
    return arr

    
@jit(nopython=True)
def nmb_kron(ar01, ar02):
    """ 
    Perform Kronecker product using numpy implementation. 
    Use kronnb for implementation using numba.
    """
    return np.kron(ar01, ar02)
    
@jit(nopython=True)
def jitted_loop(res, A, B):
    """ 
    Perform Kronecker product between A and B and store the result in res. 
    """
    ni = 0
    nf = len(B)
    n_inc = len(B)
    
    for n in range(len(A)):
        mi = 0
        mf = len(B[0])
        m_inc = len(B[0])
        if n != 0:
            ni = nf
            nf += n_inc
            
        for m in range(len(A[0])):
            res[ni:nf, mi:mf] = A[n,m] * B
            mi = mf
            mf += m_inc

    return res

@jit
def kronnb(A, B):
    """ 
    Perform Kronecker product using numba implementation. 
    """
    shape = ( A.shape[0]*B.shape[0], A.shape[1]*B.shape[1] )
    res = np.zeros(shape)
    return jitted_loop(res, A, B)


@jit(nopython=True)
def nfd_size(allmodeN, mode):
    res=np.zeros((2))
    col=1
    if mode==1:
        lin=allmodeN[-2]
    if mode==2:
        lin=allmodeN[-1]
    if mode==3:
        lin=allmodeN[-3]
    if mode==4:
        lin=allmodeN[-4]
    col=np.prod(allmodeN)/lin
    res[0]=lin; res[1]=col
    return res

    
@jit(nopython=True)
def nfd_tnsr3(X, allmode, mode, XN):
    if mode==1:
        iCol = 0
        incCol=X[0,:,:].shape[1]
        for jtr in range(allmode[0]):
                XN[:,iCol:iCol+incCol] = X[jtr,:,:]
                iCol+=incCol
    if mode==2:
        iCol = 0
        incCol=X[0,:,:].shape[0]
        for jtr in range(allmode[0]):
                XN[:,iCol:iCol+incCol] = X[jtr,:,:].T
                iCol+=incCol
    if mode==3:
        iCol = 0
        incCol=X[:,0,:].shape[1]
        for jtr in range(allmode[1]):
                XN[:,iCol:iCol+incCol] = X[:,jtr,:]
                iCol+=incCol
    return XN
  
    
@jit(nopython=True)
def nfd_tnsr4(X, allmode, mode, XN):
    if mode==1:
        iCol = 0
        incCol=X[0,0,:,:].shape[1]
        for kk in range(allmode[0]):
            for jtr in range(allmode[1]): 
                XN[:,iCol:iCol+incCol] = X[kk,jtr,:,:]
                iCol+=incCol
    if mode==2:
        iCol = 0
        incCol=X[0,0,:,:].shape[0]
        for kk in range(allmode[0]):            
            for jtr in range(allmode[1]):             
                XN[:,iCol:iCol+incCol] = X[kk,jtr,:,:].T
                iCol+=incCol
    if mode==3:
        iCol = 0
        incCol=X[0,:,0,:].shape[1]
        for jtr in range(allmode[0]): 
            for kk in range(allmode[2]):
                XN[:,iCol:iCol+incCol] = X[jtr,:,kk,:]
                iCol+=incCol
    if mode==4:
        iCol = 0
        incCol=X[:,0,0,:].shape[1]
        for jtr in range(allmode[0]): 
            for kk in range(allmode[2]):
                XN[:,iCol:iCol+incCol] = X[:,jtr,kk,:]
                iCol+=incCol
    return XN
       

@jit(nopython=True)
def build_CP3(Zshape, rankZ, A, B, C):
    X_hat = np.zeros((Zshape))
    for r in range(rankZ):
        for ktr in range(Zshape[0]):
            for n in range(Zshape[1]):
                for m in range(Zshape[2]):
                    X_hat[ktr, n, m] += C[ktr,r]*A[n,r]*B[m,r]
    return X_hat
    
    
@jit(nopython=True)
def build_PRTCK2(Zshape, A, DA, R, DB, BT):
    X_hat = np.zeros((Zshape))
    for ktr in range(Zshape[0]):
        X_hat[ktr] = np.dot(np.dot(np.dot(np.dot(A, DA[ktr]), R), DB[ktr]), BT)
    return X_hat