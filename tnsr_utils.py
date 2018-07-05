"""
    Description
    ----------
    Supporting functions to be used by the tensor resolution files. 
    
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
    1.0 | 01 Dec 2016 | Jeremy CHARLIER | Initial Creation\n
"""

import numpy as np


def kr_size(C_dim, D_dim):
    """Determine shape of the Khatri-Rao product
    """
    res=[]
    res.append(C_dim[0]*D_dim[0])
    res.append(C_dim[1])
    return res


def tenseur_init(tnsr_size):
    """
    Computes a standard tensor for backtesting operations
    """
    if len(tnsr_size)==3:
        curr=0
        X = np.zeros((tnsr_size))
        for itr in range(tnsr_size[-3]):
            for jtr in range(tnsr_size[-2]):
                for ktr in range(tnsr_size[-1]):
                    curr+=1
                    X[itr,jtr,ktr] = curr
    elif len(tnsr_size)==4:
        curr=0
        X = np.zeros((tnsr_size))
        for iM in range(tnsr_size[-4]):
            for itr in range(tnsr_size[-3]):
                for jtr in range(tnsr_size[-2]):
                    for ktr in range(tnsr_size[-1]):
                        curr+=1
                        X[iM,itr,jtr,ktr] = curr
    return X
    
    
def tnsr_compute(X, lmbda, A, R):
    """
    Computes the third/fourth order tensor from CANDECOMP/PARAFAC ALS Decomposition 

    @author: jeremy.charlier
    @date: 06-Dec-16

    Parameters
    ----------
    X:          (tensor) original tensor
    lmbda:      (int) normalization factor
    A:          (array) matrices of size R \times I_n, R \times I_n, ...
    R:          (int) tensor rank
    
    Returns
    -------
    Third order tensor of size len(C), len(A), len(B)
    
    References
    ----------
    .. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and 
           Applications", 2009, DOI 10.1137/0707111X
    """
    
    Xdim=X.ndim
    if Xdim==3:
        M=A[Xdim-2]; N=A[Xdim-1]; O=A[Xdim-3]
        nRow, nCol, nDepth = len(M), len(N), len(O)
        X_hat = np.zeros((nDepth, nRow, nCol))
        D = np.zeros((len(N), R))
        for ztr in range(R):
            D[:,ztr] = N[:, ztr]*lmbda[ztr]
            X_hat+=np.multiply.outer(O[:,ztr],np.outer(M[:,ztr],D[:,ztr]))
        
    if Xdim==4:
        M=A[Xdim-2]; N=A[Xdim-1]; O=A[Xdim-3]; P=A[Xdim-4]
        n1=len(M); n2=len(N); n3=len(O); n4=len(P)
        X_hat = np.zeros((n4,n3,n1,n2))
        D = np.zeros((len(N), R))
        for ztr in range(R):
            D[:,ztr] = N[:, ztr]*lmbda[ztr]
            outpd1=np.outer(M[:,ztr],D[:,ztr])
            outpd2=np.multiply.outer(O[:,ztr],outpd1)
            X_hat+=np.multiply.outer(P[:,ztr],outpd2)
    return X_hat
    
    
def tnsr_norm(X):
    """
    Computes tensor norm
    
    Parameters
    ----------
    X:          (array) tensor
    
    Returns
    -------
    Norm of the tensor X (scalar)
    """
    return np.sqrt(np.sum(X**2))
    

def tnsr_unfolding(X, modeN):
    """
    Computes tensor unfolding for N-dimensional tensor according to a specified 
    mode. Particularity of numpy is that for a tensor the index are classified 
    as I_N, I_(N-1), ..., I_3, I_1, I_2

    @author: jeremy.charlier
    @date: 07-Dec-16
    @update: 10-Feb-17 Manage tensor up to 20th order
        
    Parameters
    ----------
    X:          (array) tensor
    modeN:      (int) unfolding mode

    Returns
    -------
    XN:         (array) unfolding matrix according to mode N
    
    References
    ----------
    .. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and 
           Applications"
   """
   
    modeDesc = {'1':X.ndim-2, '2':X.ndim-1, '3':X.ndim-3, '4':X.ndim-4, 
                '5':X.ndim-5, '6':X.ndim-6, '7':X.ndim-7, '8':X.ndim-8, 
                '9':X.ndim-9, '10':X.ndim-10, '11':X.ndim-11, '12':X.ndim-12, 
                '13':X.ndim-13, '14':X.ndim-14, '15':X.ndim-15, '16':X.ndim-16, 
                '17':X.ndim-17, '18':X.ndim-18, '19':X.ndim-19, '20':X.ndim-20}
                
    col = 1
    for mode in range(X.ndim):
        if mode+1!=modeN:
            col *= X.shape[modeDesc[str(mode+1)]]
        else:
            lin = X.shape[modeDesc[str(mode+1)]]

    if X.ndim==3:
        if modeN!=X.ndim:
            kind = 3-modeN
            XN = np.moveaxis(X,X.ndim-kind,0).reshape([X.shape[X.ndim-kind],col], order='C')
        else:
            XN = np.reshape(X, (X.shape[0], col), order='C')
    if X.ndim==4:
        XN=np.zeros((lin,col))
        if modeN==1:
            iCol = 0    
            for jtr in range(X.shape[modeDesc[str(4)]]):
                for itr in range(X.shape[modeDesc[str(3)]]):
                    for ktr in range(X.shape[modeDesc[str(2)]]):
                        XN[:,iCol] = X[jtr,itr,:,ktr]
                        iCol+=1
        if modeN==2:
            iCol = 0    
            for jtr in range(X.shape[modeDesc[str(4)]]):
                for itr in range(X.shape[modeDesc[str(3)]]):
                    for ktr in range(X.shape[modeDesc[str(1)]]):
                        XN[:,iCol] = X[jtr,itr,ktr,:]
                        iCol+=1
        if modeN==3:
            iCol = 0    
            for jtr in range(X.shape[modeDesc[str(4)]]):
                for ktr in range(X.shape[modeDesc[str(1)]]):
                    for itr in range(X.shape[modeDesc[str(2)]]):
                        XN[:,iCol] = X[jtr,:,ktr,itr]
                        iCol+=1
        if modeN==4:
            iCol = 0    
            for jtr in range(X.shape[modeDesc[str(3)]]):
                for ktr in range(X.shape[modeDesc[str(1)]]):
                    for itr in range(X.shape[modeDesc[str(2)]]):
                        XN[:,iCol] = X[:,jtr,ktr,itr]
                        iCol+=1
    if X.ndim==5:
        ValueError("Tensor unfolding for tensor of order 5 not implemented")
        
    return XN
    
    
def tnsr_n_mode_product(X, U, mode):
    """    
    TNSR_N_MODE_PRODUCT: Computes n-mode product

    @author: jeremy.charlier
    @date: 09-Dec-16 

    Parameters
    ----------
    A:          (array) tensor
    U:          (array) matrix
    mode:       (int) n-mode multiplication

    Returns
    -------
    tensor n-mode product
    
    References
    ----------
    .. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and 
           Applications", 2009, DOI 10.1137/0707111X
       [2] Tamara G. Kolda, "Multilinear Operators for High-Order Decompositions",
           April 2006, SAND2006-2081
    """

    modeDesc = {'1':X.ndim-2, '2':X.ndim-1, '3':X.ndim-3, '4':X.ndim-4, '5':X.ndim-5, 
    '6':X.ndim-6, '7':X.ndim-7, '8':X.ndim-8, '9':X.ndim-9, '10':X.ndim-10, 
    '11':X.ndim-11, '12':X.ndim-12, '13':X.ndim-13, '14':X.ndim-14, '15':X.ndim-15, 
    '16':X.ndim-16, '17':X.ndim-17, '18':X.ndim-18, '19':X.ndim-19, '20':X.ndim-20}
    
    # tensor n-mode product size
    tmp = []
    for itr in range(X.ndim):
        if (itr+1)!=mode:
            tmp.append(X.shape[modeDesc[str(itr+1)]])
        else:
            tmp.append(U.shape[0])
    
    #reordering for Python 
    YDim = []
    for itr in range(X.ndim-3,-1,-1):
        YDim.append(tmp[2+itr])
    YDim.append(tmp[0])            
    YDim.append(tmp[1])
    
    #tensor n-mode product
    Y = np.dot(U, tnsr_unfolding(X, mode))
    
    # store the result in the tensor as in Matlab/Octave
    Y_New = np.zeros((YDim))
    iRow, iCol = -1, 0
    if X.ndim == 3:
        for itr in range(YDim[0]):
            for jtr in range(YDim[2]):
                for ztr in range(YDim[1]):
                    iRow +=1
                    if iRow>=Y.shape[0]:
                        iRow = 0
                        iCol +=1
                    Y_New[itr, ztr, jtr] = Y[iRow, iCol]
    if X.ndim == 4:
        for ytr in range(YDim[X.ndim-4]):
            for itr in range(YDim[X.ndim-3]):
                for jtr in range(YDim[X.ndim-1]):
                    for ztr in range(YDim[X.ndim-2]):
                        iRow +=1
                        if iRow>=Y.shape[0]:
                            iRow = 0
                            iCol +=1
                        Y_New[ytr, itr, ztr, jtr] = Y[iRow, iCol]
    return Y_New