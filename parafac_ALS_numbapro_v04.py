import time
import numpy as np
import optnsr as opt
from tnsr_utils import tenseur_init

                
def parafac(X, rank_R=5, maxiter=1000, eps_objfnct=1.0E-5, 
                  nnALS="Y", Normalization="N"):    
    """
    PARAFAC decomposition (also known as CP decomposition): 
        Computes CANDECOMP/PARAFAC Decomposition using ALS

    Parameters
    ----------
        X:          (array) tensor\n    
        rank_R:     (int) tensor rank\n    
        maxiter:    (int) maximum number of iterations\n
        eps:        (double) convergence criteria\n
        algorithm:  (string) "Default" for standard ALS, "Optim" for optimized ALS\n
        nnALS:       (string) "Y" for non negative PARAFAC decomposition, otherwise "N"\n
        Normalization (string) "Y" to compute a_r, b_r, c_r as unit vectors

    Returns
    -------
    CANDECOMP/PARAFAC Decomposition
    
    References
    ----------
        .. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and 
           Applications", 2009, DOI 10.1137/0707111X
        .. [2] Tamara G. Kolda, "Multilinear Operators for High-Order Decompositions",
           April 2006, SAND2006-2081
           
    Version
    ----------
        1.0 | 08 Dec 2016 | Jeremy CHARLIER | Initial Creation\n
        2.0 | 14 Dec 2016 | Jeremy CHARLIER | Optimized algorithm\n 
        3.0 | 06 Jan 2017 | Jeremy CHARLIER | Non Negative PARAFAC decomposition\n
        4.0 | 23 Jan 2017 | Jeremy CHARLIER | Normalization\n
        5.0 | 03 May 2017 | Jeremy CHARLIER | Mode Optimization\n
        6.0 | 15 May 2017 | Jeremy CHARLIER | Code Optimization\n
        7.0 | 04 Jul 2017 | Jeremy CHARLIER | Convergence Optimization\n
        7.1 | 06 Jul 2018 | Jeremy CHARLIER | Krao Product Update\n
    """
    
    # 0 CALCULATION TIME
    time_start = time.clock()
    errNan = False
    
    # 1 VARIABLES INITIALIZATION
    A = []    
    for itr in range(X.ndim):
        matrix1 = np.random.random([X.shape[itr], rank_R])*100
        A.append(matrix1)

    lmbda = np.ones((rank_R))
    iteri = 0
    nb_it = maxiter
    allMode = np.arange(X.ndim)
    allmode = np.asarray(X.shape)
    totMode = np.arange(2)
    for n in range(2,X.ndim):
        totMode = np.insert(totMode, 0, n)

    # convergence criteria initialization
    Xshape = X.shape
    Xhat = opt.build_CP3(Xshape, rank_R, A[1], A[2], A[0])
    print("Initialization\t\tObj. Fnct: {:.6f}".format(opt.tnsr_norm(X-Xhat)))
    
    curr_norm=0
    for n in range(len(A)):
        curr_norm += opt.m_norm(A[n])
    
    # 2 ALS ALGORITHM
    #Optimized CP ALS presented by Kolda
    while iteri<maxiter:
        for mode in range(X.ndim):
            curAllMode = np.zeros((len(allMode)-1), dtype='int32')
            ii=0
            for n in range(len(allMode)):
                if allMode[n] != mode:
                    curAllMode[ii] = n
                    ii+=1
                
            icnt = 0
            for n in range(X.ndim-2,-1,-1):
                currMode = curAllMode[n]
                if icnt == 0:
                    W = A[currMode]
                else:
                    Xkr = opt.kraonb(A[currMode], W)
                    W = Xkr
                icnt = 1    
                            
            # computation of A[mode]
            currmode = totMode[mode] + 1
            Ydim = opt.nfd_size(allmode, currmode)
            XN = np.zeros((int(Ydim[0]),int(Ydim[1])))
            if X.ndim == 3:
                XN = opt.nfd_tnsr3(X, allmode, currmode, XN)
            if X.ndim == 4:
                XN = opt.nfd_tnsr4(X, allmode, currmode, XN)
            
            if nnALS == "Y":
                A_mode = A[mode]
                num = np.dot(XN, W) + 1.0E-9
                denum = np.dot(A_mode, np.dot(W.T, W)) + 1.0E-9
                A_mode = np.multiply( A_mode, np.divide(num, denum) )
                A[mode] = A_mode
            else:
                raise ValueError("Only nn-ALS resolution implemented for PARAFAC")
                iteri=maxiter
                
                #if len(X.shape) >= 4:
                #    ValueError("Only 3-way tensor implemented for ALS resolution")
                #else:
                #    A_mode = A[mode]
                #    ma = opt.kraonb(A[curAllMode[::-1][0]], A[curAllMode[::-1][1]])
                #    mba = np.dot( A[curAllMode[::-1][0]].T, A[curAllMode[::-1][0]] )
                #    mbb = np.dot( A[curAllMode[::-1][1]].T, A[curAllMode[::-1][1]] )
                #    mb = np.multiply(mba, mbb)
                #    A[mode] = np.dot(XN, np.dot(ma, np.linalg.pinv(mb)))

        # convergence criteria
        Xhat = opt.build_CP3(Xshape, rank_R, A[1], A[2], A[0])
        obj_fnct = opt.tnsr_norm(X-Xhat)
                
        # Evolution of the calculation
        if iteri%(maxiter/10) == 0: 
            print("{:.2f}".format(iteri/maxiter*100), " % \t" \
                  "\tObj. Fnct: {:.6f}".format(obj_fnct) )
            
        # loop increment 
        if (obj_fnct>eps_objfnct): # or (speed_norm>eps)
            iteri += 1
        elif np.any(np.isnan(Xhat)) == True:
            nb_it=iteri
            iteri=maxiter
            errNan = True
        else:
            nb_it=iteri
            iteri=maxiter
    
    #Scalar weight of the r^th component
    if Normalization=="Y":
        for itr in range(rank_R):
            for jtr in range(X.ndim):
                lmbda[itr] *= np.linalg.norm(A[jtr][:,itr])
                A[jtr][:,itr] = A[jtr][:,itr] / np.linalg.norm(A[jtr][:,itr])
                
   # Construct Approximate Tensor
    if errNan:
        print("Error: NaN in the decomposition!")
        print("Computation time (h): ", (time.clock() - time_start) / 3600)
        print("Number of iterations: ", min(nb_it,iteri))
    else:
        print("Computation time (h): ", 
              "{:.4f}".format((time.clock() - time_start) / 3600))
        print("Number of iterations: ", min(nb_it,iteri))
        Xhat = opt.build_CP3(Xshape, rank_R, A[1], A[2], A[0])
        obj_fnct = opt.tnsr_norm(X - Xhat)
        print("Norm difference: {:.6f}".format(obj_fnct))
        
    return A, lmbda


def main():
    tnsr_size = (10, 10, 10)
    X = tenseur_init((tnsr_size))
    
    R = 5
    maxiter = 1000
    A, lmbda = parafac(X, R, maxiter, eps_objfnct=1.0E-3, 
                             nnALS="N", Normalization="N")
                    
    print("\nX = ", X)
    print("\nX_hat = ", opt.build_CP3(X.shape, R, A[1], A[2], A[0]))
    
if __name__ == "__main__":
    main()