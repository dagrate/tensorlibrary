import time
import numpy as np
import optnsr as opt
from tnsr_utils import tenseur_init

       
def tnsr_to_vect(A, mthd="F"):
    return A.flatten(order=mthd)

      
def init_parameters(I, J, K, P, Q):
    A = np.random.rand(I, P)
    DA = np.zeros((K, P, P))
    DA[:] = np.diag(np.ones(P))
    R = np.random.random([P, Q])
    DB = np.zeros((K, Q, Q))
    DB[:] = np.diag(np.ones(Q))
    B = np.random.rand(Q, J).T
    return A, DA, R, DB, B
 

def PARATUCK2_Decomposition(X, latfact, 
                            decompType = "PARATUCK2", 
                            maxiter = 20000, eps = 1.0E-4, 
                            eps_objfnct = 1.0E-8, NonNgtv="Y"):
    """
    PARATUCK2_DECOMPOSITION: Perform PARATUCK2 decomposition such that
    ``X_k approx AD_k^A R D_k^B B^T`` or DEDICOM decomposition such that
    ``X_k approx AD_k^A R D_k^B A^T``
    
    where
        A, B latent factors matrices\n
        D_k degree of participation for each latent component w.r.t. 3rd dimension\n     
        R interaction between P latent components in A and Q latent components in B\n

    Parameters
    ----------
        X (array): tensor to decompose\n
        latfact (tuple): P and Q latent factors\n
        decompType (string): "PARATUCK2" or "DEDICOM"\n
        maxiter (int): maximum numbers of iterations\n
        eps (float): convergance criteria\n
        eps_objfnct (float): convergence criteria for objective function\n
        NonNgtv (string): force non-negativity constraint\n
    
    Returns
    -------
        Return matrices A, R, B and tensors D^A, D^B
    
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
    1.0 | 14 Dec 2016 | Jeremy CHARLIER | Initial Creation\n
    2.0 | 20 Dec 2016 | Jeremy CHARLIER | Further Development\n
    3.0 | 02 Feb 2017 | Jeremy CHARLIER | Optimization
    3.1 | 03 Jun 2018 | Jeremy CHARLIER | Optimization using numba
"""
    time_start = time.clock() #-> measure calculation time
    ep_ = 1.0E-9
    errNan = False
        
    #DIMENSION CHECK IF DEDICOM CHOSEN
    if decompType=="DEDICOM":
        if X.shape[X.ndim-2]!=X.shape[X.ndim-1]:
            raise ValueError("Dimensions I and J in tensor X are not equal.")
                
    # DIMENSION DEFINITION AND INITIALIZATION
    Xshape = X.shape
    if len(Xshape)==4: 
        ValueError('Only 3-way tensor implemented. Reduce the dimensionality of the tensor')
    I = Xshape[-2]    
    J = Xshape[-1]
    K = Xshape[-3]
    P = latfact[0]
    Q = latfact[1]
    A, DA, R, DB, B = init_parameters(I, J, K, P, Q)
    
    if decompType=="DEDICOM":
        B = A.T
    
    Xhat = opt.build_PRTCK2(Xshape, A, DA, R, DB, B.T)
    print("Initialization \t\tObj. Fnct: {:.6f}".format(opt.tnsr_norm(X-Xhat)))
    
    # ALS ALGORITHM
    iteri=0  
    nb_it=maxiter
    while iteri<maxiter:      
        
        # 1 Estimation of A
        tempX = np.zeros((I, J*K))
        tempF = np.zeros((P, J*K))
        for ktr in range(K):
            tempX[:,ktr*J:(ktr+1)*J] = X[ktr]
            tempF[:,ktr*J:(ktr+1)*J] = np.dot( DA[ktr], np.dot( R, \
                np.dot( DB[ktr], B.T) ) )
        if NonNgtv=="N":
            A = np.dot(tempX, np.linalg.pinv(tempF))
        else:
            num = np.dot(tempX, tempF.T) + ep_
            denum = np.dot(A, np.dot(tempF, tempF.T)) + ep_
            A = np.multiply( A , np.divide( num, denum ) )
        
        # 2 Estimation of DA
        for ktr in range(K):
            F = np.dot(B, np.dot( DB[ktr], R.T ) )
            flin, fcol = F.shape
            alin = A.shape[0]
            Z = opt.nmb_krao(F, A, flin, fcol, alin, 
                         np.zeros((flin*alin, fcol))).T
            vecXk = tnsr_to_vect(X[ktr], "F")
            if NonNgtv=="N":
                C = np.dot(vecXk, np.linalg.pinv(Z))
                DA[ktr] = np.diag(C)
            else:
                num = np.dot(Z, vecXk) + ep_
                denum = np.dot( np.diag(DA[ktr]), np.dot ( Z, Z.T ) ) + ep_
                DA_ktr = DA[ktr]
                DA[ktr] = np.multiply(DA[ktr], np.divide(num, denum))                
                DA[ktr] = DA_ktr 
        
        # 3 Estimation of R
        vecXk = np.zeros((I*J*K))
        Z = np.zeros((I*J*K, P*Q))
        for ktr in range(K):
            strt = I*J*(ktr); nd = I*J*(ktr+1)
            vecXk[strt:nd] = tnsr_to_vect(X[ktr], "F")
            Z[strt:nd, :] = opt.kronnb(np.dot(B, DB[ktr]), np.dot(A, DA[ktr]))
        if NonNgtv == "N":
            R = np.dot(np.linalg.pinv(Z), vecXk)
        else:
            num = np.dot(Z.T, vecXk) + ep_
            fact01 = tnsr_to_vect(R, 'F')
            denum = np.dot(fact01, np.dot(Z.T, Z)) + ep_
            R = np.multiply(fact01, np.divide(num, denum))
        R = np.reshape(R, (P, Q), order="F")
                    
        # 4 Estimation of DB
        for ktr in range(K):
            F = np.dot( R.T, np.dot( DA[ktr], A.T ) )
            F = F.T
            blin, bcol = B.shape
            flin = F.shape[0]
            Z = opt.nmb_krao(B, F, blin, bcol, flin, 
                         np.zeros((blin*flin, bcol)))
            vecXk = tnsr_to_vect(X[ktr], "F")
            if NonNgtv=="N":
                C = np.dot(np.linalg.pinv(Z), vecXk)
                DB[ktr] = np.diag(C)
            else:
                num = np.dot(vecXk, Z) + ep_
                denum = np.dot( np.diag(DB[ktr]), np.dot( Z.T, Z ) ) + ep_
                DB_ktr = DB[ktr]
                DB[ktr] = np.multiply(DB[ktr], np.divide(num, denum))                
                DB[ktr] = DB_ktr
        
        # 5 Estimation of B
        if decompType=="DEDICOM":
            B = A.T
        else:
            tempX = np.zeros((J, I*K))
            tempF = np.zeros((Q, I*K))
            for ktr in range(K):
                tempX[:,ktr*I:(ktr+1)*I] = X[ktr].T
                tempF[:,ktr*I:(ktr+1)*I] = np.dot( A, np.dot( DA[ktr], \
                    np.dot( R, DB[ktr] ) ) ).T
            if NonNgtv=="N":
                 B = np.dot(np.linalg.pinv(tempF), tempX).T
            else:
                num = np.dot(tempX, tempF.T) + ep_
                denum = np.dot(B, np.dot(tempF, tempF.T)) + ep_
                B = np.multiply(B, np.divide(num, denum))       
        
        # convergence criteria
        Xhat = opt.build_PRTCK2(Xshape, A, DA, R, DB, B.T)
        obj_fnct = opt.tnsr_norm(X-Xhat)
                
        # Evolution of the calculation
        if iteri%(maxiter/10) == 0: 
            print("{:.2f}".format(iteri/maxiter*100), " % \t" \
                  "\tObj. Fnct: {:.6f}".format(obj_fnct) )
            
        # loop increment 
        if ( obj_fnct > eps_objfnct ): # or (speed_norm>eps)
            iteri += 1
        elif np.any(np.isnan(Xhat)) == True:
            nb_it = iteri
            iteri = maxiter
            errNan = True
        else:
            nb_it = iteri
            iteri = maxiter
    
    # Construct Approximate Tensor
    if errNan:
        print("Error: NaN in the decomposition!")
        print("Computation time (h): ", (time.clock() - time_start)/3600)
        print("Number of iterations: ", min(nb_it,iteri))
    else:
        print("Computation time (h): ", 
              "{:.4f}".format((time.clock() - time_start)/3600))
        print("Number of iterations: ", min(nb_it,iteri))
        Xhat = opt.build_PRTCK2(Xshape, A, DA, R, DB, B.T)
        obj_fnct = opt.tnsr_norm(X-Xhat)
        print("Norm difference: {:.6f}".format(obj_fnct))
    
    return A, DA, R, DB, B

    
def main():
    tnsr_size = (4, 5, 6)
    X = tenseur_init((tnsr_size))
    latfact = (5,7)
    maxiter = 1000
    A, DA, R, DB, B = \
        PARATUCK2_Decomposition(X, latfact, 
                                decompType="PARATUCK2",  
                                maxiter=maxiter, eps=1.0E-8, 
                                eps_objfnct = 1.0E-3, NonNgtv="Y")
                    
    print("\nX = ", X)
    print("\nX_hat = ", opt.build_PRTCK2(tnsr_size, A, DA, R, DB, B.T))
    
if __name__ == "__main__":
    main()
