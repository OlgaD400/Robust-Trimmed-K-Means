import numpy as np
from scipy.cluster.vq import vq


def KMOR(Data, k, gamma, n0, delta=1e-6, Nmax=100, **kwargs):
    N,m = Data.shape
    
    Z = kwargs.pop('Z', Data[np.random.choice(N,k),:])
                   
    #init_center_ind = np.random.randint(0,N,k)
    #Z = Data[init_center_ind,:]
    
    #For test purposes, let Z = [0,1] and [2,3]
    #Z = np.array([[0,1], [2,3]])
    U, dist = vq(Data, Z)

    s=0
    pold=0
    err = 1 + delta
    new_out = []
    inliers = np.arange(N)

    Umat = np.zeros((N, k+1))        
    Umat[np.arange(N), U] = 1
        
    while err>=delta:  
        
        U, dist = vq(Data, Z)
        
        dist_points = np.linalg.norm(Data[inliers,:] - Z[U[inliers]],2, axis = 1)**2
        
        D = gamma*np.sum(dist_points)/(N - len(new_out))
        
        #Update Outliers based on D
        if n0>0:
            n0largest = np.argsort(dist)[-int(n0):]
        else: 
            n0largest = []
        beyond_outlier_threshold= np.where(dist**2>D)[0]
        new_out = np.intersect1d(n0largest, beyond_outlier_threshold)
        inliers = np.setdiff1d(np.arange(N), new_out)
        
        if len(new_out>0):
            U[new_out]= k
            
        Umat = np.zeros((N, k+1))
        Umat[np.arange(N), U] = 1
        
        #Choose random center if no points assigned to that center
        denom = np.sum(Umat[inliers,:k],axis = 0)
        badInd = np.where(denom == 0)
        Z = (Umat[inliers,:k]).T@Data[inliers,:]/denom[:, np.newaxis]
        Z[badInd, :] = np.random.rand(m)
        s+=1
        
        
        inlierSum = np.sum(np.linalg.norm(Data[inliers,:] - Z[U[inliers]],2, axis = 1)**2)        
        secondSum = len(new_out)*gamma*inlierSum/(N - len(new_out))
        pnew = inlierSum + secondSum
        
        err = abs(pnew - pold)
        
        pold = pnew
    
        
        if s>=Nmax:
            #U[new_out] = k
            return U, Z, pnew
    if len(new_out)>0:
        U[new_out] = k
    return U,Z, pnew
            
        