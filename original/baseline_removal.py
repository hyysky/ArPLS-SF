import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse




def ArPLS(y, lam=1e4, ratio=0.05, itermax=10):
    '''
    copy from https://irfpy.irf.se/projects/ica/_modules/irfpy/ica/baseline.html
    
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Inputs:
        y:
            input data (i.e. SED curve)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
    '''

    N = len(y)
    #  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    D = D.T
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
        
    return z

def baseline_removal(data):
    l,m,n = data.shape
    for i in range(l):
        for j in range(m):
            data[i,j,:] -= ArPLS(data[i,j,:])
    data[data<0]=0
    return data