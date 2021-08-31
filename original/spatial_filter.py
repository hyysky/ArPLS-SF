#projection interference for fits file and PCP. using matrix instead of loop ,t_sample,f_sample
import numpy as np
import config
#import skcuda.linalg as linalg


t_sample = config.t_sample
f_sample = config.f_sample
beam = config.beam
t_shape = config.t_shape
f_shape = config.f_shape


def make_covariance_matrix(data):
    step = int(t_shape/t_sample)
    f_step = int(f_shape/f_sample)
    D = np.zeros((beam,beam,t_shape,f_shape))
    for i in range(beam):
        for j in range(beam):
            a = np.sqrt(data[i]*data[j])
            D[i][j] = a.reshape(t_shape,f_shape)    
    D_ms = np.transpose(D,(2,3,0,1))
    D = D.reshape(beam,beam,t_sample,step,f_shape).mean(axis=3)
    D = D.reshape(beam,beam,t_sample,f_shape,f_step).mean(axis=-1).squeeze()
    D = np.transpose(D,(2,3,0,1))
    print('covariance matrix shape:',D.shape)
    return D_ms,D


#@jit(nopython=True)
def make_matrix(D_ms,covariance):
    data_clean = np.zeros((beam,t_shape,f_shape))
    t_step = int(t_shape/t_sample)
    f_step = int(f_shape/f_sample)
    D = D_ms
    for i in range(t_shape):
        for j in range(f_shape):
            U,S,V = np.linalg.svd(covariance[int(i/t_step)][int(j/f_step)])
            P = np.dot(U[:,:1],U[:,:1].T)
            c = D[i][j]
            matrix_clean = c - np.dot(P,np.dot(c,P))
            spectrum = np.diag(matrix_clean)
            for k in range(beam):
                data_clean[k][i][j]=spectrum[k]
    data_clean[data_clean<0]=0
    return data_clean          
 
