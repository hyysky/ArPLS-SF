#projection interference for 1/4 fits file and PCP. using matrix instead of loop ,t_sample,f_sample
import numpy as np
from torch._C import device
import config
import torch
import sys

t_sample = config.t_sample
f_sample = config.f_sample
beam = config.beam
t_shape = config.t_shape
f_shape = config.f_shape

def make_covariance_matrix(data):
    t_step = int(t_shape/t_sample)
    f_step = int(f_shape/f_sample)    
    D=np.zeros((beam,beam,t_shape,f_shape))
    for i in range(beam):
        for j in range(beam):
            a = np.sqrt(data[i]*data[j])
            D[i][j] = a.reshape(t_shape,f_shape)
    D_ms = np.transpose(D,(2,3,0,1))
    D = D.reshape(beam,beam,t_sample,t_step,f_shape).mean(axis=3)
    D = D.reshape(beam,beam,t_sample,f_shape,f_step).mean(axis=-1).squeeze()
    D = np.transpose(D,(2,3,0,1))
    print('correlation shape:',D.shape)
    return D_ms,D


def make_matrix(D_ms,correlation):
    t_step = int(t_shape/t_sample)
    f_step = int(f_shape/f_sample)
    D=D_ms
    data_clean = np.zeros((beam,t_shape,f_shape))
    cuda0 = torch.device('cuda:0')
    correlation_cpu = torch.tensor(correlation.reshape(-1,beam,beam))
    u_cpu, s_cpu, vh_cpu = torch.svd(correlation_cpu)
    u_cpu = u_cpu.reshape(t_sample,f_sample,beam,beam)
    #correlation_gpu = torch.tensor(correlation.reshape(-1,beam,beam),device=cuda0)
    #u_gpu, s_gpu, vh_gpu = torch.svd(correlation_gpu)
    #u_gpu = u_gpu.reshape(t_sample,f_sample,beam,beam)
    #u_cpu = u_gpu.cpu()
    #torch.cuda.empty_cache()
    u = np.array(u_cpu)
    spectrum = np.zeros((t_shape,f_shape,beam))
    for i in range(t_shape):
        for j in range(f_shape): 
            u_sample = u[int(i/t_step)][int(j/f_step)].squeeze()
            u_rfi = u_sample[:,:1]
            P = np.dot(u_rfi,u_rfi.T)
            c=D[i][j]
            matrix_clean = c - np.dot(P,np.dot(c,P))
            spectrum[i][j]=np.diag(matrix_clean)
    spectrum = np.transpose(spectrum,(2,0,1))
    for k in range(beam):
        data_clean[k]=spectrum[k]
    data_clean[data_clean<0]=0
    return data_clean          

