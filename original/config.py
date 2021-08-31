import numpy as np

debug = False
t_sample = int(32)   #time size
f_sample = int(4096)   #freq size
subint= int(256)          # subint number
beam = 19                #beam number to process
path2save = 'D:\\pynote\\spatial_filter_original\\result\\'
pol_num = 0               #0,1,2,3 which pol to process
t_shape = int(256)  
f_shape = int(4096)
factor = 1.5             #setting threshold 1,2,3 sigma
