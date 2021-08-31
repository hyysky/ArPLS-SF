# -*- coding: UTF-8 -*- 
import numpy as np
import time
import spatial_filter as sf
import baseline_removal as baseline
import read_write as rw
import os,sys
import config



def t_split(name):
        filename, extension = os.path.splitext(name)
        num = filename.split('_')[-1]
        return num

def text_split(name):
	filename, extension = os.path.splitext(name)
	num = filename.split('_')[-1]
	value = 1
	#if int(num)>244 and int(num)<312:
	number.append(num)
	value = 0
	return value

### main code ###
time_start=time.time()
t_sample = config.t_sample
f_sample = config.f_sample
beam = config.beam
t_shape = config.t_shape
f_shape = config.f_shape
path = sys.argv[1] 

fileList = os.listdir(path)
number = []
d = 0
for i in range(len(fileList)):
    value = text_split(fileList[i+d])
    if value==1:
        del fileList[i+d]
        print(i,d)
        d = d-1
number_key = set(number)
print(number_key)

if not os.path.exists('result/'):
    os.mkdir("result/")

for key in number_key:
    filenames = []
    delta = 0
    for i in range(len(fileList)):
        num = t_split(fileList[i+delta])
        if num == key:
            filenames.append(fileList[i+delta])
            del fileList[i+delta]
            delta = delta -1		
    load_start = time.time() 
    data,filename = rw.read_fits(path)
    load_stop = time.time()
    print('loading data cost:'+str(load_stop-load_start))
    if config.debug:
        print('data shape is :',data.shape)
    data = baseline.baseline_removal(data)
    baseline_stop = time.time()
    print('baseline removal cost:'+str(baseline_stop-load_stop))
    D_ms,covariance =  sf.make_covariance_matrix(data)
    d_clean = sf.make_matrix(D_ms,covariance)
    time_clean_end = time.time()
    print('spatial filter cost:',str(time_clean_end-baseline_stop))
    matrix1 = data.reshape(beam,t_shape,f_shape,-1).mean(axis=-1)
    print('d_clean shape is : ', d_clean.shape)
    del data
    print('matrix1 shape is :', matrix1.shape)

    for i in range(beam):
        rw.out(matrix1[i], d_clean[i], filename[i])
    write_stop = time.time()
    print('write mask files cost:'+str(write_stop-time_clean_end))

time_end=time.time()
print('totally cost:',time_end-time_start)
