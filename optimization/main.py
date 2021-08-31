# -*- coding: UTF-8 -*- 
import numpy as np
import time
import os,sys
from threading import Thread
import config
import spatial_filter as sf
import read_write as rw
import baseline_removal as baseline


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
if __name__=='__main__':
    time_start=time.time()
    t_sample = config.t_sample
    f_sample = config.f_sample
    t_shape = config.t_shape
    f_shape = config.f_shape
    beam = config.beam
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
    #print(fileList)

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
        ### data load
        start_load = time.time()
        data,filename = rw.read_fits(path)
        end_load = time.time()
        print('data load cost:'+str(end_load-start_load))
        if config.debug:
            print('data shape is :',data.shape)

        ### baseline removal
        start_baseline = time.time()
        data = baseline.baseline_removal(data)
        end_baseline = time.time()
        print('baseline removal cost:'+str(end_baseline-start_baseline))

        ### spatial filter
        D_ms,correlation =  sf.make_covariance_matrix(data)
        d_clean = sf.make_matrix(D_ms,correlation)
        end_filter = time.time()
        print('filter cost:'+str(end_filter-end_baseline))
        matrix1 = data.reshape(beam,t_shape,f_shape,-1).mean(axis=-1)
        print('d_clean shape is : ', d_clean.shape)
        del data
        print('matrix1 shape is :', matrix1.shape)

        ### flagging RFI and generating mask files
        threads = []
        for i in range(beam):
            threads.append(
                Thread(target=rw.out, args=(matrix1[i],d_clean[i],filename[i],))
                )
        
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
    write_stop = time.time()
    print('write mask files cost:'+str(write_stop-end_filter))       

    time_end=time.time()
    print('totally cost',time_end-time_start)

