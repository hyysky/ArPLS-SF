import astropy.io.fits as pyfits
from astropy.io.fits import file
import numpy as np
import matplotlib.pyplot as plt
import config
import os,sys,glob
from threading import Thread
from decimal import Decimal
from copy import copy

beam = config.beam
t_shape = config.t_shape
f_shape = config.f_shape


### thread class to get the results from different threads
class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def readfits(file):
    hdulist = pyfits.open(file)
    data = hdulist[1].data['data'].squeeze()
    data = data[:,:,config.pol_num,:].squeeze()
    a,b,c=data.shape
    data = data.reshape(config.t_shape,-1,config.f_shape).mean(axis=1)
    return data


def do_rms(data,t_sample,f_sample):
    matrix = np.zeros((t_sample//3,40))
    f_idx = int(f_sample*(1330-1000)/(500))
    matrix=data[t_sample//3:t_sample//3+t_sample//3, f_idx:f_idx+40]
    matrix = matrix.reshape(-1)
    std = np.std(matrix,ddof=1)
    return std


def plot_rawdata(data,types):
    l,m,n = data.shape
    for i in range(l):
        fig=plt.figure()
        plt.imshow(data[i], 
                aspect='auto', 
                rasterized=True, 
                interpolation='nearest', 
                cmap='hot',
                extent=(0,n,0,m))
        figure_name = config.path2save + 'M' + str(i+1) + '_' + types
        plt.xlabel('Channel')
        plt.colorbar()
        plt.ylabel('Interval Number')
        plt.savefig(figure_name,dpi=400)
        plt.close()


def plot_mask2(mask,i):
    l,m=mask.shape
    #mask = 1-mask
    fig=plt.figure()
    plt.imshow(mask,
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            cmap='hot',extent=(0,m,0,l),
        )
    figure_name = config.path2save + 'M' + str(i+1)+ '_residual' + '.png'
    plt.xlabel("Channel")
    plt.ylabel("Interval Number")
    plt.title('Spatial Filter')
    plt.colorbar()
    plt.savefig(figure_name,dpi=400)
    plt.close()


def plot_mask(mask,filename):
    palette=copy(plt.cm.hot)
    palette.set_bad('cyan', 1.0)
    basename = filename[filename.find('J0528'):filename.find('.fits')]
    l,m=mask.shape
    plt.imshow(mask,
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            cmap='hot',extent=(0,m,0,l),
        )
    figure_name = config.path2save + basename + '_mask' + '.png'
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Subint Number")
    plt.savefig(figure_name,dpi=400)
    plt.close()


### generate the mask data ###
def find_mask(residual,t_sample,f_sample,factor):
    a = residual.reshape(-1)
    threshold = np.mean(a)+ factor*np.std(a, ddof = 1)
    for i in range(t_sample):
        for j in range(f_sample):
            if residual[i][j] < threshold:
                residual[i][j] = 0    
            else:
                residual[i][j] = 1    #RFI pixel
    for i in range(t_sample):
        if residual[i,:].sum()>f_sample*0.5:
            residual[i,:]= 1
    for j in range(f_sample):
        if residual[:,j].sum()>t_sample*0.5:
            residual[:,j] = 1
    return residual.astype(np.uint8)


''' generate the mask file '''
def write_mask(filename,mask):
    basename = filename[filename.find('J0528'):filename.find('.fits')]
    hdu = pyfits.open(filename)
    time_sig=np.float64(10.0)
    freq_sig=np.float64(4.0)
    tsamp = hdu[1].header['TBIN']
    secperday = 3600 * 24
    samppersubint = int(hdu[1].header['NSBLK'])
    subintoffset = hdu[1].header['NSUBOFFS']
    MJD = "%.11f" % (Decimal(hdu[0].header['STT_IMJD']) + Decimal(hdu[0].header['STT_SMJD'] + tsamp * samppersubint * subintoffset )/secperday )
    MJD = np.float64(MJD)
    lofreq = hdu[0].header['obsfreq'] - hdu[0].header['obsbw']/2
    lofreq = np.float64(lofreq)
    df = np.float64(hdu[1].header['chan_bw'])
    nchan = np.int32(hdu[1].header['nchan'])
    nint = np.int32(config.t_sample)
    ptsperint = config.t_shape * config.subint / config.t_sample
    ptsperint = np.int32(ptsperint)
    dtint = np.float64(ptsperint * tsamp)
    nzap_f=0
    mask_zap_chans=[]
    for i in range(nchan):
        if mask[:,i].sum(axis=0)==nint:
            nzap_f = nzap_f+1
            mask_zap_chans.append(i)    
    nzap_f=np.int32(nzap_f)
    mask_zap_chans = np.array(mask_zap_chans).astype(np.int32)
    nzap_t = 0
    mask_zap_ints = []
    for i in range(nint):
        if mask[i,:].sum(axis=-1)==nchan:
            nzap_t = nzap_t+1
            mask_zap_ints.append(i)
    nzap_t=np.int32(nzap_t)
    mask_zap_ints = np.array(mask_zap_ints).astype(np.int32)
    nzap_per_int = []
    tozap = []
    for i in range(nint):
        spec = mask[i]
        nzap = spec.sum()
        index_rfi = np.where(spec==1)
        nzap_per_int.append(nzap)
        tozap.append(index_rfi)
    nzap_per_int = np.array(nzap_per_int).astype(np.int32)
    maskfile = config.path2save + basename +'.mask'
    f = open(maskfile,'wb+')
    f.write(time_sig)
    f.write(freq_sig)
    f.write(MJD)
    f.write(dtint)
    f.write(lofreq)
    f.write(df)
    f.write(nchan)
    f.write(nint)
    f.write(ptsperint)
    f.write(nzap_f)
    f.write(mask_zap_chans)
    f.write(nzap_t)
    f.write(mask_zap_ints)
    f.write(nzap_per_int)
    for i in range(len(tozap)):
        f.write(np.array(tozap[i]).astype(np.int32))
    f.close()


def out(matrix1, d_clean, filename):
    std1 = do_rms(matrix1,config.t_shape,config.f_shape)
    std2 = do_rms(d_clean,config.t_shape,config.f_shape)
    matrix3 = d_clean*std1/std2
    residual = matrix1 - matrix3
    #plot_mask2(residual,i)
    mask = find_mask(residual,t_shape,f_shape,config.factor)
    mask = mask.astype(bool)
    insert_step=int(config.t_shape/config.subint)
    mask2 = np.zeros((int(config.subint),config.f_shape),dtype=bool)
    for j in range(len(mask2)):
        mask2[j] = mask[j//insert_step]
    filepath = filename
    filepath = filepath.replace('\\J0528+2200\\','\\data\\')
    #data = readfits(filepath)
    #data = np.ma.array(data,mask=mask2)
    #plot_mask(data,filename)
    mask = mask.astype('int32')
    write_mask(filepath,mask)
  


def read_fit(filename):
    if os.path.splitext(filename)[-1]=='.fits':
        hdulist = pyfits.open(filename)
        hdu1 = hdulist[1]
        data1 = hdu1.data['data']
        data1 = data1.squeeze()
        a,b,c,d = data1.shape
        if (a != config.subint) | (d != config.f_shape) :
            print('data shape is wrong!')
        pol0_data = data1[:,:,int(config.pol_num),:].squeeze()
        l,m,n = pol0_data.shape
        if config.debug:
            print('pol0_data shape is',pol0_data.shape)
        t_step = int(l*m/config.t_shape) #time
        f_step = int(n/config.f_shape)
        p0_data = pol0_data.reshape(config.t_shape,t_step,n).mean(axis=1)
        p0_data = p0_data.reshape(config.t_shape,config.f_shape,f_step).mean(axis=-1).squeeze()
    return p0_data


### multi threads read_fits code
def read_fits(path):
    fileList = sorted(glob.glob(path+'*.fits')) #read 19 beam fits data into data(19,T_sample,F_sample)
    data=[]
    f_name=[]
    threads = []
    for i in range(beam):
        threads.append(
            MyThread(read_fit, args=(fileList[i],))
        )
        f_name.append(fileList[i])

    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()

    for thread in threads: 
        data.append(thread.get_result())
    data=np.array(data)
    return data,f_name