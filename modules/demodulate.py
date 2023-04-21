import numpy as np
from scipy.io import readsav
import h5py
from tqdm import tqdm
import matplotlib.pyplot as pl
from ipdb import set_trace as stop

def combine_pcal_mtel(pcal, mtel):    
    return np.einsum('mnki,jk->mnij', pcal, mtel)

def demodulate_simple(image, d):
    return np.einsum('mnji,jmn->mni', image, d)

def remove_crosstalk(image, line, border):    
    if (line == '8542'):
        idx = [0,-1]
    else:
        idx = [9]

    crt = np.sum(image[border:-border,border:-border,idx,:] * image[border:-border,border:-border,idx,0][:,:,:,None], axis=(0,1,2)) / np.sum(image[border:-border,border:-border,idx,0]**2)
    crt[0] = 0.0

    image -= crt[None,None,None,:] * image[:,:,:,0][:,:,:,None]

    return image

def read_demodulation(matrix=None, telescope=None, scan=0):
    tmp = readsav(telescope)
    scans = tmp['scan']
    
    for i in range(len(scans)):
        if (int(scans[i]) == scan):
            idx = i

    mtel = tmp['itel'][idx,:,:]

    tmp = readsav(matrix)
    mtel_tc = tmp['pcal'].tc[0]
    mtel_rc = tmp['pcal'].rc[0]

    return mtel, mtel_tc, mtel_rc

def demodulate(cube, scan=0, line='6302', mtel=None, ptel_tc=None, ptel_rc=None, wavelength=None, border=100):
    """
    Demodulate a CRISP cube
    Args:
        cube : float
            Array of size [ncams, nstokes, nlambda, ny, nx] with modulated data
        scan : int
            Specific scan
        line : str
            Spectral line of interest
        mtel : float
            Modulation matrix
        ptel_tc : float
            Telescope matrix for reflected beam
        ptel_rc : float
            Telescope matrix for transmitted beam
        wavelength : int
            Indices of wavelength to demodulate. None for all wavelengths. If not None, crosstalk correction will not be applied
        border : int
            Border in pixels to discard when scaling the beams before combination and also cross-talk correction
    
    Returns:
        cube : float
            Modulated cube of size [nx, ny, nlambda, 4]
    """

    ncams, nstokes, nlambda, ny, nx = cube.shape

    if (wavelength is None):
        wavelength = list(range(nlambda))
    
    n_wavelength = len(wavelength)

    out = np.zeros((nx,ny,n_wavelength,nstokes))

    it = combine_pcal_mtel(ptel_tc, mtel)
    ir = combine_pcal_mtel(ptel_rc, mtel)

    for i in tqdm(wavelength):
        re0 = demodulate_simple(it, cube[0,:,i,:,:])
        re1 = demodulate_simple(ir, cube[1,:,i,:,:])

        met = np.mean(re0[border:-border,border:-border,0])
        mer = np.mean(re1[border:-border,border:-border,0])

        scl_t = (met + mer) * 0.25 / met
        scl_r = (met + mer) * 0.25 / mer

        out[:,:,i,:] = np.transpose((re0*scl_t + re1*scl_r), axes=(1,0,2))

    if (n_wavelength == nlambda):
        out = remove_crosstalk(out, line, border)
    else:
        print('Crosstalk removal not applied')

    return out        

if (__name__ == '__main__'):
    f = h5py.File('/scratch1/deepLearning/mfbd_sst/restored_8542_t=09:30:20_scan=00025.h5')

    mtel, ptel_tc, ptel_rc = read_demodulation(matrix='demodulation/ipolcal_8542.idlsave', telescope='demodulation/imtel_8542_09-30-20.idlsave', scan=25)

    nx = 900
    ny = 900

    ptel_tc = ptel_tc[0:nx,0:ny,:,:]
    ptel_rc = ptel_rc[0:nx,0:ny,:,:]

    out = demodulate(f['restored'][1:3,:,:,0:nx,0:ny], line='8542', mtel=mtel, ptel_tc=ptel_tc, ptel_rc=ptel_rc)