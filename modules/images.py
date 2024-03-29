import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import GridMatch as GM
from PIL import Image, ImageDraw
import sunpy.io
from tqdm import tqdm
import os
import h5py

def read_images_sst(iregion, index, wav=0, mod=0, cam=0):
    """
    Read a monochromatic cube from all available observations

    Parameters
    ----------
    iregion : int
        Index of the available regions
    index : int
        Index of the scans for every available region
    wav : int, optional
        Wavelength index
    mod : int, optional
        Modulation index, only for CRISP
    cam : int, optional
        Camera index, only for CRISP

    Returns
    -------
    _type_
        _description_
    """
    tiles = [8,16,32,48]
    clips = [15,10,5,3]
    nthreads = 16

    root = '/net/diablos/scratch/sesteban/reduc/reduc_andres'
    root_sst = '/scratch1/aasensio/sst_unroll'
    
    regions = ['qs_20190801_073940', 'qs_20190801_105555', 'pl_20200806_083038_3934', 'pl_20200806_083038_6173', \
    'pl_20200806_083038_8542', 'spot_20200726_090257_3934', 'spot_20200726_090257_6173', 'spot_20200726_090257_8542', \
    # Validation datasets
    'spot_20200727_083509_3934', 'spot_20200727_083509_8542', 'qs_20190801_081547']

    wls = [8542.0, 8542.0, 3934.0, 6173.0, 8542.0, 3934.0, 6173.0, 8542.0, 3934.0, 8542.0, 8542.0]

    region = regions[iregion]
    wl = wls[iregion]
    
    tmp = region.split('_')[1:]
    tmp = '_'.join(tmp)

    h5file = f'{root_sst}/{region}/raw/wav{wav}_mod{mod}_cam{cam}_{index:05d}.h5'

    if (os.path.exists(h5file)):
        print(f"File {h5file} already exists, reading from disk...")
        f = h5py.File(h5file, 'r')
        im_wb = f['wb'][:]
        im_nb = f['nb'][:]
        wl = f['wl'][()]
    else:

        filename = f'{root}/{region}/wb_{tmp}_nwav_al_{index:05d}.fits'
        f_wb = fits.open(filename)
        filename = f'{root}/{region}/nb_{tmp}_nwav_al_{index:05d}.fits'
        f_nb = fits.open(filename)

        if (wl == 3934):
            instr = 'CHROMIS'
            nac, nwav, ny, nx = f_wb[0].data.shape
            nac = 12              
        else:
            instr = 'CRISP'
            nac, nwav, nmod, ny, nx = f_wb[0].data.shape

        wb = f_wb[0].data
        nb = f_nb[0].data

        if (instr == 'CHROMIS'):
            im_wb = wb[0:nac, wav, :, :]
            im_nb = nb[0:nac, wav, :, :]
        else:
            im_wb = wb[0:nac, wav, mod, :, :]
            im_nb = nb[cam, 0:nac, wav, mod, :, :]

        print("Aligning images...")

        cor, im_wb = GM.DSGridNestBurst(im_wb.astype('float64'), tiles, clips, nthreads = nthreads, apply_correction = True)
                
        # Apply the destretching to the NB image
        for j in range(nac):
            im_nb[j, :, :] = GM.Stretch(im_nb[j, :, :].astype('float64'), cor[j], nthreads= nthreads)   

    # If HDF5 file does not exist, generate it
    if (not os.path.exists(h5file)):
        print(f"Saving {h5file} to disk...")
        fout = h5py.File(h5file, 'w')
        dset_wb = fout.create_dataset('wb', data=im_wb)
        dset_nb = fout.create_dataset('nb', data=im_nb)
        dset_wl = fout.create_dataset('wl', data=wl)
        fout.close()

    return im_wb, im_nb, wl


def read_images_hifi(root, nac=100):
    """
    Read a monochromatic cube from all available observations

    Parameters
    ----------
    iregion : int
        Index of the available regions
    index : int
        Index of the scans for every available region
    wav : int, optional
        Wavelength index
    mod : int, optional
        Modulation index, only for CRISP
    cam : int, optional
        Camera index, only for CRISP

    Returns
    -------
    _type_
        _description_
    """
    tiles = [8,16,32,48]
    clips = [15,10,5,3]
    nthreads = 16

    root_hifi = '/scratch1/aasensio/sst_unroll'

    wl = 6563

    tmp = root.split('/')[-1]

    h5file = f'{root_hifi}/hifi/raw/{tmp}.h5'

    if (os.path.exists(h5file)):
        print(f"File {h5file} already exists, reading from disk...")
        f = h5py.File(h5file, 'r')
        wb_mem = f['wb'][:]
        nb_mem = f['nb'][:]
        wl = f['wl'][()]
    else:

        for i in tqdm(range(nac)):
            f = sunpy.io.ana.read(f'{root}/scanbb_{i:06d}.fz')
            if (i == 0):
                nx, ny = f[0][0].shape

                wb_mem = np.zeros((nac, nx, ny), dtype='i4')
                nb_mem = np.zeros((nac, nx, ny), dtype='i4')
            wb_mem[i, :, :] = f[0][0]
            f = sunpy.io.ana.read(f'{root}/scannb_pos0001_{i:06d}.fz')        
            nb_mem[i, :, :] = f[0][0]

        print("Aligning images...")

        cor, wb_mem = GM.DSGridNestBurst(wb_mem.astype('float64'), tiles, clips, nthreads = nthreads, apply_correction = True)
                
        # Apply the destretching to the NB image
        for j in range(nac):
            nb_mem[j, :, :] = GM.Stretch(nb_mem[j, :, :].astype('float64'), cor[j], nthreads= nthreads)    

    # If HDF5 file does not exist, generate it
    if (not os.path.exists(h5file)):
        print(f"Saving {h5file} to disk...")
        fout = h5py.File(h5file, 'w')
        dset_wb = fout.create_dataset('wb', data=wb_mem)
        dset_nb = fout.create_dataset('nb', data=nb_mem)
        dset_wl = fout.create_dataset('wl', data=wl)
        fout.close()

    return wb_mem, nb_mem, wl



def merge_images(image_batch, size, labels=None):
    b, h, w = image_batch.shape    
    img = np.zeros((int(h*size[0]), int(w*size[1])))
    for idx in range(b):
        i = idx % size[1]
        j = idx // size[1]
        maxval = np.max(image_batch[idx, :, :])
        minval = np.min(image_batch[idx, :, :])
        img[j*h:j*h+h, i*w:i*w+w] = (image_batch[idx, :, :] - minval) / (maxval - minval)

    img_pil = Image.fromarray(np.uint8(pl.cm.viridis(img)*255))
    I1 = ImageDraw.Draw(img_pil)
    n = len(labels)
    for i in range(n):
        I1.text((2, 1+h*i), labels[i], fill=(255,255,255))
    img = np.array(img_pil)

    return img

def align(a, b):        

    if(a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]):
        print("align: ERROR, both images must have the same size")
        return(0.0,0.0)
    
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    corr = np.fft.ifft2(np.conj(fa) * fb)
    cc = np.fft.fftshift(corr.real)
        
    mm = np.argmax(cc)
    xy = ( mm // fa.shape[1], mm % fa.shape[1])

    cc = cc[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]

    y = 2.0*cc[1,1]
    x = (cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-y)*0.5
    y = (cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-y)*0.5

    x += xy[1] - fa.shape[1]//2
    y += xy[0] - fa.shape[0]//2

    return (-y,-x)
