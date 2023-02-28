import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import GridMatch as GM
from tqdm.auto import trange
import sunpy.io
from PIL import Image, ImageDraw
from tqdm import trange


def read_images(root, nac=100):
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

    for i in trange(nac, desc='t', leave=False):
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

    return wb_mem, nb_mem


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