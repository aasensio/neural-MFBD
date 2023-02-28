import numpy as np
import h5py
import images

outf = '/net/drogon/scratch1/aasensio/unroll_mfbd/classic_regularized/test.h5'

nac = 12
nscans = 50

fout = h5py.File(outf, 'w')

for i in range(0, 50):
    wb, nb, wl = images.read_images_sst(iregion=9, index=i, wav=7)    
    _, nx, ny = wb.shape
    
    if (i == 0):
        dset = fout.create_dataset('images', (nscans, nac, nx, ny), dtype='float32')
        dset_mem = np.zeros((nscans, nac, nx, ny), dtype='float32')

    dset_mem[i, :, :, :] = wb

dset[:] = dset_mem[:]

fout.close()