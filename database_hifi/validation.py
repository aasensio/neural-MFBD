import numpy as np
import matplotlib.pyplot as pl
import zarr
import sunpy.io
from tqdm.auto import trange
import GridMatch as GM

root_in = '/net/dracarys/scratch/ckuckein/gregor/hifiplus/2'
root_out = '/net/drogon/scratch1/aasensio/hifi'

days = ['20220607']
extra = ['']
n_days = len(days)

obs_index = [[170,390]]

fout = zarr.open(f'{root_out}/validation.zarr', 'w')

for id in trange(n_days, desc='days'):
    group = fout.create_group(f'{days[id]}')
    for io in trange(len(obs_index[0]), desc='obs'):
        obs = group.create_group(f'{io:03d}')        

        for i in trange(100, desc='t', leave=False):
            f = sunpy.io.ana.read(f'{root_in}/{days[id]}{extra[id]}/scan_b{obs_index[id][io]:03d}/scanbb_{i:06d}.fz')
            if (i == 0):
                nx, ny = f[0][0].shape
                bb = obs.create_dataset('bb', shape=(100, nx, ny), dtype='i4')
                nb = obs.create_dataset('nb', shape=(100, nx, ny), dtype='i4')

                bb_mem = np.zeros((100, nx, ny), dtype='i4')
                nb_mem = np.zeros((100, nx, ny), dtype='i4')
            bb_mem[i, :, :] = f[0][0]
            f = sunpy.io.ana.read(f'{root_in}/{days[id]}{extra[id]}/scan_b{obs_index[id][io]:03d}/scannb_pos0001_{i:06d}.fz')
            nb_mem[i, :, :] = f[0][0]
        
        tiles = [8,16,32,48]
        clips = [15,10,5,3]
        nthreads = 16

        # Calculate the distorsion grid and apply corrections to the
        # cube
        cor, bb_mem = GM.DSGridNestBurst(bb_mem, tiles, clips, nthreads = nthreads, apply_correction = True)

        bb[:] = bb_mem[:]

        for i in range(100):
            nb_mem[i, :, :] = GM.Stretch(nb_mem[i, :, :], cor[i], nthreads= nthreads)

        nb[:] = nb_mem[:]

