import numpy as np
import matplotlib.pyplot as pl
import zarr
from astropy.io import fits
from tqdm.auto import trange
import GridMatch as GM
import glob
from noise_svd import noise_estimation

n_sequences = 10
n_patches = 100
n_pix = 64

root = '/net/diablos/scratch/sesteban/reduc/reduc_andres'
root_out = '/net/drogon/scratch1/aasensio/sst_unroll'

region = ['qs_20190801_073940', 'qs_20190801_105555', 'pl_20200806_083038_3934', 'pl_20200806_083038_6173', \
    'pl_20200806_083038_8542', 'spot_20200726_090257_3934', 'spot_20200726_090257_6173', 'spot_20200726_090257_8542']

wl = [8542, 8542, 3934, 6173, 8542, 3934, 6173, 8542]

n_regions = len(region)
n_indices = 9

n_total = n_regions * n_indices * n_sequences * n_patches

print(f'Number of total patches : {n_total}')

tiles = [8,16,32,48]
clips = [15,10,5,3]
nthreads = 16

create = True
loop = 0

for ireg, reg in enumerate(region):
    tmp = reg.split('_')[1:]
    tmp = '_'.join(tmp)

    wavel = wl[ireg]

    if (wavel == 3934):
        instr = 'CHROMIS'
    else:
        instr = 'CRISP'

    # List all files and select n_indices from them
    files = glob.glob(f'{root}/{reg}/wb_{tmp}_nwav_al_*.fits')
    files = files[0:n_indices]
    files = [f.split('_')[-1].split('.')[0] for f in files]

    counter = 0
    
    for index in files:
        filename = f'{root}/{reg}/wb_{tmp}_nwav_al_{index}.fits'
        f_wb = fits.open(filename)
        filename = f'{root}/{reg}/nb_{tmp}_nwav_al_{index}.fits'
        f_nb = fits.open(filename)

        tmp2 = filename.split('/')        
        print('/'.join(tmp2[-2:]))

        if (instr == 'CHROMIS'):
            nac, nwav, ny, nx = f_wb[0].data.shape
            nac = 12              
        else:
            nac, nwav, nmod, ny, nx = f_wb[0].data.shape      

        if create:
            fout = zarr.open(f'{root_out}/training.zarr', 'w')
            db_wb = fout.create_dataset('wb', shape=(n_total, nac, n_pix, n_pix), chunks=(nac, n_pix, n_pix), dtype='float32')
            db_nb = fout.create_dataset('nb', shape=(n_total, nac, n_pix, n_pix), chunks=(nac, n_pix, n_pix), dtype='float32')
            db_wl = fout.create_dataset('wl', shape=(n_total), dtype='float32')
            db_sigmawb = fout.create_dataset('sigma_wb', shape=(n_total), dtype='float32')
            db_sigmanb = fout.create_dataset('sigma_nb', shape=(n_total), dtype='float32')
            create = False

        wb = f_wb[0].data
        nb = f_nb[0].data

        if (instr == 'CHROMIS'):
            ind_wav = np.random.randint(0, nwav, n_sequences)            
        else:
            ind_wav = np.random.randint(0, nwav, n_sequences)
            ind_nmod = np.random.randint(0, nmod, n_sequences)
            ind_cam = np.random.randint(0, 2, n_sequences)

        wb_mem = np.zeros((n_sequences, n_patches, nac, n_pix, n_pix), dtype='float32')
        nb_mem = np.zeros((n_sequences, n_patches, nac, n_pix, n_pix), dtype='float32')
        wl_mem = np.zeros((n_sequences, n_patches), dtype='float32')
        sigmawb_mem = np.zeros((n_sequences, n_patches), dtype='float32')
        sigmanb_mem = np.zeros((n_sequences, n_patches), dtype='float32')        

        for i in range(n_sequences):
            if (instr == 'CHROMIS'):
                im_wb = wb[0:nac, ind_wav[i], :, :]
                im_nb = nb[0:nac, ind_wav[i], :, :]
                print(f'CHROMIS : wl={ind_wav[i]} - Images: {counter}-{counter+n_patches}')
            else:
                im_wb = wb[0:nac, ind_wav[i], ind_nmod[i], :, :]
                im_nb = nb[ind_cam[i], 0:nac, ind_wav[i], ind_nmod[i], :, :]
                print(f'CRISP   : wl={ind_wav[i]} - cam={ind_cam[i]} - modul={ind_nmod[i]} - Images: {counter}-{counter+n_patches}')

            # Destretch WB image
            cor, im_wb = GM.DSGridNestBurst(im_wb.astype('float64'), tiles, clips, nthreads = nthreads, apply_correction = True)
            
            # Apply the destretching to the NB image
            for j in range(nac):
                im_nb[j, :, :] = GM.Stretch(im_nb[j, :, :].astype('float64'), cor[j], nthreads= nthreads)            

            # Now extract patches from these images
            indx = np.random.randint(0, nx - n_pix, n_patches)
            indy = np.random.randint(0, ny - n_pix, n_patches)
            
            for j in range(n_patches):
                wb_mem[i, j, :, :, :] = im_wb[:, indy[j]:indy[j]+n_pix, indx[j]:indx[j]+n_pix]
                nb_mem[i, j, :, :, :] = im_nb[:, indy[j]:indy[j]+n_pix, indx[j]:indx[j]+n_pix]
                wl_mem[i, j] = wl[ireg]
                
                # Estimate the noise level in each image
                x = np.concatenate([wb_mem[i, j, :, :, :][None, ...], nb_mem[i, j, :, :, :][None, ...]], axis=0).reshape((2, nac, n_pix*n_pix))
                sigma = noise_estimation(x)

                sigmawb_mem[i, j] = sigma[0]
                sigmanb_mem[i, j] = sigma[1]

            counter += n_patches
        
        # Save the data
        db_wb[loop:loop+n_sequences*n_patches, :, :, :] = wb_mem.reshape((n_sequences*n_patches, nac, n_pix, n_pix))
        db_nb[loop:loop+n_sequences*n_patches, :, :, :] = nb_mem.reshape((n_sequences*n_patches, nac, n_pix, n_pix))
        db_wl[loop:loop+n_sequences*n_patches] = wl_mem.reshape((n_sequences*n_patches))
        db_sigmawb[loop:loop+n_sequences*n_patches] = sigmawb_mem.reshape((n_sequences*n_patches))
        db_sigmanb[loop:loop+n_sequences*n_patches] = sigmanb_mem.reshape((n_sequences*n_patches))

        loop += n_sequences*n_patches