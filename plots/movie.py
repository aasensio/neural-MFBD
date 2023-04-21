import napari
import numpy as np
import h5py
from tqdm import tqdm
import sys
sys.path.append('../modules')
from movie_gen import WRITERS
from astropy.io import fits


def setting_ranges(root='reconstructed/validation_plage_8542', label='', i0=0, i1=10):
    minval = np.array([1e100, 1e100])
    maxval = np.array([-1e100, -1e100])

    for i in range(i0, i1):        
        f = h5py.File(f'{root}.{i:02d}.{label}.h5', 'r')
        rec = np.nan_to_num(f['reconstruction_nn'][:])        
        f.close()

        ind0 = np.where(rec[0, :, :] > 0)
        ind1 = np.where(rec[1, :, :] > 0)

        minv = np.array([np.min(rec[0, ind0[0], ind0[1]]), np.min(rec[1, ind1[0], ind1[1]])])
        maxv = np.array([np.max(rec[0, ind0[0], ind0[1]]), np.max(rec[1, ind1[0], ind1[1]])])
        
        minval = np.minimum(minv, minval)
        maxval = np.maximum(maxv, maxval)

        return minval, maxval

def dummy_image_generator(root=None, name_nn=None, mfbd_name=None, label_unroll='', label_unsup='', i0=0, i1=10, minval=0.0, maxval=1.0):
    for index in range(i0, i1):
        
        f_raw = h5py.File(f'{root[0]}/wav7_mod0_cam0_{index:05d}.h5', 'r')
        f_mfbd_wb = fits.open(f'{root[1]}/camXX_{mfbd_name}_{index:05d}_8542_8542_+65_lc0.fits')[0].data[:, ::-1]
        f_mfbd_nb = fits.open(f'{root[1]}/camXIX_{mfbd_name}_{index:05d}_8542_8542_+65_lc0.fits')[0].data[:, ::-1]
        f_unroll = h5py.File(f'{root[2]}/val_{name_nn}.8542.{index:02d}.{label_unroll}.h5')['reconstruction_nn'][:, 20:, 14:]
        f_unsup = h5py.File(f'{root[3]}/val_{name_nn}.8542.{index:02d}.{label_unsup}.h5')['reconstruction_nn'][:, 20:, 14:]

        raw_wb = f_raw['wb'][0, 0:800, 0:800]
        raw_nb = f_raw['nb'][0, 0:800, 0:800]

        mfbd_wb = f_mfbd_wb[20:20+800, 14:14+800]
        mfbd_nb = f_mfbd_nb[20:20+800, 14:14+800]

        unroll_wb = f_unroll[0, 0:800, 0:800]
        unroll_nb = f_unroll[1, 0:800, 0:800]

        unsup_wb = f_unsup[0, 0:800, 0:800]
        unsup_nb = f_unsup[1, 0:800, 0:800]

        out = np.block([[raw_wb, mfbd_wb, unroll_wb, unsup_wb],[raw_nb, mfbd_nb, unroll_nb, unsup_nb]])

        out = np.tile(out[:, :, None], (1, 1, 3))

        # maxval = np.max(out)
        # minval = np.min(out)
        
        out = ((out - minval) / (maxval - minval) *255).round(0).astype(np.uint8)
                        
        yield out

if __name__ == '__main__':
    
    label_unroll = '2023-03-11-08:03.All'
    label_unsup =  '2023-03-10-12:44.All'

    name = 'spot_20200727_083509_8542'
    name_nn = 'spot_20200727_083509'
    mfbd_name = '2020-07-27T08:35:09'
    i0 = 0
    i1 = 10

    name = 'qs_20190801_081547'
    name_nn = 'qs_20190801_081547'
    mfbd_name = '2019-08-01T08:15:47'
    i0 = 10
    i1 = 50

        
    root_unrolled = f'/scratch1/aasensio/sst_unroll/{name}/unrolled'
    root_unsup = f'/scratch1/aasensio/sst_unroll/{name}/unsup'
    root_mfbd = f'/scratch1/aasensio/sst_unroll/{name}/momfbd'
    root_raw = f'/scratch1/aasensio/sst_unroll/{name}/raw'
    
    roots = [root_raw, root_mfbd, root_unrolled, root_unsup]
    
    minval = 0
    maxval = 700
    # minval, maxval = setting_ranges(root=root_unsup[0], 
    #     label=label,
    #     i0=i0, 
    #     i1=i1)

    
    iterator = dummy_image_generator(root=roots, 
        label_unroll=label_unroll,
        label_unsup=label_unsup,
        name_nn=name_nn,
        mfbd_name=mfbd_name,
        i0=i0, 
        i1=i1, 
        minval=0.95*minval, 
        maxval=1.05*maxval)
    WRITERS["ffmpeg"](itr=iterator, out_file=f"{name}.mp4", fps=5)


#     # Plage 8542
#     minval, maxval = setting_ranges(root='reconstructed/validation_plage_8542', 
#         label=label,
#         i0=0, 
#         i1=11)

#     iterator = dummy_image_generator(root='reconstructed/validation_plage_8542', 
#         label=label,
#         i0=0, 
#         i1=11,
#         minval=0.95*minval, 
#         maxval=1.05*maxval)
#     WRITERS["ffmpeg"](itr=iterator, out_file="reconstructed/plage_8542.mp4", fps=5)

#     # Spot 8542
#     minval, maxval = setting_ranges(root='reconstructed/validation_spot_8542', 
#         label=label,
#         i0=20, 
#         i1=40)

#     iterator = dummy_image_generator(root='reconstructed/validation_spot_8542', 
#         label=label,
#         i0=20, 
#         i1=40, 
#         minval=0.95*minval, 
#         maxval=1.05*maxval)
#     WRITERS["ffmpeg"](itr=iterator, out_file="reconstructed/spot_8542.mp4", fps=5)

#     # Spot 3934
#     minval, maxval = setting_ranges(root='reconstructed/validation_spot_3934', 
#         label=label,
#         i0=300, 
#         i1=320)

#     iterator = dummy_image_generator(root='reconstructed/validation_spot_3934', 
#         label=label,
#         i0=300, 
#         i1=320,
#         minval=0.95*minval,
#         maxval=1.05*maxval)
#     WRITERS["ffmpeg"](itr=iterator, out_file="reconstructed/spot_3934.mp4", fps=5)


# # print("Reading images...")
# # img = []
# # for i in tqdm(range(20,40)):
# #     f = h5py.File(f'validation_spot_8542_{i:02d}.h5', 'r')
# #     tmp = f['reconstruction_nn'][:]
# #     img.append(tmp[None, :, :, :])
# #     f.close()

# # img = np.concatenate(img, axis=0)

# # viewer = napari.view_image(img[:, 1, :, :])
