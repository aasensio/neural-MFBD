import napari
import numpy as np
import h5py
from tqdm import tqdm
from movie_gen import WRITERS


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

def dummy_image_generator(root='reconstructed/validation_plage_8542', label='', i0=0, i1=10, minval=0.0, maxval=1.0):
    for i in range(i0, i1):        
        f = h5py.File(f'{root}.{i:02d}.{label}.h5', 'r')
        rec = np.nan_to_num(f['reconstruction_nn'][:])
        frame = np.nan_to_num(f['frame0'][:])
        f.close()
        
        rec = (rec - minval[:, None, None]) / (maxval[:, None, None] - minval[:, None, None])
        _, nx, ny = rec.shape
        rec = rec.reshape((2*nx, ny))        
        rec = np.tile(rec[:, :, None], (1, 1, 3))

        frame = (frame - minval[:, None, None]) / (maxval[:, None, None] - minval[:, None, None])
        _, nx, ny = frame.shape
        frame = frame.reshape((2*nx, ny))        
        frame = np.tile(frame[:, :, None], (1, 1, 3))

        out = np.concatenate([frame, rec], axis=1)
                        
        yield out

if __name__ == '__main__':
    
    label = '2023-02-27-20:34.SST'

    roots = ['reconstructed/val_spot_20200727_083509.8542', 'reconstructed/val_spot_20200727_083509.3934', 'reconstructed/val_qs_20190801_081547.8542']
    i0s = [0, 70, 10]
    i1s = [50, 130, 50]

    for i in range(len(roots)):
        root = roots[i]
        i0 = i0s[i]
        i1 = i1s[i]
        minval, maxval = setting_ranges(root=root, 
            label=label,
            i0=i0, 
            i1=i1)

        iterator = dummy_image_generator(root=root, 
            label=label,
            i0=i0, 
            i1=i1, 
            minval=0.95*minval, 
            maxval=1.05*maxval)
        WRITERS["ffmpeg"](itr=iterator, out_file=f"{root}.{label}.mp4", fps=5)


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