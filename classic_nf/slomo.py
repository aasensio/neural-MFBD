import numpy as np
import model_slomo as model
from torchvision import transforms
from torch.functional import F
import torch

def find_frames(video_frames, bad_frames):
    below = []
    above = []
    for bf in bad_frames:        
        for j in range(0, bf):            
            if j not in bad_frames:
                tmp_below = j
        for j in range(len(video_frames), bf, -1):            
            if j not in bad_frames:
                tmp_above = j
        below.append(tmp_below)
        above.append(tmp_above)

    below = np.array(below)
    above = np.array(above)
    t = (bad_frames - below) / (above - below)

    return below, above, t

class VideoInterpolation(object):
    def __init__(self, device):
        super().__init__()
        self.device = device
                
        if self.device != "cpu":
            mean = [0.429, 0.431, 0.397]
            mea0 = [-m for m in mean]
            std = [1] * 3
            self.trans_forward = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
            self.trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std)])

        self.flow = model.UNet(6, 4).to(self.device)
        self.interp = model.UNet(20, 5).to(self.device)
        self.back_warp = None       
        self.window = None
        self.checkpoint = 'SuperSloMo.ckpt' 
        self.load_models()

    def setup_back_warp(self, w, h):        
        self.w, self.h = w, h
        with torch.set_grad_enabled(False):
            self.back_warp = model.backWarp(w, h, self.device).to(self.device)

    def load_models(self):
        self.states = torch.load(self.checkpoint, map_location='cpu')
        self.interp.load_state_dict(self.states['state_dictAT'])
        self.flow.load_state_dict(self.states['state_dictFC'])

    def convert_images(self, image0, image1):
        
        frame0 = self.trans_forward(image0.permute(2, 0, 1))
        frame1 = self.trans_forward(image1.permute(2, 0, 1))

        return frame0, frame1

    def deconvert_image(self, image):        
        frame = self.trans_backward(image)
        
        return frame

    def interpolate(self, frame0, frame1, t):
        
        if (self.back_warp is None):
            w, h = (frame0.shape[2] // 32) * 32, (frame0.shape[3] // 32) * 32
            self.setup_back_warp(w, h)        

        if (self.window is None):
            win = np.hanning(self.npix_apod)
            winOut = np.ones(self.w)
            winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
            winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
            self.window = np.outer(winOut, winOut)
            self.window = torch.tensor(self.window.astype('float32')).to(self.device)

        i0 = frame0.to(self.device)
        i1 = frame1.to(self.device)
        ix = torch.cat([i0, i1], dim=1)

        flow_out = self.flow(ix)
        f01 = flow_out[:, :2, :, :]
        f10 = flow_out[:, 2:, :, :]

        temp = -t * (1 - t)
        co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        ft0 = co_eff[0] * f01 + co_eff[1] * f10
        ft1 = co_eff[2] * f01 + co_eff[3] * f10

        gi0ft0 = self.back_warp(i0, ft0)
        gi1ft1 = self.back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = self.interp(iy)

        ft0f = io[:, :2, :, :] + ft0
        ft1f = io[:, 2:4, :, :] + ft1
        vt0 = F.sigmoid(io[:, 4:5, :, :])
        vt1 = 1 - vt0

        gi0ft0f = self.back_warp(i0, ft0f)
        gi1ft1f = self.back_warp(i1, ft1f)

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
            (co_eff[0] * vt0 + co_eff[1] * vt1)

        ft_p = self.deconvert_image(ft_p)

        # Apodize
        mean_val = torch.mean(ft_p, dim=(2, 3), keepdims=True)
        ft_p -= mean_val
        ft_p *= self.window[None, None, :, :]
        ft_p += mean_val
                
        return ft_p
    
    def fix_video(self, images, all_frames, bad_frames, npix_apod):
        self.npix_apod = npix_apod

        n_b, _, nx, ny = images.shape
        images_PIL = torch.zeros((n_b, nx, ny, 3))
        for i in range(n_b):
            for j in range(3):
                images_PIL[i, :, :, j] = images[i, 0, :, :]

        below, above, t = find_frames(all_frames, bad_frames)

        corrected = []
        loop = 0
        for i in range(n_b):
            if i in bad_frames:      
                frame0, frame1 = self.convert_images(images_PIL[below[loop], ...], images_PIL[above[loop], ...])
                with torch.no_grad():
                    out = self.interpolate(frame0[None, ...], frame1[None, ...], t[loop])
                corrected.append(out[0, 0, :, :][None, :, :])
                loop += 1
            else:
                corrected.append(images[i, 0, ...][None, :, :])
        
        corrected = torch.cat(corrected, dim=0)

        return corrected

