# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from math import pi
import torchvision
import torch.fft
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
from data_utils import PhasesOnlineDataset
import pickle
import time
import skimage.io
from skimage.transform import rescale, resize
from beam_profile_gen import phase_gen, phase_mask_gen


N = 500 # grid size
px = 1e-6 # pixel size (um)
focal_length = 2e-3
wavelength = 0.561e-6
refractive_index = 1.0
psf_width_pixels = 101
pixel_size_meters = 1e-6
psf_width_meters = psf_width_pixels * pixel_size_meters
numerical_aperture = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
# This function creates a 2D gaussian filter with std=1, without normalization.
# during training this filter is scaled with a random std to simulate different blur per emitter
def gaussian2D_unnormalized(shape=(7, 7), sigma=1.0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    hV = torch.from_numpy(h).type(torch.FloatTensor)
    return hV
        
class BlurLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gauss = gaussian2D_unnormalized(shape=(7, 7)).to(device)
        self.std_min = 0.8
        self.std_max = 1.2
    
    def forward(self, img_4d):
        # number of the input PSF images
        Nbatch = img_4d.size(0)
        Nemitters = img_4d.size(1)
        # generate random gaussian blur for each emitter
        RepeatedGaussian = self.gauss.expand(1, Nemitters, 7, 7)
        stds = (self.std_min + (self.std_max - self.std_min) * torch.rand((Nemitters, 1))).to(device)
        MultipleGaussians = torch.zeros_like(RepeatedGaussian)
        for i in range(Nemitters):
            MultipleGaussians[:, i, :, :] = 1 / (2 * pi * stds[i] ** 2) * torch.pow(RepeatedGaussian[:, i, :, :], 1 / (stds[i] ** 2))
        # blur each emitter with slightly different gaussian
        images4D_blur = F.conv2d(img_4d, MultipleGaussians, padding=(2, 2))
        return images4D_blur

# ================================
# Cropping layer: keeps only the center part of the FOV to prevent unnecessary processing
# ==============================
class Croplayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, images4D):
        H = images4D.size(2)
        mid = int((H-1)/2)
        images4D_crop = images4D[:,:,mid - 20:mid + 21,mid - 20:mid + 21]
        return images4D_crop



class imgs4dto3d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,images4D,xyz):
        Nbatch, Nemitters,H,W = images4D.shape[0],images4D.shape[1],images4D.shape[2],images4D.shape[3]
        img = torch.zeros((Nbatch,1,200,200)).type(torch.FloatTensor).to(device)
        #img.requires_grad_()
        for i in range(Nbatch):
            for j in range(Nemitters):
                x = int(xyz[i,j,0])
                y = int(xyz[i,j,1])
                img[i,0,x - 15:x + 16,y - 15: y + 16] += images4D[i,j]
        return img

class poisson_noise_approx(nn.Module):
    def __init__(self):
        super().__init__()
        self.H, self.W = 200, 200
        self.device = device
        self.mean = 3e8
        self.std = 2e8
    def forward(self, input):
        # number of images
        Nbatch = input.size(0)
        # approximate the poisson noise using CLT and reparameterization
        input = input + 1e5 + (self.std*torch.randn(input.size()) + self.mean).type(torch.FloatTensor).to(self.device)
        input[input <= 0] = 0
        input_poiss = input + torch.tensor(100)*torch.sqrt(input)*torch.randn(Nbatch, 1, self.H, self.W).type(torch.FloatTensor).to(self.device)
        # if torch.isnan(input_poiss).any():
        #     print('yes')
        
        # result
        return input_poiss
    


# Overall noise layer
class NoiseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.poiss = poisson_noise_approx()
        self.unif_bg = 100
        
    def forward(self,input):
        inputb = input + self.unif_bg
        inputb_poiss = self.poiss(inputb)
        # if torch.isnan(inputb).any():
        #     print('yes')
        return inputb_poiss
        
class Normalize01(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,result_noisy):
        Nbatch = result_noisy.size(0)
        result_noisy_01 = torch.zeros_like(result_noisy)
        #min_val = (result_noisy[0, 0, :, :]).min()
        min_val = 0
        #max_val = (result_noisy[:, :, :, :]).max()
        #print(max_val)
        max_val = 4e9
        # if torch.isnan(result_noisy).any():
        #     print('yes')
        result_noisy[result_noisy <= 10] = 1
        result_noisy[result_noisy >= max_val] = max_val
        # for i in range(Nbatch):
        
        #     result_noisy_01[i, :, :, :] = (result_noisy[i, :, :, :] - min_val) / (max_val - min_val)
        result_noisy_01 = (result_noisy)/(max_val)
        return result_noisy_01
    
# ==================================================
# Physical encoding layer, from 3D to 2D:
# this layer takes in the learnable parameter "mask"
# and output the resulting 2D image corresponding to the emitters location.
# ===================================================

class PhysicalLayer(nn.Module):
    def __init__(self):
        super(PhysicalLayer,self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        x = list(range(-N//2,N//2))
        y = list(range(-N//2,N//2))
        [X, Y] = np.meshgrid(x,y)
        X = X*px
        Y = Y*px
        C1 = (np.pi/(wavelength*focal_length) * (np.square(X) + np.square(Y)))%(2*np.pi) # lens function lens as a phase transformer
        
        self.B1 = np.exp(-1j*C1)
        self.B1 = torch.from_numpy(self.B1).type(torch.cfloat).to(device)
        
        xx = list(range(-N + 1, N + 1))
        yy = list(range(-N + 1, N + 1))
        [XX, YY] = np.meshgrid(xx,yy)
        XX = XX*px
        YY = YY*px
        Q1 = np.exp(1j * (np.pi/(wavelength*focal_length)) * (np.square(XX) + np.square(YY))) # Fresnet diffraction equation at distance = focal length
        self.Q1 = torch.from_numpy(Q1).type(torch.cfloat).to(device)
        
        # angular specturm
        k = 2*np.pi/wavelength
        self.k = k
        phy_x = N*px # physical width (meters) 
        phy_y = N*px # physical length (meters) 
        obj_size = [N,N]
        # generate meshgrid
        dx = np.linspace(-phy_x/2,phy_x/2,obj_size[1])
        dy = np.linspace(-phy_y/2,phy_y/2,obj_size[0])
        Fs_x = obj_size[1]/phy_x
        Fs_y = obj_size[0]/phy_y
        dFx = Fs_x/obj_size[1]
        dFy = Fs_y/obj_size[0]
        Fx = np.arange(-Fs_x/2,Fs_x/2,dFx)
        Fy = np.arange(-Fs_y/2,Fs_y/2,dFy)
        # alpha and beta (wavenumber components) 
        alpha = wavelength*Fx
        beta = wavelength*Fy
        [ALPHA,BETA] = np.meshgrid(alpha,beta)
        #gamm_cust = np.zeros((len(alpha),len(beta)))
        gamma_cust = np.sqrt(1 - np.square(ALPHA) - np.square(BETA))
        
        self.gamma_cust = torch.from_numpy(gamma_cust).type(torch.FloatTensor).to(device)
        # read defocus images
        self.imgs = []
        for z in range(0,31):
            img = skimage.io.imread('beads_img_defocus/z' + str(z).zfill(2) + '.tiff')
            self.imgs.append(img[50-15:50+16,50-15:50+16])
        
        
        # initialize phase mask
        x = list(range(-N//2,N//2))
        y = list(range(-N//2,N//2))
        [X, Y] = np.meshgrid(x,y)
        X = X*px
        Y = Y*px
        #c = 0.02
        #mask_init = np.exp(-np.sqrt(np.square(X) + np.square(Y))/(2*c**2))
        c = 0.0001
        mask_init = 1*np.exp(-(np.square(X) + np.square(Y))/(2*c**2))
        #set_2 = mask_init <= 0.75
        #mask_init[set_2] = 0
        
        
        
        self.mask_real = torch.from_numpy(mask_init).type(torch.FloatTensor).to(device)
        
        
        
        self.blur = BlurLayer()
        self.crop = Croplayer()
        self.img4dto3d = imgs4dto3d()
        self.noise = NoiseLayer()
        self.norm01 = Normalize01()
    def forward(self,mask_param,xyz,nphotons):
        a = 1
        #PSF4D = maskphaseTointensity.apply(mask_real,mask_phase,xyz,nphotons)
        Nbatch, Nemitters = xyz.shape[0],xyz.shape[1]
        
        # mask_param = mask_param[None,None,:]
        # mask_real = torch.real(mask_param)
        # mask_imag = torch.imag(mask_param)
        
        mask_param = self.mask_real*torch.exp(1j*mask_param)
        mask_param = mask_param[None,None,:]
        B1 = self.B1*mask_param
        
        
        E1 = F.pad(B1,(250,250,250,250),'constant',0)
        E2 = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(E1) * torch.fft.fft2(self.Q1)))
        
        output_layer = E2[:,:,N//2:3*N//2,N//2:3*N//2]

        # depth-wise normalization
        #imgs_norm_4d = torch.zeros((Nbatch,Nemitters,31,31)).type(torch.FloatTensor).to(device)
        imgs3D = torch.zeros(Nbatch,1,200,200).type(torch.FloatTensor).to(device)
        #imgs_norm_4d.requires_grad_()
        U1 = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(output_layer))*torch.exp(1j*self.k*self.gamma_cust*0)))
        
        U1 = torch.real(U1*torch.conj(U1))
        
        #U1 = U1[:,:,249 - 100:249 + 101,249 - 100:249 + 101] # zeros_position
        #max_intensity = torch.sum(U1[0,0,:,249])
        max_intensity = torch.tensor(5e4)
        # b = a[0,0].detach().cpu().numpy()
        #print(max_intensity)
        #all_intensity = []
        for i in range(Nbatch):
            for j in range(Nemitters):
                # change x value to fit different field of view
                x = xyz[i,j,0].type(torch.LongTensor) - 100
                y = xyz[i,j,1].type(torch.LongTensor)
                z = xyz[i,j,2].type(torch.LongTensor)
                
                x_ori = xyz[i,j,0].type(torch.LongTensor)
                U1 = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(output_layer))*torch.exp(1j*self.k*self.gamma_cust*x * 1e-6)))
                U1 = torch.real(U1*torch.conj(U1))
                # crop middle part int 201*201
                #U1 = U1[:,:,249 - 100:249 + 101,249 - 100:249 + 101]
                intensity = torch.sum(U1[0,0,:,249 + z])
                # if intensity >= max_intensity:
                #     max_intensity = intensity
                #print(x.cpu().item(),z.item(),intensity.cpu().item()/max_intensity.cpu().item())
                #imgs_norm_4d[i,j,:,:] = torch.from_numpy(self.imgs[abs(z.item())].astype('float32')).type(torch.FloatTensor).to(device)*intensity
                imgs3D[i,0,x_ori - 15:x_ori + 16,y - 15: y + 16] += torch.from_numpy(self.imgs[abs(z.item())].astype('float32')).type(torch.FloatTensor).to(device)*intensity
        
        #imgs_norm_4d = imgs_norm_4d/max_intensity
        imgs3D = imgs3D/max_intensity
        # # pure function test
        # for i in range(12):
        #     x = xyz[i,j,0].type(torch.LongTensor) - 300
        #     z = xyz[i,j,2].type(torch.LongTensor) + i
        #     U1 = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(output_layer))*torch.exp(1j*self.k*self.gamma_cust*x * 1e-6)))
        #     U1 = torch.real(U1*torch.conj(U1))
        #     # crop middle part int 201*201
        #     U1 = U1[:,:,249 - 100:249 + 101,249 - 100:249 + 101]
        #     intensity = torch.sum(U1[0,0,100:,100 + z])
        #     print(z.item(),intensity.cpu().item()/max_intensity.cpu().item())
        
        #print(imgs_norm_4d.shape)
        #imgs4D_crop = self.crop(imgs_norm_4d)
        #print(imgs4D_crop.shape)
        #imgs3D = self.img4dto3d(imgs_norm_4d,xyz)
        #print(imgs3D.shape)
        result_noisy = self.noise(imgs3D)
        result_noisy01 = self.norm01(result_noisy)
        return result_noisy01
    
if __name__ == '__main__':
    a = 1
    path_save = 'data_mask_learning/'
    path_train = 'traininglocations/'
    batch_size = 1
    max_epochs = 1
    ntrain = 10000
    nvalid = 1000
    initial_learning_rate = 0.001
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)
    
    # load all locations pickle file
    path_pickle = path_train + 'labels.pickle'
    with open(path_pickle, 'rb') as handle:
        labels = pickle.load(handle)
    
    # parameters for data loaders batch size is 1 because examples are generated 16 at a time
    params_train = {'batch_size': 1, 'shuffle': False}
    params_valid = {'batch_size': 1, 'shuffle': False}
    batch_size_gen = 2
    ntrain_batches = int(ntrain/batch_size_gen)
    nvalid_batches = int(nvalid/batch_size_gen)
    steps_per_epoch = ntrain_batches
    # partition built in simulation
    ind_all = np.arange(0, ntrain_batches + nvalid_batches, 1)
    list_all = ind_all.tolist()
    list_IDs = [str(i) for i in list_all]
    train_IDs = list_IDs[:ntrain_batches]
    valid_IDs = list_IDs[ntrain_batches:]
    partition = {'train': train_IDs, 'valid': valid_IDs}
    
    training_set = PhasesOnlineDataset(partition['train'],labels)
    training_generator = DataLoader(training_set, **params_train)
    
    start_epoch, end_epoch, num_epochs = 0, max_epochs, max_epochs

    phylayer = PhysicalLayer()
    
    # generate initial mask
    x = list(range(-N//2,N//2))
    y = list(range(-N//2,N//2))
    [X, Y] = np.meshgrid(x,y)
    X = X*px
    Y = Y*px
    c = 0.02
    mask_init = np.exp(-np.sqrt(np.square(X) + np.square(Y))/(2*c**2))
    set_2 = mask_init <= 0.75
    mask_init[set_2] = 0

    mask_phase = phase_gen()
    #mask_phase = skimage.io.imread('./results/model_fig4/mask_phase_epoch_100_0.tiff')
    mask_phase = torch.from_numpy(mask_phase).type(torch.FloatTensor).to(device)
    #mask_param = mask_real + 1j*mask_phase
    
    mask_param = nn.Parameter(mask_phase)
    
    
    # cnn = OpticsDesignCNN()
    # cnn.to(device)
    # model_path = './results/phase_model__20221103-031118/net_10.pt'
    # cnn.load_state_dict(torch.load(model_path))
    # mask_real = phase_mask_gen()
    # mask_phase = skimage.io.imread('results/phase_model__20221103-031118/mask_phase_epoch_0_0.tiff')
    # mask_param = nn.Parameter(mask_phase)
    
    # with torch.no_grad():
    #     cnn.eval()
    #     for batch_ind, (xyz_np, Nphotons, targets) in enumerate(training_generator):
    #         xyz_np = xyz_np.to(device)
    #         xyz_np = xyz_np.squeeze()
    #         outputs = cnn(mask_param,xyz_np,Nphotons)
    #         print(outputs)
    #         break
    # os._exit()
    
    for epoch in np.arange(start_epoch,end_epoch):
        epoch_start_time = time.time()
        # print current epoch number
        print('='*20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('='*20)
        with torch.set_grad_enabled(True):
            for batch_ind, (xyz_np, Nphotons, targets) in enumerate(training_generator):
                print(batch_ind)
                print(xyz_np.shape)
                print(targets.shape)
                targets = targets.to(device)
                Nphotons = Nphotons.to(device)
                xyz_np = xyz_np.to(device)
                xyz_np = xyz_np.squeeze()
                targets = targets.squeeze(0)
                Nphotons = Nphotons.squeeze(0)
                a = phylayer(mask_param,xyz_np,Nphotons)
                img = a[0,0,:,:]
                img_np = img.detach().cpu().numpy()
                
                skimage.io.imsave('results/aaaa_' +str(batch_ind) +'.tiff',img_np)
                #print(img_np.shape)
                if np.isnan(img_np).any():
                    print('yes')
                tar = targets[0,0,:,:]
                tar = tar.detach().cpu().numpy()
                skimage.io.imsave('bbbb.tiff',tar)
                #print(targets.shape)
                
                break
                
        break
    torch.cuda.empty_cache()
    
    
    
    
