# This code is modified from the original code https://github.com/kookie12/FlexiEdit/blob/main/flexiedit/frequency_utils.py
# Copyright (c) 2024 FlexiEdit 
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.fft as fft
import math
import pdb
from scipy.ndimage import gaussian_filter
import numpy as np
from PIL import Image

'''This code is from freeinit => https://github.com/TianxingWu/FreeInit'''

def freq_2d(x, LPF, alpha):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    #x_freq_low = x_freq * (a * LPF + b * HPF)
    #x_freq_high = x_freq * ( (1-a) * LPF + (1-b) * HPF )
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    #x_freq_sum = x_freq_low + x_freq_high
    x_freq_sum = x_freq

    # IFFT
    _x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(_x_freq_low, dim=(-3, -2, -1)).real
    x_low_alpha = fft.ifftn(_x_freq_low*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(_x_freq_high, dim=(-3, -2, -1)).real
    x_high_alpha = fft.ifftn(_x_freq_high*alpha, dim=(-3, -2, -1)).real
    _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
    
    _x_freq_low_alpha_high = fft.ifftshift(x_freq_low + x_freq_high*alpha, dim=(-3, -2, -1))
    x_low_alpha_high = fft.ifftn(_x_freq_low_alpha_high, dim=(-3, -2, -1)).real
    
    _x_freq_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high, dim=(-3, -2, -1))
    x_high_alpha_low = fft.ifftn(_x_freq_high_alpha_low, dim=(-3, -2, -1)).real

    _x_freq_alpha_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high*alpha, dim=(-3, -2, -1))
    x_alpha_high_alpha_low = fft.ifftn(_x_freq_alpha_high_alpha_low, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum, x_low_alpha, x_high_alpha, x_low_alpha_high, x_high_alpha_low, x_alpha_high_alpha_low

def freq_1d(x, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3))
    x_freq = fft.fftshift(x_freq, dim=(-3))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    #x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    #x_freq_mixed = x_freq_low + x_freq_high

    # IFFT
    x_freq_low = fft.ifftshift(x_freq_low, dim=(-3))
    x_low = fft.ifftn(x_freq_low, dim=(-3)).real
    
    x_freq_high = fft.ifftshift(x_freq_high, dim=(-3))
    x_high = fft.ifftn(x_freq_high, dim=(-3)).real

    return x_low, x_high

def freq_3d(x, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    a = 1.0
    b = 0.1
    #x_freq_low = x_freq * (a * LPF + b * HPF)
    #x_freq_high = x_freq * ( (1-a) * LPF + (1-b) * HPF )
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    #x_freq_sum = x_freq_low + x_freq_high
    x_freq_sum = x_freq

    # IFFT
    x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(x_freq_low, dim=(-3, -2, -1)).real
    
    x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(x_freq_high, dim=(-3, -2, -1)).real
    
    x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(x_freq_sum, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum

def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = (noise_freq) * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    
    #x_freq_high = x_freq * HPF
    #x_freq_mixed = 1.5*x_freq_low + x_freq_high

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed

def org_freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed

def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "gaussian_b":
        return gaussian_band_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask

def gaussian_band_pass_filter(shape, d_s=0.3, d_t=0.1, d_l=0.3):  
    """  
    Compute the Gaussian band-pass filter mask.  
    Consider that the highest part of image is noise.
    Filter it as well as filter the low-frequency conponents
    """  
    T, H, W = shape[-3], shape[-2], shape[-1]  
    # Create a low-pass filter using the existing logic  
    low_pass_mask = torch.zeros(shape)  
    high_pass_mask = torch.zeros(shape)  
    # Avoid operations if the stop frequencies are set to zero  
    if d_s == 0 or d_t == 0:  
        return low_pass_mask - high_pass_mask  
    for t in range(T):  
        for h in range(H):  
            for w in range(W):  
                # Compute squared distances for the low-pass filter  
                d_square_lp = (((d_s/d_l)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)  
                low_pass_mask[..., t, h, w] = math.exp(-1/(2*d_s**2) * d_square_lp)  
            
                # Compute squared distances for the high-pass filter  
                d_square_hp = (((d_t/d_l)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)  
                high_pass_mask[..., t, h, w] = 1 - math.exp(-1/(2*d_t**2) * d_square_hp)  
    # Band-pass filter is low-pass minus high-pass  
    band_pass_mask =  high_pass_mask + low_pass_mask
    
    return band_pass_mask  

def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask

def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask

def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    #mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0
    mask[..., crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask

def freq_exp_mask(feat, mode, user_mask, auto_mask, filter_type= "gaussian", n=4, d_s=0.3, d_t=0.3, alpha=0.7):
    """ Frequency manipulation for latent space. """
    feat = feat.view(4,1,64,64)
    f_shape = feat.shape # 1, 4, 64, 64
    LPF = get_freq_filter(f_shape, feat.device, filter_type, n, d_s, d_t) # d_s, d_t
    f_dtype = feat.dtype
    feat_low, feat_high, feat_sum, feat_low_alpha, feat_high_alpha, feat_low_alpha_high, feat_high_alpha_low, x_alpha_high_alpha_low = freq_2d(feat.to(torch.float64), LPF, alpha)
    feat_low = feat_low.to(f_dtype)
    feat_high = feat_high.to(f_dtype)
    feat_sum = feat_sum.to(f_dtype)
    feat_low_alpha = feat_low_alpha.to(f_dtype)
    feat_high_alpha = feat_high_alpha.to(f_dtype)
    feat_low_alpha_high = feat_low_alpha_high.to(f_dtype)
    feat_high_alpha_low = feat_high_alpha_low.to(f_dtype)

    # latent LPF
    latent_low = feat_low.view(1,4,64,64)
    # latent HPF
    latent_high = feat_high.view(1,4,64,64)
    # latent SUM (original)
    latent_sum = feat_sum.view(1,4,64,64)
    
    # latent_low_alpha = feat_low_alpha.view(1,4,64,64)
    # latent_high_alpha = feat_high_alpha.view(1,4,64,64)
    latent_low_alpha_high = feat_low_alpha_high.view(1,4,64,64)
    latent_high_alpha_low = feat_high_alpha_low.view(1,4,64,64)
    
    mask = torch.zeros_like(latent_sum)
    if mode == "auto_mask":
        auto_mask = auto_mask.unsqueeze(1) # [1,64,64] => [1,1,64,64]
        mask = auto_mask.expand_as(latent_sum) # [1,1,64,64] => [1,4,64,64]
        
    elif mode == "user_mask":
        bbx_start_point, bbx_end_point = user_mask
        mask[:, :, bbx_start_point[1]//8:bbx_end_point[1]//8, bbx_start_point[0]//8:bbx_end_point[0]//8] = 1
        
    latents_shape = latent_sum.shape
    random_gaussian = torch.randn(latents_shape, device=latent_sum.device)
    
    # Apply gaussian scaling
    g_range = random_gaussian.max() - random_gaussian.min()
    l_range = latent_low_alpha_high.max() - latent_low_alpha_high.min()
    random_gaussian = random_gaussian * (l_range/g_range)

    # No scaling applied. If you wish to apply scaling to the mask, replace the following lines accordingly.
    s_range, r_range, s_range2, r_range2 = 1, 1, 1, 1
    
    # edit区域为1
    latent_mask_h = latent_sum * (1 - mask) + (latent_low_alpha_high + (1-alpha)*random_gaussian) * (s_range/r_range) *mask # edit 할 부분에 high frequency가 줄어들고 가우시안 더하기
    latent_mask_l = latent_sum * (1 - mask) + (latent_high_alpha_low + (1-alpha)*random_gaussian) * (s_range2/r_range2) *mask # edit 할 부분에 low frequency가 줄어들고 가우시안 더하기
    
    return latent_mask_h, latent_mask_l, latent_sum # latent_low, latent_high, latent_sum

def freq_exp(feat, filter_type= "gaussian", n=4, d_s=0.3, d_t=0.3, alpha=0.7):
    """ Frequency manipulation for latent space, maintain the high frequency part of content image and low frequency part of style image.
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
        alpha: add noise strength (0.0 ~ 1.0)
    """
    f_shape = feat.shape # 1, 4, 128, 128
    # feat = feat.permute(1, 0, 2, 3) # 4, 1, 128, 128
    LPF = get_freq_filter(f_shape, feat.device, filter_type, n, d_s, d_t) # d_s, d_t
    f_dtype = feat.dtype
    feat_low, feat_high, feat_sum, feat_low_alpha, feat_high_alpha, feat_low_alpha_high, feat_high_alpha_low, x_alpha_high_alpha_low = freq_2d(feat.to(torch.float64), LPF, alpha)
    feat_low = feat_low.to(f_dtype)
    feat_high = feat_high.to(f_dtype)
    feat_sum = feat_sum.to(f_dtype)
    feat_low_alpha = feat_low_alpha.to(f_dtype)
    feat_high_alpha = feat_high_alpha.to(f_dtype)
    feat_low_alpha_high = feat_low_alpha_high.to(f_dtype)
    feat_high_alpha_low = feat_high_alpha_low.to(f_dtype)

    # latent LPF
    latent_low = feat_low.view(f_shape)
    # latent HPF
    latent_high = feat_high.view(f_shape)
    # latent SUM (original)
    latent_sum = feat_sum.view(f_shape)
    
    # latent_low_alpha = feat_low_alpha.view(1,4,64,64)
    # latent_high_alpha = feat_high_alpha.view(1,4,64,64)
    latent_low_alpha_high = feat_low_alpha_high.view(f_shape)
    latent_high_alpha_low = feat_high_alpha_low.view(f_shape)

    mask = torch.ones_like(latent_sum)

    latents_shape = latent_sum.shape
    random_gaussian = torch.randn(latents_shape, device=latent_sum.device)
    
    # Apply gaussian scaling
    g_range = random_gaussian.max() - random_gaussian.min()
    l_range = latent_low_alpha_high.max() - latent_low_alpha_high.min()
    random_gaussian = random_gaussian * (l_range/g_range)

    # No scaling applied. If you wish to apply scaling to the mask, replace the following lines accordingly.
    s_range, r_range, s_range2, r_range2 = 1, 1, 1, 1
    
    # edit区域为1
    latent_mask_h = latent_sum * (1 - mask) + (latent_low_alpha_high + (1-alpha)*random_gaussian) * (s_range/r_range) * mask # edit 할 부분에 high frequency가 줄어들고 가우시안 더하기
    latent_mask_l = latent_sum * (1 - mask) + (latent_high_alpha_low + (1-alpha)*random_gaussian) * (s_range2/r_range2) * mask # edit 할 부분에 low frequency가 줄어들고 가우시안 더하기
    
    return latent_mask_h, latent_mask_l, latent_sum # latent_low, latent_high, latent_sum

def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered

def Gaussian_highpass_filter(x, sigma = 5, amp = 0.5):  
    dtype = x.dtype  
    x_origin = x.clone()
    x = x.type(torch.float32)
    # FFT  
    x_freq = fft.fftn(x, dim=(-2, -1))  
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))  
    B, C, H, W = x_freq.shape  
    crow, ccol = H // 2, W // 2  
    # 创建高斯高通掩码  
    y = torch.arange(H).unsqueeze(1).expand(H, W)  
    x = torch.arange(W).unsqueeze(0).expand(H, W)  
    # 中心点  
    x_center = ccol  
    y_center = crow  
    # 高斯函数   
    gauss_mask = torch.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))  
    highpass_mask = 1 - amp * gauss_mask  # 反转成高通滤波掩码  
    highpass_mask = highpass_mask.unsqueeze(0).unsqueeze(0)  # 调整为形状 (1, 1, H, W)  
    highpass_mask = highpass_mask.expand(B, C, H, W)  # 扩展到 batch size 和通道数  
    # 应用高通掩码  
    x_freq = x_freq * highpass_mask.to(x_freq.device)
    
    # IFFT  
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))  
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real  
    
    x_filtered = x_filtered.type(dtype)# + 0.5 * x_origin.to(x_filtered.device)
    return x_filtered

def get_lowAndHigh_image(image, low_thred=5):
    image_array = np.array(image)  
    low_pass = gaussian_filter(image_array, sigma=low_thred)  
    high_pass = image_array - low_pass  
    high_pass = np.clip(high_pass, 0, 255)
    low_pass_image = Image.fromarray(np.uint8(low_pass))  
    high_pass_image = Image.fromarray(np.uint8(high_pass))
    return low_pass_image, high_pass_image


