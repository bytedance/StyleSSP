# Copyright (c) 2023 pix2pixzero 
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import PIL
from torchvision import transforms

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def noise_regularization(
    e_t, noise_pred_optimal, lambda_kl, lambda_ac, num_reg_steps, num_ac_rolls, generator=None
):
    for _outer in range(num_reg_steps):
        if lambda_kl > 0:
            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
            l_kld = patchify_latents_kl_divergence(_var, noise_pred_optimal)
            l_kld.backward()
            _grad = _var.grad.detach()
            _grad = torch.clip(_grad, -100, 100)
            e_t = e_t - lambda_kl * _grad
        if lambda_ac > 0:
            for _inner in range(num_ac_rolls):
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_ac = auto_corr_loss(_var, generator=generator)
                l_ac.backward()
                _grad = _var.grad.detach() / num_ac_rolls
                e_t = e_t - lambda_ac * _grad
        e_t = e_t.detach()

    return e_t

# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def auto_corr_loss(
        x, random_shift=True, generator=None
):
    B, C, H, W = x.shape
    assert B == 1
    x = x.squeeze(0)
    # x must be shape [C,H,W] now
    reg_loss = 0.0
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = torch.randint(0, noise.shape[2] // 2, (1,), generator=generator).item()
            else:
                roll_amount = 1
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=2)
            ).mean() ** 2
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=3)
            ).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss

def patchify_latents_kl_divergence(x0, x1, patch_size=4, num_channels=4):

    def patchify_tensor(input_tensor):
        patches = (
            input_tensor.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        patches = patches.contiguous().view(-1, num_channels, patch_size, patch_size)
        return patches

    x0 = patchify_tensor(x0)
    x1 = patchify_tensor(x1)

    kl = latents_kl_divergence(x0, x1).sum()
    return kl

def Fourier_filter(x, threshold, scale):
    # b1: 1.3, b2: 1.4, s1: 0.9, s2: 0.2
    # b1: 1 ≤ b1 ≤ 1.2
    # b2: 1.2 ≤ b2 ≤ 1.6
    # s1: s1 ≤ 1
    # s2: s2 ≤ 1
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    # scale > 1时，对低频部分进行增强，为低通滤波
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered

def latents_kl_divergence(x0, x1):
    EPSILON = 1e-6
    x0 = x0.view(x0.shape[0], x0.shape[1], -1)
    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
    mu0 = x0.mean(dim=-1)
    mu1 = x1.mean(dim=-1)
    var0 = x0.var(dim=-1)
    var1 = x1.var(dim=-1)
    kl = (
        torch.log((var1 + EPSILON) / (var0 + EPSILON))
        + (var0 + (mu0 - mu1) ** 2) / (var1 + EPSILON)
        - 1
    )
    kl = torch.abs(kl).sum(dim=-1)
    return kl

def inversion_step(
    pipe,
    z_t: torch.tensor,
    t: torch.tensor,
    prompt_embeds,
    added_cond_kwargs,
    num_renoise_steps: int = 100,
    first_step_max_timestep: int = 250,
    generator=None,
    pipe_inf=None,
    prompt=None,
    feature_extractor=None,
    style_embedding=None,
    content_embedding=None,
    neg_style_embedding=None,
    neg_content_embedding=None,
    enable_guidance=False,
    used_NPI_guidance=False,
) -> torch.tensor:
    extra_step_kwargs = {}
    avg_range = pipe.cfg.average_first_step_range if t.item() < first_step_max_timestep else pipe.cfg.average_step_range
    num_renoise_steps = min(pipe.cfg.max_num_renoise_steps_first_step, num_renoise_steps) if t.item() < first_step_max_timestep else num_renoise_steps

    nosie_pred_avg = None
    noise_pred_optimal = None
    z_tp1_forward = pipe.scheduler.add_noise(pipe.z_0, pipe.noise, t.view((1))).detach()

    approximated_z_tp1 = z_t.clone()
    for i in range(num_renoise_steps + 1):
        with torch.no_grad():
            # if noise regularization is enabled, we need to double the batch size for the first step
            if pipe.cfg.noise_regularization_num_reg_steps > 0 and i == 0:
                approximated_z_tp1 = torch.cat([z_tp1_forward, approximated_z_tp1])
                prompt_embeds_in = torch.cat([prompt_embeds, prompt_embeds])
                if added_cond_kwargs is not None:
                    added_cond_kwargs_in = {}
                    added_cond_kwargs_in['text_embeds'] = torch.cat([added_cond_kwargs['text_embeds'], added_cond_kwargs['text_embeds']])
                    added_cond_kwargs_in['time_ids'] = torch.cat([added_cond_kwargs['time_ids'], added_cond_kwargs['time_ids']])
                else:
                    added_cond_kwargs_in = None
            else:
                prompt_embeds_in = prompt_embeds
                added_cond_kwargs_in = added_cond_kwargs
            noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds_in, added_cond_kwargs_in)
            # 往白化上guidance
            # noise_pred = Fourier_filter(noise_pred, threshold=30, scale=0.5)
            # if noise regularization is enabled, we need to split the batch size for the first step
            if pipe.cfg.noise_regularization_num_reg_steps > 0 and i == 0:
                noise_pred_optimal, noise_pred = noise_pred.chunk(2)
                if pipe.do_classifier_free_guidance:
                    noise_pred_optimal_uncond, noise_pred_optimal_text = noise_pred_optimal.chunk(2)
                    noise_pred_optimal = noise_pred_optimal_uncond + pipe.guidance_scale * (noise_pred_optimal_text - noise_pred_optimal_uncond)
                noise_pred_optimal = noise_pred_optimal.detach()
            # perform guidance
            if pipe.do_classifier_free_guidance:
                if used_NPI_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Calculate average noise
            if  i >= avg_range[0] and i < avg_range[1]:
                j = i - avg_range[0]
                if nosie_pred_avg is None:
                    nosie_pred_avg = noise_pred.clone()
                else:
                    nosie_pred_avg = j * nosie_pred_avg / (j + 1) + noise_pred / (j + 1)

        if i >= avg_range[0] or (not pipe.cfg.average_latent_estimations and i > 0):
            noise_pred_ = noise_regularization(noise_pred, noise_pred_optimal, lambda_kl=pipe.cfg.noise_regularization_lambda_kl, lambda_ac=pipe.cfg.noise_regularization_lambda_ac, num_reg_steps=pipe.cfg.noise_regularization_num_reg_steps, num_ac_rolls=pipe.cfg.noise_regularization_num_ac_rolls, generator=generator)
            
            has_nan = torch.isnan(noise_pred_).any().item()
            if not has_nan:
                noise_pred = noise_pred_
        
        approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()

        if enable_guidance: # False
            guidance = get_guidace(pipe_inf=pipe_inf, 
                                    latents=approximated_z_tp1, 
                                    prompt=prompt, 
                                    feature_extractor=feature_extractor,
                                    style_embedding=style_embedding,
                                    content_embedding=content_embedding,
                                    neg_style_embedding=neg_style_embedding,
                                    neg_content_embedding=neg_content_embedding,
                                    get_grad_guidance=pipe.cfg.get_grad_guidance)
            scale = rescale_guidance(guidance, noise_pred_text, noise_pred_uncond, pipe.guidance_scale)
            guidance = guidance * scale
            approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred - guidance, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()
        # if i < num_renoise_steps // 2:
        #     approximated_z_tp1 = Fourier_filter(approximated_z_tp1, threshold=1, scale=0.8)
    
        '''
        # img = from_latents2img(pipe_inf, approximated_z_tp1)
        '''
        '''
        这个z_tp1就是latents
        import pdb;pdb.set_trace()
        pipe_inf.vae.to(dtype=torch.float32)
        latents = approximated_z_tp1.to(dtype=torch.float32)
        latents = latents / pipe_inf.vae.config.scaling_factor
        image = pipe_inf.vae.decode(latents, return_dict=False)[0]
        # image = pipe.vae.decode(prev_sample).sample       
        image = (image / 2 + 0.5).clamp(0, 1)
        vis_image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # (2, 1024, 1024, 3)
        vis_image = (vis_image * 255).round().astype("uint8")
        vis_image = PIL.Image.fromarray(vis_image[0])
        vis_image.save(f"image_{i}.jpg")'''

    # if average latents is enabled, we need to perform an additional step with the average noise
    if pipe.cfg.average_latent_estimations and nosie_pred_avg is not None:
        nosie_pred_avg = noise_regularization(nosie_pred_avg, noise_pred_optimal, lambda_kl=pipe.cfg.noise_regularization_lambda_kl, lambda_ac=pipe.cfg.noise_regularization_lambda_ac, num_reg_steps=pipe.cfg.noise_regularization_num_reg_steps, num_ac_rolls=pipe.cfg.noise_regularization_num_ac_rolls, generator=generator)
        approximated_z_tp1 = pipe.scheduler.inv_step(nosie_pred_avg, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()

    # perform noise correction
    if pipe.cfg.perform_noise_correction: # False
        noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds, added_cond_kwargs)

        # perform guidance
        if pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        pipe.scheduler.step_and_update_noise(noise_pred, t, approximated_z_tp1, z_t, return_dict=False, optimize_epsilon_type=pipe.cfg.perform_noise_correction)

    return approximated_z_tp1

def spherical_dist_loss(x, y):
    return -x@y.T

@torch.no_grad()
def rescale_guidance(guidance, noise_pred_text, noise_pred_uncond, guidance_scale, cutoff = 2000.0):
    norm_cfg = torch.norm(guidance_scale * (noise_pred_text - noise_pred_uncond), p=2)
    norm_guidance = torch.norm(guidance, p=2)
    scale = norm_cfg/norm_guidance
    scale = torch.where(scale < cutoff, scale, torch.tensor(cutoff, device=scale.device))  # 确保 tensor 在相同设备上  
    return scale

@torch.enable_grad()
def get_guidace(pipe_inf, 
                latents, 
                prompt, 
                feature_extractor,
                style_embedding,
                content_embedding,
                neg_style_embedding,
                neg_content_embedding,
                clip_model_used=True,
                get_grad_guidance=False):
    origin_dtype = latents.dtype
    if get_grad_guidance:
        latents = latents.detach().requires_grad_(True)
        '''
        img = pipe_inf(prompt = prompt,
                    num_inference_steps = 1,#cfg.num_inference_steps,
                    negative_prompt = prompt,
                    image = latents,
                    strength = pipe_inf.cfg.inversion_max_step,
                    denoising_start = 1.0 - pipe_inf.cfg.inversion_max_step,
                    guidance_scale = 1.0,
                    return_dict=False,
                    get_grad_guidance=get_grad_guidance)[0]#.images[0]
        img = pipe_inf.inf_step(
            latents=latents,
            get_grad_guidance=get_grad_guidance,
            return_dict=False,
        )[0]'''
        # pipe_inf.upcast_vae()
        pipe_inf.vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
        latents = latents / pipe_inf.vae.config.scaling_factor
        image = pipe_inf.vae.decode(latents, return_dict=False)[0]
        if clip_model_used:
            _, content_output, style_output = feature_extractor(normalize(transforms.Resize(224)(image[0:1]))) # 当前timeStep的图像
        else:
            image_tensor = (transforms.Resize(224)(image))
            # clip_image = self.clip_image_processor(images=image_tensor, return_tensors='pt',do_rescale=False).pixel_values
            clip_image = image_tensor.to(pipe_inf.device)
            style_output = feature_extractor.get_decouple_embeds(clip_image=clip_image, prompt="", query="use the style from the image").to(latents.dtype)
            content_output = feature_extractor.get_decouple_embeds(clip_image=clip_image, prompt="", query="use the composition from the image").to(latents.dtype)
    else:
        img = pipe_inf(prompt = prompt,
                    num_inference_steps = 1,#cfg.num_inference_steps,
                    negative_prompt = prompt,
                    image = latents,
                    strength = pipe_inf.cfg.inversion_max_step,
                    denoising_start = 1.0 - pipe_inf.cfg.inversion_max_step,
                    guidance_scale = 1.0,
                    get_grad_guidance=get_grad_guidance).images[0]
        # img.save('tmp.jpg')
        if clip_model_used:
            _, content_output, style_output = feature_extractor(normalize(transforms.Resize(224)(image[0:1]))) # 当前timeStep的图像
        else:
            style_output = feature_extractor.get_decouple_embeds(pil_image=img, prompt="", query="use the style from the image")
            content_output = feature_extractor.get_decouple_embeds(pil_image=img, prompt="", query="use the composition from the image")

    loss = 0.0
    if pipe_inf.cfg.inv_style_guidance_scale > 0.0:
        # style_loss = (1 - torch.nn.CosineSimilarity(dim=-1)(style_output, style_embedding.to(latents.dtype)).mean())  * pipe_inf.cfg.inv_style_guidance_scale
        style_loss = spherical_dist_loss(style_output, style_embedding.to(latents.dtype)).mean()  * pipe_inf.cfg.inv_style_guidance_scale
        loss += style_loss
    if pipe_inf.cfg.inv_content_guidance_scale > 0.0:
        # content_loss = (1 - torch.nn.CosineSimilarity(dim=-1)(content_output, content_embedding.to(latents.dtype)).mean())  * pipe_inf.cfg.inv_content_guidance_scale
        content_loss = spherical_dist_loss(content_output, content_embedding.to(latents.dtype)).mean()  * pipe_inf.cfg.inv_content_guidance_scale
        loss += content_loss
    if pipe_inf.cfg.inv_neg_style_guidance_scale > 0.0:
        # neg_style_loss = torch.nn.CosineSimilarity(dim=-1)(style_output, neg_style_embedding.to(latents.dtype)).mean()  * pipe_inf.cfg.inv_neg_style_guidance_scale
        neg_style_loss = -spherical_dist_loss(style_output, neg_style_embedding.to(latents.dtype)).mean()  * pipe_inf.cfg.inv_neg_style_guidance_scale
        loss += neg_style_loss
    if pipe_inf.cfg.inv_neg_content_guidance_scale > 0.0:
        # neg_content_loss = torch.nn.CosineSimilarity(dim=-1)(content_output, neg_content_embedding.to(latents.dtype)).mean()  * pipe_inf.cfg.inv_neg_content_guidance_scale
        neg_content_loss = -spherical_dist_loss(content_output, neg_content_embedding.to(latents.dtype)).mean()  * pipe_inf.cfg.inv_neg_content_guidance_scale
        loss += neg_content_loss
    if get_grad_guidance:
        grad = -torch.autograd.grad(loss, latents)[0]
        latents = latents.to(origin_dtype)
        grad = grad.to(origin_dtype)
        pipe_inf.vae.to(dtype=torch.float16)
        return grad
    return loss

def from_latents2img(pipe_inf, latents):
    needs_upcasting = pipe_inf.vae.dtype == torch.float16 and pipe_inf.vae.config.force_upcast
    if needs_upcasting:
        pipe_inf.upcast_vae()
        latents = latents.to(next(iter(pipe_inf.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe_inf.vae.dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            pipe_inf.vae = pipe_inf.vae.to(latents.dtype)
    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    has_latents_mean = hasattr(pipe_inf.vae.config, "latents_mean") and pipe_inf.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe_inf.vae.config, "latents_std") and pipe_inf.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipe_inf.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(pipe_inf.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / pipe_inf.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe_inf.vae.config.scaling_factor
    image = pipe_inf.vae.decode(latents, return_dict=False)[0]
    # cast back to fp16 if needed
    if needs_upcasting:
        pipe_inf.vae.to(dtype=torch.float16)
    return image

@torch.no_grad()
def unet_pass(pipe, z_t, t, prompt_embeds, added_cond_kwargs):
    latent_model_input = torch.cat([z_t] * 2) if pipe.do_classifier_free_guidance else z_t
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    return pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
