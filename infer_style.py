'''
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

'''
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import PIL
import torch.fft as fft

import diffusers
from diffusers.utils import load_image, make_image_grid
from diffusers import AutoencoderKL, DDIMScheduler, ControlNetModel, UniPCMultistepScheduler, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

import torch
import torchvision
import clip
from torchvision import transforms

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from scipy.ndimage import gaussian_filter
from inversion import run as invert
from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline

from ip_adapter.pipeline_stable_diffusion_sdxl_extra_cfg import StableDiffusionXLPipelineExtraCFG
from ip_adapter.pipeline_stable_diffusion_extra_cfg import StableDiffusionPipelineCFG
from ip_adapter.ip_adapter_instruct import IPAdapterInstructSDXL, IPAdapterInstruct

from src.frequency_utils import freq_exp


''' This function may have been modified by [InstantStyle-plus][https://github.com/instantX-research/InstantStyle-Plus]'''
def generate_caption(
    image: Image.Image,
    text: str = None,
    decoding_method: str = "Nucleus sampling",
    temperature: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.5,
    max_length: int = 50,
    min_length: int = 1,
    num_beams: int = 5,
    top_p: float = 0.9,
) -> str:
    
    if text is not None:
        inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs)
    else:
        inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            do_sample=decoding_method == "Nucleus sampling",
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            top_p=top_p,
        )
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result

''' This function may have been modified by [InstantStyle-plus][https://github.com/instantX-research/InstantStyle-Plus]'''
def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def get_depth_map(image):
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def get_canny_map(input_image_cv2):
    input_image_cv2 = cv2.Canny(input_image_cv2, 100, 200)
    input_image_cv2 = input_image_cv2[:, :, None]
    input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
    anyline_image = Image.fromarray(input_image_cv2)
    return anyline_image

def init_models(config):
    noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    )
    # load SD pipeline StableDiffusionXLPipelineExtraCFG
    if config.choose_pipeline == "sd15":
        ip_ckpt = "./checkpoints/models/ip-adapter-instruct-sd15.bin"
        image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" # "openai/clip-vit-large-patch14"
        pipe = StableDiffusionPipelineCFG.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            scheduler=noise_scheduler,
            # vae=vae,
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None,
        )
        ip_model = IPAdapterInstruct(
            sd_pipe=pipe, 
            image_encoder_path=image_encoder_path,
            ip_ckpt=ip_ckpt,
            device=config.device,
            dtypein=torch.float16,
            num_tokens=16)
    else:
        ip_ckpt = "./checkpoints/models/ip-adapter-instruct-sdxl.bin"
        image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        pipe = StableDiffusionXLPipelineExtraCFG.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            scheduler=noise_scheduler,
            # vae=vae,
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None,
        )
        ip_model = IPAdapterInstructSDXL(
            sd_pipe=pipe, 
            image_encoder_path=image_encoder_path,
            ip_ckpt=ip_ckpt,
            device=config.device,
            dtypein=torch.float16,
            num_tokens=16)
        
    return ip_model

if __name__ == "__main__":

    if not os.path.exists("results"):
        os.makedirs("results")

    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    config = RunConfig(model_type = model_type,
                        num_inference_steps = 50,
                        num_inversion_steps = 50,
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = 1234
                        )
    device = config.device
    
    # load blip2 model
    MODEL_ID = "Salesforce/blip2-flan-t5-xl"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="cuda", load_in_8bit=False, torch_dtype=torch.float16)
    model.eval()

    # load IPAdapter-Instruct Pipeline
    ip_instruct_model = init_models(config)
    
    # load images
    content_instruct_prompt = "use the composition from the image"
    style_instruct_prompt = "use the style from the image"
    style_image_dir = config.style_image_dir
    style_image = Image.open(style_image_dir).convert("RGB").resize((config.resolution, config.resolution))

    content_image_dir = config.content_image_dir
    content_image = Image.open(content_image_dir).convert("RGB")
    ori_img_size = content_image.size
    content_image = content_image.resize((config.resolution, config.resolution))

    # use BLIP Model to get the prompt of style image
    if config.style_image_prompt is None:
        style_image_prompt = generate_caption(style_image)
    else:
        style_image_prompt = config.style_image_prompt

    # use IP-Instruct Model get embedding of style information and content information
    style_embeddings_instruct = ip_instruct_model.get_decouple_embeds(pil_image=style_image,prompt="",query=style_instruct_prompt)
    style_content_embeddings = ip_instruct_model.get_decouple_embeds(pil_image=style_image,prompt="",query=content_instruct_prompt)
    print(style_image_prompt)

    content_image_prompt = generate_caption(content_image)
    content_embeddings_instruct = ip_instruct_model.get_decouple_embeds(pil_image=content_image,prompt="",query=content_instruct_prompt)
    content_style_instruct = ip_instruct_model.get_decouple_embeds(pil_image=content_image,prompt="",query=style_instruct_prompt)
    print(content_image_prompt)

    # if need mask, read from mask_path
    entire_mask = Image.new("RGB", (config.resolution, config.resolution), color=(255, 255, 255))
    masks = [entire_mask]

    # preprocess
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    # inversion
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device, model_name=config.base_model_path)#"./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE")
    # obtain content latent
    _, inv_latent, _, all_latents = invert(content_image,
                                           content_image_prompt,
                                           config,
                                           pipe_inversion=pipe_inversion,
                                           pipe_inference=pipe_inference,
                                           do_reconstruction=False,
                                           feature_extractor=ip_instruct_model,
                                           style_embedding=style_embeddings_instruct,
                                           content_embedding=content_embeddings_instruct,
                                           neg_style_embedding=content_style_instruct,
                                           neg_content_embedding=style_content_embeddings,
                                           enable_guidance = False,
                                           used_NPI_guidance = True) # torch.Size([1, 4, 128, 128])
    # frequency manipulation
    latent_h, latent_l, latent_sum = freq_exp(inv_latent,d_s=0.3,d_t=0.9,alpha=0.7,filter_type="gaussian_b")
    latent_l = latent_l.to(inv_latent.dtype)
    latent_h = latent_h.to(inv_latent.dtype)

    del pipe_inversion, pipe_inference, all_latents, model
    torch.cuda.empty_cache()
    
    ####################################################################################################################################################################
    # load ControlNet
    ####################################################################################################################################################################

    control_type = config.control_type
    if control_type == "tile":
        # condition image
        cond_image = load_image(content_image_dir)
        cond_image = cond_image.resize((config.resolution, config.resolution))#resize_img(cond_image)
        
        controlnet_path = config.tile_controlnet_path
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
        controlnet_conditioning_scale = 0.25
        
    elif control_type == "canny":
        # condition image
        input_image_cv2 = cv2.imread(content_image_dir)
        input_image_cv2 = np.array(input_image_cv2)
        anyline_image = get_canny_map(input_image_cv2)
        cond_image = anyline_image.resize((config.resolution, config.resolution))#resize_img(anyline_image)

        # load ControlNet
        controlnet_path = config.canny_controlnet_path
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        controlnet_conditioning_scale = 0.2

    elif control_type == "depth":
        # condition image
        depth_image = get_depth_map(content_image)
        cond_image = depth_image.resize((config.resolution, config.resolution))#resize_img(depth_image)
        controlnet_path = config.depth_controlnet_path
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, 
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        controlnet_conditioning_scale = 0.4

    elif control_type == "combine":
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        depth_image = get_depth_map(content_image)
        cond_depth_image = depth_image.resize((config.resolution, config.resolution))#resize_img(depth_image)

        input_image_cv2 = cv2.imread(content_image_dir)
        input_image_cv2 = np.array(input_image_cv2)
        anyline_image = get_canny_map(input_image_cv2)
        cond_canny_image = anyline_image.resize((config.resolution, config.resolution))#resize_img(anyline_image)

        controlnet_depth_path = config.depth_controlnet_path
        controlnet_canny_path = config.canny_controlnet_path

        controlnet = [
            ControlNetModel.from_pretrained(
                controlnet_depth_path, 
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device),
            ControlNetModel.from_pretrained(
                controlnet_canny_path, 
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device),
        ]
        cond_image = [cond_depth_image, cond_canny_image]
        controlnet_conditioning_scale = [0.4, 0.4]

    elif control_type == "tile_canny":
        cond_image = load_image(content_image_dir)
        cond_tile_image = cond_image.resize((config.resolution, config.resolution))#resize_img(cond_image)
        
        input_image_cv2 = cv2.imread(content_image_dir)
        input_image_cv2 = np.array(input_image_cv2)
        anyline_image = get_canny_map(input_image_cv2)
        cond_canny_image = anyline_image.resize((config.resolution, config.resolution))#resize_img(anyline_image)

        # load ControlNet
        controlnet_canny_path = config.canny_controlnet_path
        controlnet_tile_path = config.tile_controlnet_path
        controlnet = [
            ControlNetModel.from_pretrained(
                controlnet_tile_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(device),
            ControlNetModel.from_pretrained(
                controlnet_canny_path,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device)
        ]
        cond_image = [cond_tile_image,cond_canny_image]
        controlnet_conditioning_scale = [0.25, 0.40]

    # load pipeline
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=config.dtype
    ).to(config.device)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=config.dtype).to(config.device)
    pipe_inference = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    # "./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE",
                    config.base_model_path,
                    controlnet=controlnet,
                    vae=vae,                    
                    # clip_model=clip_model, # 这个只在guidance的时候用了，现在暂时不加
                    image_encoder=image_encoder,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                ).to(device)
    pipe_inference.scheduler = UniPCMultistepScheduler.from_config(pipe_inference.scheduler.config)
    # pipe_inference.scheduler = DDIMScheduler.from_config(pipe_inference.scheduler.config) # works the best
    pipe_inference.unet.enable_gradient_checkpointing()

    pipe_inference.load_ip_adapter(
        config.IP_path, subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.safetensors",
        image_encoder_folder=None,
    )
    scale_style = {
        "up": {"block_0": [0.0, 2.5, 0.0]},
    }
    pipe_inference.set_ip_adapter_scale(scale_style)

    # infer
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    save_name = config.content_image_dir.split('/')[-1][:-4] + '_' + config.style_image_dir.split('/')[-1][:-4]
    output = pipe_inference(
        prompt=content_image_prompt,                    # prompt used for inversion
        negative_prompt="watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        # negative_prompt_2=style_image_prompt,
        num_inference_steps=config.num_inference_steps,
        eta=1.0,
        mask_image=entire_mask,
        image=content_image,
        control_image=cond_image, 
        ip_adapter_image=style_image,
        generator=generator,
        latents=latent_l,#inv_latent,#
        guidance_scale=config.guidance_scale,
        #denoising_start=0.0001,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        npi_interp=0.5,
        style_embeddings_instruct=style_embeddings_instruct,
        content_embeddings_instruct=content_embeddings_instruct,
        style_guidance_scale=config.style_guidance_scale,
        content_guidance_scale=config.content_guidance_scale,
        ip_instruct_model=ip_instruct_model,
        CSD_model = None,#clip_model,
        inv_guidance=config.inv_guidance,
        feature_extractor=ip_instruct_model,
        do_NPI = False,
    ).images[0]
    output.save(os.path.join(config.result_path, f"ours_{save_name}.png"))
        
    torch.cuda.empty_cache()