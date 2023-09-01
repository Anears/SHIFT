import cv2
import random
import numpy as np
from PIL import Image
from compel import Compel

import torch

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class MaskFormer:
    def __init__(self, device):
        print(f"Initializing MaskFormer to {device}")
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cahce_dir='/data1/gitaek')
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir='/data1/kirby/.cache').to(device)

    def inference(self, image_path, text):
        threshold = 0.2
        min_area = 0.02
        padding = 25
        if isinstance(image_path, str):
            original_image = Image.open(image_path)
        else: original_image = image_path
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        
        visual_mask = cv2.dilate((mask*255).astype(np.uint8), np.ones((padding, padding), np.uint8))
        
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)


class ImageEditing:
    def __init__(self, device):
        print(f"Initializing ImageEditing to {device}")
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", revision=self.revision, torch_dtype=self.torch_dtype, cache_dir='/data1/kirby/.cache').to(device)
        self.compel = Compel(tokenizer=self.inpaint.tokenizer, text_encoder=self.inpaint.text_encoder)

    def inference_kirby(self, original_image, to_be_replaced_txt, 
                        replace_with_txt='backdrop++, background++, backgrounds++', 
                        seed=42, num_images_per_prompt=1, negative_prompt=''):
        if seed is not None:
            seed_everything(seed)
        
        assert original_image.size == (512, 512)
        
        mask_image = self.mask_former.inference(original_image, to_be_replaced_txt)
        if mask_image is None:
            return None, None
        
        list_negative_prompt = negative_prompt.split(', ')
        list_negative_prompt.insert(0, list_negative_prompt.pop(list_negative_prompt.index(to_be_replaced_txt)))
        negative_prompt = ', '.join(list_negative_prompt)
        negative_prompt = negative_prompt.replace(to_be_replaced_txt, f'{to_be_replaced_txt}++')
        
        conditioning_pos = self.compel.build_conditioning_tensor(replace_with_txt)
        conditioning_neg = self.compel.build_conditioning_tensor(negative_prompt)
        updated_images = self.inpaint(
            image=original_image,
            prompt_embeds=conditioning_pos,
            negative_prompt_embeds=conditioning_neg,
            mask_image=mask_image, 
            guidance_scale=7.5, 
            num_inference_steps=50,
            num_images_per_prompt=num_images_per_prompt
        ).images
            
        return updated_images, mask_image


class SuperResolution:
    def __init__(self, device):
        print(f"Initializing SuperResolution to {device}")
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.Upscaler_sr = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", revision=self.revision, 
            torch_dtype=self.torch_dtype, cache_dir='/data1/kirby/.cache').to(device)
    
    def inference(self, image, prompt, seed=None, baselen=128):
        if seed is not None:
            seed_everything(seed)
        
        old_img = image.resize((baselen, baselen))
        upscaled_img = self.Upscaler_sr(prompt=prompt, guidance_scale=7.5, image=old_img, num_inference_steps=50).images[0]
        
        return upscaled_img