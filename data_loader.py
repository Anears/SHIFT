import os
import cv2
import numpy as np
from PIL import Image
from typing import Any, Tuple

from torch.utils.data import Dataset


class OODDataset(Dataset):
    def __init__(self, files, num_classes, inpaint_method='shift', transform=None):
        self.files = files
        self.num_classes = num_classes
        self.inpaint_method = inpaint_method
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.inpaint_method == 'shift':
            img_file = self.files[index]
            img = np.array(Image.open(img_file).convert('RGB')).astype(np.uint8)
            img = Image.fromarray(img)

        else:
            shift_img_pth = self.files[index]
            img_size = Image.open(shift_img_pth).size
            
            split_pth = shift_img_pth.split('/')
            mask_pth = shift_img_pth.replace('/'.join(split_pth[-4:-2]), 'mask')
            sr_pth = shift_img_pth.replace('/'.join(split_pth[-4:-2]), 'sr')
            
            inpainted_img_pth = shift_img_pth.replace('/'.join(split_pth[-4:-2]), self.inpaint_method)
            os.makedirs(os.path.dirname(inpainted_img_pth), exist_ok=True)
            
            if os.path.exists(inpainted_img_pth):
                img = Image.open(inpainted_img_pth)
                
            else:
                mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
                sr = cv2.imread(sr_pth)
                sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
                
                if self.inpaint_method == 'zero':
                    sr[mask > 0, :] = 0
                elif self.inpaint_method == 'fast-marching':
                    sr = cv2.inpaint(sr, mask, 3, cv2.INPAINT_TELEA)
                
                img = Image.fromarray(sr)
                img = img.resize(img_size)
                img.save(inpainted_img_pth)
            
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.num_classes