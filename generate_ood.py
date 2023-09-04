import os
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torchvision.datasets import STL10, CIFAR10, CIFAR100

from models.shift import ImageEditing, SuperResolution


def save_file(save_dir, subdir, file_name, data):
    file_path = os.path.join(save_dir, subdir, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.save(file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='stl10', choices=['stl10', 'cifar10', 'cifar100'])
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--save_dir', type=str, default='./datasets')
    
    parser.add_argument('--save_raw', action='store_true')
    parser.add_argument('--save_sr', action='store_true')
    parser.add_argument('--save_mask', action='store_true')
    
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.dataset)


    # Dataset
    if args.dataset == 'stl10':
        trainset = STL10(args.data_dir, split='train', download=True)
    elif args.dataset == 'cifar10':
        trainset = CIFAR10(args.data_dir, train=True, download=True)
    elif args.dataset == 'cifar100':
        trainset = CIFAR100(args.data_dir, train=True, download=True)
    with open(f'{args.dataset}_classes.txt') as f:
        classes = f.read().splitlines()


    # Initialize models
    device = f"cuda:{args.gpu}"
    sr_model = SuperResolution(device=device)
    remove_model = ImageEditing(device=device)


    # Generate OOD dataset
    cnt_class = [0] * len(classes)
    negative_prompt = ', '.join(classes)
    for i in range(len(trainset)):
        img, target = trainset.__getitem__(i)
        lbl = classes[target]
        
        file_name = f'{cnt_class[target]}.png'
        print(f'[{i}/{len(trainset)}] {lbl} ({cnt_class[target]})')
        if args.save_raw:
            save_file(args.save_dir, f'raw/{lbl}', file_name, img)
        
        # upscaling (32x32 or 96x96 -> 128x128 -> 512x512)
        upscaled_img = sr_model.inference(img, lbl, seed=args.seed)
        if args.save_sr:
            save_file(args.save_dir, f'sr/{lbl}', file_name, upscaled_img)
        
        # remove object (512x512 -> 512x512)
        updated_image, mask = remove_model.inference_kirby(
            upscaled_img, lbl, seed=args.seed, 
            num_images_per_prompt=args.num_images_per_prompt,
            negative_prompt=negative_prompt
        )
        if updated_image is not None:
            if args.save_mask:
                save_file(args.save_dir, f'mask/{lbl}', file_name, mask)
            for idx in range(len(updated_image)):
                save_file(args.save_dir, f'{idx}/train/{lbl}', file_name, updated_image[idx])
        else:
            print(f'No object found in {cnt_class[target]}th image. (target: {lbl})')
        
        cnt_class[target] += 1
        torch.cuda.empty_cache()