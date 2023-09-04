import os
import glob
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.utils.data import DataLoader, ConcatDataset

from models.wrn import WideResNet
from models.resnet import ResNet34
from models.densenet import DenseNet3
from models.ood_detector import OODDetector
from utils import Accumulator, get_ood_score, get_ood_performance, set_seed
from data_loader import OODDataset


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

def get_ood_test_loaders(args):
    transform_ood = trn.Compose([
        trn.Resize((args.img_size, args.img_size)),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    
    # /////////////// SVHN ///////////////
    svhn_data = dset.SVHN(root=f'{args.data_root}/svhn/', split='test',
                     transform=transform_ood, download=True)
    svhn_loader = DataLoader(svhn_data, batch_size=args.batch_size, num_workers=4)

    # /////////////// LSUN-R ///////////////
    if args.dataset != 'stl10':
        lsun_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/LSUN_resize', transform=transform_ood)
    else:
        lsun_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/SUN', transform=transform_ood)
    lsun_loader = DataLoader(lsun_data, batch_size=args.batch_size, num_workers=4)

    # /////////////// LSUN-C ///////////////
    if args.dataset != 'stl10':
        lsun_crop_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/LSUN_C', transform=transform_ood)
    else:
        lsun_crop_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/SUN',
                                          transform=trn.Compose([trn.Pad(4), trn.Resize((args.img_size, args.img_size)), trn.ToTensor(), trn.Normalize(mean, std)]))
    lsun_crop_loader = DataLoader(lsun_crop_data, batch_size=args.batch_size, num_workers=4)

    # /////////////// Textures ///////////////
    dtd_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/dtd/images', transform=transform_ood)
    dtd_loader = DataLoader(dtd_data, batch_size=args.batch_size, num_workers=4)

    # /////////////// PLACE ///////////////
    place_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/Places', transform=transform_ood)
    place_loader = DataLoader(place_data, batch_size=args.batch_size, num_workers=4)

    # /////////////// iSUN ///////////////
    isun_data = dset.ImageFolder(root=f'{args.data_root}/kirby_dataset/iSUN', transform=transform_ood)
    isun_loader = DataLoader(isun_data, batch_size=args.batch_size, num_workers=4)
    
    return {'svhn': svhn_loader, 'dtd': dtd_loader, 'lsun-crop': lsun_crop_loader, 
            'lsun-resize': lsun_loader, 'places': place_loader, 'isun': isun_loader}


def get_ind_dataset(args):
    transform_train = trn.Compose([
        trn.Resize((args.img_size, args.img_size)), 
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])
    if args.dataset == 'stl10':
        ind_train_data = dset.STL10(f'{args.data_root}/stl10', split='train', download=True, transform=transform_train)
        ind_test_data = dset.STL10(f'{args.data_root}/stl10', split='test', download=True, transform=transform_train)
    elif args.dataset == 'cifar10':
        ind_train_data = dset.CIFAR10(f'{args.data_root}/cifarpy', train=True, download=True, transform=transform_train)
        ind_test_data = dset.CIFAR10(f'{args.data_root}/cifarpy', train=False, download=True, transform=transform_train)
    else:
        ind_train_data = dset.CIFAR100(f'{args.data_root}/cifarpy', train=True, download=True, transform=transform_train)
        ind_test_data = dset.CIFAR100(f'{args.data_root}/cifarpy', train=False, download=True, transform=transform_train)
    
    return ind_train_data, ind_test_data


def get_ood_train_dataset(args):
    valid_image_files = glob.glob(os.path.join(args.data_root, 'shift', '*/*'))
    print(f"OOD NUM : {len(valid_image_files)}")
    
    transform_ood = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomAutocontrast(p=0.5),
        trn.RandomChoice([
            trn.RandomAffine((-180, 180), (0, 0.1), (1, 2)),
            trn.RandomResizedCrop((args.img_size, args.img_size), scale=(0.08,0.3)),
        ]),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])
    return OODDataset(valid_image_files, args.num_classes, transform_ood)


def get_model(args):
    if args.arch == 'wrn':
        net = WideResNet(40, args.num_classes, 2, 0.3)
    elif args.arch == 'densenet':
        net = DenseNet3(100, args.num_classes, dataset=args.dataset)
    elif args.arch == 'resnet':
        net = ResNet34(num_classes=args.num_classes, dataset=args.dataset)
    net.load_state_dict(torch.load(f'{args.data_root}/kirby_pretrained_model_ckpt/{args.arch}/{args.dataset}_{args.arch}_pretrained_epoch_99.pt', map_location=args.device))
    net = net.eval()
    ood_detector = OODDetector(arc=args.arch, img_size=args.img_size)
    
    if torch.cuda.is_available():
        net = net.cuda()
        ood_detector = ood_detector.cuda()
    
    return net, ood_detector


def train_ood_detector(args, test_ood_loaders):
    # get loaders
    ind_train_dataset, ind_test_dataset = get_ind_dataset(args)
    ood_train_loader = get_ood_train_dataset(args)
    train_dataset = ConcatDataset([ind_train_dataset, ood_train_loader])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    ind_test_loader = DataLoader(ind_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # get model
    model, ood_detector = get_model(args)

    criterion = F.binary_cross_entropy_with_logits()
    optimizer = torch.optim.SGD(ood_detector.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # train ood detector
    for epoch in range(args.epochs):
        metrics = Accumulator()
        loader = tqdm(train_loader, disable=False, ascii=True)
        loader.set_description('[%s %03d/%03d]' % ('train', epoch + 1, args.epochs))
        cnt = 0
        for x_data, y_data in loader:
            cnt += args.batch_size

            if torch.cuda.is_available():
                x_data = x_data.to(args.device)
                y_data = y_data.to(args.device)

            optimizer.zero_grad()
            features = model.get_all_blocks(x_data)

            pred = ood_detector(features.detach())
            y_data = torch.where(y_data == args.num_classes, torch.ones_like(y_data), torch.zeros_like(y_data))
            loss = criterion(pred, y_data.view(-1, 1).to(args.device).to(torch.float32))

            metrics.add_dict({
                'loss': loss.item() * args.batch_size,
            })
            postfix = metrics / cnt
            loader.set_postfix(postfix)
            loss.backward()
            optimizer.step()

        scheduler.step()
    
    # evaluate ood detector
    ind_scores = get_ood_score(model, ood_detector, ind_test_loader, args.device)

    for ood_name, ood_loader in enumerate(test_ood_loaders):
        ood_scores = get_ood_score(model, ood_detector, ood_loader, args.device)
        auroc, aupr, fpr = get_ood_performance(-ind_scores, -ood_scores)
        print(f"[{args.dataset}, {ood_name}] AUROC : {auroc}, AUPR : {aupr}, FPR : {fpr}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--dataset', default='stl10', type=str, choices=['cifar10', 'cifar100', 'stl10'])
    parser.add_argument('--data_root', default='/data1/shift', type=str)
    parser.add_argument('--num_try', default=1, type=int)
    parser.add_argument('--num_ood', default=1, type=int)

    # Loading details
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    # rejection network params
    parser.add_argument('--arch', default='wrn', type=str, choices=['wrn', 'resnet', 'densenet'])
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_id}')
    args.img_size = 96 if args.dataset == 'stl10' else 32
    args.num_classes = 100 if args.dataset == 'cifar100' else 10
    
    ood_loaders = get_ood_test_loaders(args)
    
    train_ood_detector(args, ood_loaders)