import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from models.resnet_custom import resnet50_baseline, resnet50_full, resnet50_MoCo, resnet18_ST
from importlib import import_module

import argparse
from utils.utils import print_network, collate_features
from utils.utils import get_slide_id, get_slide_fullpath
from utils.utils import get_color_normalizer, get_color_augmenter
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
from termcolor import colored
import torchvision.transforms.functional as TF

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

def compute_w_loader(arch, file_path, output_path, wsi, model,
    batch_size = 8, verbose = 0, print_every=20, imagenet_pretrained=True, 
    custom_downsample=1, target_patch_size=-1, sampler_setting=None, custom_transforms=None,
    save_h5_path=None, **kws):
    """
    args:
        arch: the name of model to use
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        imagenet_pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
        sampler_setting: custom defined, samlping settings
        custom_transforms: custom defined, used to transform images, e.g., mean and std normalization.
        color_normalizer: normalization for color space of pathology images
        color_augmenter: color augmentation for patch images
        add_patch_noise: adding noise to patch images
        save_h5_path: path to save features as h5 files
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, imagenet_pretrained=imagenet_pretrained, 
        custom_downsample=custom_downsample, target_patch_size=target_patch_size, 
        sampler_setting=sampler_setting, custom_transforms=custom_transforms)
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.multi_gpu:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=False)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, **kwargs, collate_fn=collate_features)
    else:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))


    all_feats = None
    all_coors = None
    for count, (batch, coords) in enumerate(loader):
        coords = torch.from_numpy(coords)
        with torch.no_grad():   
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            mini_bs = coords.shape[0]

            features = model(batch)

            features = features.cpu() if not isinstance(features, tuple) else (features[0].cpu(), features[1].cpu())
            
            if all_feats is None:
                all_feats = features
                all_coors = coords
            else:
                if isinstance(all_feats, tuple) and isinstance(features, tuple):
                    all_feats = (torch.cat([all_feats[0], features[0]], axis=0), torch.cat([all_feats[1], features[1]], axis=0))
                else:
                    all_feats = torch.cat([all_feats, features], axis=0)

                all_coors = torch.cat([all_coors, coords], axis=0)
    
    if isinstance(all_feats, tuple):
        print("two features' size:", all_feats[0].shape)
        torch.save(all_feats[0], output_path[0])
        torch.save(all_feats[1], output_path[1])
        print("saved pt files:", output_path)
    else:
        print('features size:', all_feats.shape)
        torch.save(all_feats, output_path)
        print('saved pt file:', output_path)
    
    if save_h5_path is not None:
        if isinstance(all_feats, tuple):
            asset_dict_0 = {'features': all_feats[0].numpy(), 'coords': all_coors.numpy()}
            save_hdf5(save_h5_path[0], asset_dict_0, attr_dict=None, mode='w')
            asset_dict_1 = {'features': all_feats[1].numpy(), 'coords': all_coors.numpy()}
            save_hdf5(save_h5_path[1], asset_dict_1, attr_dict=None, mode='w')
            print('saved h5 file:', save_h5_path)
        else:
            asset_dict = {'features': all_feats.numpy(), 'coords': all_coors.numpy()}
            save_hdf5(save_h5_path, asset_dict, attr_dict=None, mode='w')
            print('saved h5 file:', save_h5_path)
    
    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--arch', type=str, default='CONCH', choices=['HOVERNET'], help='Choose which architecture to use for extracting features.')
parser.add_argument('--ckpt_path', type=str, default=None, help='The checkpoint path for pretrained models.')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--feat_dir_ext', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--slide_in_child_dir', default=False, action='store_true')
parser.add_argument('--save_h5', default=False, action='store_true')
parser.add_argument('--multi_gpu', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=256)
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError('No csv_path is gotten.')

    bags_dataset = Dataset_All_Bags(csv_path)


    args_feat_dir = args.feat_dir
    os.makedirs(args_feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args_feat_dir, 'pt_files'), exist_ok=True)
    if args.save_h5:
        os.makedirs(os.path.join(args_feat_dir, 'h5_files'), exist_ok=True)


    print('loading model checkpoint of arch {} from {}'.format(args.arch, args.ckpt_path))
    args_imagenet_pretrained = True
    args_custom_transforms = None
    if args.arch == 'HOVERNET':
        model_desc = import_module("models.hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model")

        model = model_creator(mode="original")
        saved_state_dict = torch.load(args.ckpt_path)["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        def pad_to_270(image):
            return TF.pad(image, padding=(23, 23, 23, 23), fill=0, padding_mode='constant')

        model.load_state_dict(saved_state_dict, strict=True)
        args_custom_transforms = transforms.Compose([
            transforms.Resize(args.target_patch_size),
            transforms.Lambda(pad_to_270),
            transforms.ToTensor(),
            # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])
    else:
        raise NotImplementedError("Please specify a valid architecture.")
    
    print(model)
    
    if args.multi_gpu:
        # 初始化分布式进程组
        dist.init_process_group(backend="nccl")
        # 设置当前 GPU
        device = int(os.environ["LOCAL_RANK"])  # 通过 torchrun 设置的环境变量
        torch.cuda.set_device(device)
        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)
    else:
        model = model.to(device)


    # print_network(model)
    # if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
        
    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_name = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        slide_id = get_slide_id(slide_name, has_ext=False, in_child_dir=args.slide_in_child_dir)
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        
        if not os.path.exists(h5_file_path):
            print('skiped slide {}, h5 file not found'.format(slide_id))
            continue
        
        slide_file_path = get_slide_fullpath(
            args.data_slide_dir, slide_name, 
            in_child_dir=args.slide_in_child_dir
        ) + args.slide_ext
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        # prepare save paths 
        if isinstance(args_feat_dir, tuple):
            output_pt_path = (os.path.join(args_feat_dir[0], 'pt_files', slide_id + '.pt'), os.path.join(args_feat_dir[1], 'pt_files', slide_id + '.pt'))
            if args.save_h5:
                output_h5_path = (os.path.join(args_feat_dir[0], 'h5_files', slide_id + '.h5'), os.path.join(args_feat_dir[1], 'h5_files', slide_id + '.h5'))
            else:
                output_h5_path = None

            if args.auto_skip and os.path.exists(output_pt_path[0]) and os.path.exists(output_pt_path[1]):
                print('skipped {}'.format(slide_id))
                continue

        else:
            output_pt_path = os.path.join(args_feat_dir, 'pt_files', slide_id+'.pt')
            if args.save_h5:
                output_h5_path = os.path.join(args_feat_dir, 'h5_files', slide_id+'.h5')
            else:
                output_h5_path = None

            if args.auto_skip and os.path.exists(output_pt_path):
                print('skipped {}'.format(slide_id))
                continue
        
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(args.arch, h5_file_path, output_pt_path, wsi, 
            model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, imagenet_pretrained=args_imagenet_pretrained,
            custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
            custom_transforms=args_custom_transforms, 
            save_h5_path=output_h5_path, 
        )
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
