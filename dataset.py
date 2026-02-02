import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import threading
import time

class Data:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True):
        start_time = time.time()
        train_path = os.path.join(self.config.data.train_data_dir)
        val_path = os.path.join(self.config.data.test_data_dir)

        train_dataset = MyDataset(train_path,
                                  n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_load_size,
                                  transforms=self.transforms,
                                  parse_patches=parse_patches)
        val_dataset = MyDataset(val_path,
                                n=self.config.training.patch_n,
                                patch_size=self.config.data.image_load_size,
                                transforms=self.transforms,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"get_loaders: {execution_time:.6f} seconds")
        return train_loader, val_loader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()
        self.dir = dir
        self.lq_names = sorted(os.listdir(os.path.join(dir, 'LQ')))
        self.gt_names = sorted(os.listdir(os.path.join(dir, 'GT')))

        self._validate_dataset()
        
        self.patch_size = np.int32(patch_size)
        self.n = np.int32(n)
        self.parse_patches = parse_patches
        self.transforms = transforms

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = PIL.Image.open(f)
            return img.convert('RGB')

    def _validate_dataset(self):
        if len(self.lq_names) != len(self.gt_names):
            raise ValueError("LQ and GT dataset counts do not match")
        
        for lq, gt in zip(self.lq_names, self.gt_names):
            if not lq.startswith(gt.split('_')[0]):
                raise ValueError(f"File correspondence error: {lq} vs {gt}")

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), th, tw

        i_list = np.random.randint(0, h - th, size=n)
        j_list = np.random.randint(0, w - tw, size=n)
        return i_list, j_list, th, tw

    def __getitem__(self, index):
        lq_name = self.lq_names[index]
        gt_name = self.gt_names[index]
        
        lq_img = self._load_image(os.path.join(self.dir, 'LQ', lq_name))
        gt_img = self._load_image(os.path.join(self.dir, 'GT', gt_name))
        
        if self.parse_patches:
            i, j, h, w = self.get_params(lq_img, (self.patch_size, self.patch_size), self.n)
            
            lq_patches = [lq_img.crop((j[k], i[k], j[k]+w, i[k]+h)) for k in range(self.n)]
            gt_patches = [gt_img.crop((j[k], i[k], j[k]+w, i[k]+h)) for k in range(self.n)]
            
            return torch.stack(
                [torch.cat([self.transforms(lq), self.transforms(gt)], dim=0) 
                 for lq, gt in zip(lq_patches, gt_patches)]
            ), os.path.splitext(lq_name)[0]
        
        img_crop_size = 3072

        wd_new, ht_new = lq_img.size
        print("wd_new = {}, ht_new = {}".format(wd_new, ht_new))
        if ht_new > wd_new and ht_new > img_crop_size:
            wd_new = int(np.ceil(wd_new * img_crop_size / ht_new))
            ht_new = img_crop_size
        elif ht_new <= wd_new and wd_new > img_crop_size:
            ht_new = int(np.ceil(ht_new * img_crop_size / wd_new))
            wd_new = img_crop_size
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        lq_img = lq_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

        return torch.cat([self.transforms(lq_img), self.transforms(gt_img)], dim=0), lq_name

    def __len__(self):
        return len(self.lq_names)
