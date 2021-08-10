import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os


class VideoSameSizeDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoSameSizeDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}

        if opt['testing_dir'] is not None:
            testing_dir = opt['testing_dir']
            testing_dir = testing_dir.split(',')
        else:
            testing_dir = []
        print('testing_dir', testing_dir)

        # read data:
        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_GT)

            if not(subfolder_name in testing_dir):
                continue

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)

            img_paths_LQ = img_paths_LQ[0:30]
            img_paths_GT = img_paths_GT[0:30]

            max_idx = len(img_paths_LQ)
            assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
            self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                           padding=self.opt['padding'])
        imgs_LQ_path = []
        for mm in range(len(select_idx)):
            imgs_LQ_path.append(self.imgs_LQ[folder][select_idx[mm]])
        img_GT_path = self.imgs_GT[folder][idx:idx+1]
        imgs_LQ = util.read_img_seq2(imgs_LQ_path, self.opt['train_size'])
        img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
        img_GT = img_GT[0]

        img_LQ_l = list(imgs_LQ.unbind(0))

        return {
            'LQs': torch.stack(img_LQ_l),  # shape: [N, C, H, W]
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
