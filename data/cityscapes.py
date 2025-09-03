from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random


class CityScapes(Dataset):
    def __init__(self, 
                 root=None,
                 download=False,
                 split='val',
                 transform=None,
                 retname=True,
                 do_semseg=False, 
                 do_depth=False, 
                 ):
        
        self.root = os.path.expanduser(root)
        if download:
            raise NotImplementedError
        self.transform = transform
        self.retname = retname
        self.do_semseg = do_semseg
        self.do_depth = do_depth

        # read the data file
        if split=='train':
            self.data_path = root + '/train'
        if split=='val':
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))


    def __getitem__(self, index):
        x = {}

        # load data from the pre-processed npy files
        image = np.load(self.data_path + '/image/{:d}.npy'.format(index))
        x['image'] = np.array(image, dtype=np.float32)
        if self.do_semseg:
            # semantic = np.expand_dims(np.load(self.data_path + '/label_19/{:d}.npy'.format(index)), axis=0)
            semantic = np.moveaxis(np.expand_dims(np.load(self.data_path + '/label_19/{:d}.npy'.format(index)), axis=0), 0, -1)
            x['semseg'] = np.array(semantic, dtype=np.float32)
        if self.do_depth:
            depth = np.load(self.data_path + '/depth/{:d}.npy'.format(index))
            x['depth'] = np.array(depth, dtype=np.float32)

        if self.retname:
            x['meta'] = {'img_name': str(index),
                         'img_size': (image.shape[0], image.shape[1])}

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.data_len