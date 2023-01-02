# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch import from_numpy
from os import path
from compressai.registry import register_dataset
from torchvision.transforms import transforms
from torch.nn.functional import interpolate
import torch
import math
from compressai.datasets import tiling
import random

@register_dataset("ImageFolder")
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        # print(img)
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


@register_dataset("FeatureFolder256_to4")
class FeatureFolder256_to4(Dataset):
    """Load an feature map folder database. 

    Args:
        root (string): root directory of the dataset
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if (f.is_file() and f.stem[1] != '6')]
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            t (Tensor), path.split (String):
                4-channel feature map, filename of the feature map
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float'))
        
        if 128 < max(t.shape[2], t.shape[3]) <= 256:    # p2
            hpad, wpad = 256-t.shape[2], 256-t.shape[3]
        elif 64 < max(t.shape[2], t.shape[3]) <= 128:     # p3
            hpad, wpad = 128-t.shape[2], 128-t.shape[3]
        elif 32 < max(t.shape[2], t.shape[3]) <= 64:    # p4
            hpad, wpad = 64-t.shape[2], 64-t.shape[3]
        elif max(t.shape[2], t.shape[3]) <= 32:         # p5
            hpad, wpad = 32-t.shape[2], 32-t.shape[3]

        padding = torch.nn.ZeroPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
        # 1, 256, 256, 256
        t = padding(t)
        t = tiling.tile_256_to_4_torch(t.squeeze(0)).unsqueeze(0) # 1, 4, 16h, 16w

        if self.samples[index].stem[1] == '2':   # p2
            t = interpolate(t, scale_factor=0.5, mode='bicubic')
        
        return t.float(), path.split(self.samples[index])[1]
        
    def __len__(self):
        return len(self.samples)


@register_dataset("FeatureFolderScale")
class FeatureFolderScale(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train", downsize=False, crop=False, cropsize=64):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if (f.is_file() and f.stem[1] != '6')]
        self.downsize = downsize
        self.crop = crop
        self.cropsize = cropsize

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float'))
        if t.dim() == 3:
            t = t.unsqueeze(0)
        assert t.shape[0] == 1 and t.shape[1] == 4
        if self.crop:
            tt = torch.empty((1, 4, self.cropsize, self.cropsize))
            r = random.randint(0, t.shape[2]-self.cropsize-1)
            o = random.randint(0, t.shape[2]-self.cropsize-1)
            tt = t[:, :, r:r+self.cropsize, o:o+self.cropsize]
            print("r={}, o={}, tt={}".format(r,o,tt.shape))
            return tt.float()

        if self.downsize:
            if self.samples[index].stem[1] == '2':   # p2
                t = interpolate(t, scale_factor=0.25, mode='bicubic')
            elif self.samples[index].stem[1] == '3':   # p2
                t = interpolate(t, scale_factor=0.5, mode='bicubic')
        else:
            if self.samples[index].stem[1] == '2':   # p2
                t = interpolate(t, scale_factor=0.5, mode='bicubic')
            
        

        return t.float()

    def __len__(self):
        return len(self.samples)



@register_dataset("FeatureFolderPad")
class FeatureFolderPad(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train", crop=None, pad=384, eval=False, scale=None):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        self.scale = scale # '2' , '3', '4', '5'
        if self.scale is None:
            self.samples = [f for f in splitdir.iterdir() if (f.is_file() and f.stem[1] != '6')]
        else:
            self.samples = [f for f in splitdir.iterdir() if (f.is_file() and f.stem[1] == self.scale)]
        self.crop = crop
        self.pad = pad
        self.eval = eval
        self.split = split
      

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float'))
        if t.dim() == 3:
            t = t.unsqueeze(0)
        h, w = t.shape[2], t.shape[3]
        if self.samples[index].stem[1] == '2':  # p2
            hpad, wpad = self.pad-h, self.pad-w
        elif self.samples[index].stem[1] == '3':    # p3
            hpad, wpad = self.pad/2-h, self.pad/2-w
        elif self.samples[index].stem[1] == '4':    # p4
            hpad, wpad = self.pad/4-h, self.pad/4-w
        elif self.samples[index].stem[1] == '5': # p5
            hpad, wpad = self.pad/8-h, self.pad/8-w
        padding = torch.nn.ZeroPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
            
        t = padding(t).squeeze(0)
        # print("HERE: {}".format(t.shape))
        # # 1, 256, 384, 384
        # print(self.samples[index].stem)
        t = feature_rearrange_torch_16(t).unsqueeze(0) # 16, 384*4, 384*4
        assert t.shape[0] == 1 and t.shape[1] == 16
        if self.samples[index].stem[1] == '5':   # p5
            t = interpolate(t, scale_factor=2, mode='bicubic')
        print("{}: {}".format(self.samples[index], t.shape))
        if self.crop is not None and self.split != 'test':
            tt = torch.empty((1, 16, self.crop, self.crop))
            r = random.randint(0, t.shape[1]-self.crop-1)
            o = random.randint(0, t.shape[2]-self.crop-1)
            tt = t[:, :, r:r+self.crop, o:o+self.crop]
            return tt.float()

        if self.crop is None and self.samples[index].stem[1] == '2':   # p2
            t = interpolate(t, scale_factor=0.5, mode='bicubic')
        
        
        if self.eval:
            return t.float(), h, w, self.samples[index].stem # p2_xxxx (without extension)
        else:
            return t.float()

    def __len__(self):
        return len(self.samples)

        ###################
def feature_rearrange_torch_16(feature): ## 256, 4, 4 ->  16, 16, 16
                                        ## 256, 3, 3 -> 16, 12, 12
    h,w = feature.shape[1],feature.shape[2]
    featuremap = torch.zeros((16, 4*h,4*w))
    for c in range(16):
        for i in range(4):
            for j in range(4):
                c_num = i*4+j
                featuremap[c, i*h:(i+1)*h,j*w:(j+1)*w] = feature[c*16 + c_num,:,:]
    return featuremap

