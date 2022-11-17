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

    # train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

@register_dataset("FeatureFolder")
class FeatureFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

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
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        
        if self.transform:
            # print(self.transform(fmap).shape)  torch.Size([1, 256, 256, 256])
            return self.transform(from_numpy(np.load(self.samples[index])))
        return from_numpy(np.load(self.samples[index]))
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)

@register_dataset("FeatureFolderTest")
class FeatureFolderTest(Dataset):

    def __init__(self, root, split="test"):
        # splitdir = Path(root) / split

        # if not splitdir.is_dir():
        #     raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in Path(root).iterdir() if (f.is_file() and f.stem[1] != '6')]
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        fmap = from_numpy(np.load(self.samples[index]))
        head_tail = path.split(self.samples[index])
        return fmap, head_tail[1]
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)


@register_dataset("FeatureFolderTest4c")
class FeatureFolderTest4c(Dataset):

    def __init__(self, root, split="test"):
        splitdir = Path(root) / split

        # if not splitdir.is_dir():
        #     raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if (f.is_file() and f.stem[0] == 'p')]
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))
      

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float')).unsqueeze(0)
        if self.samples[index].stem[1] == '2':   # p2
            t = interpolate(t, scale_factor=0.5, mode='bicubic')
        head_tail = path.split(self.samples[index])
       # print(head_tail[1])
        return t, head_tail[1]
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)

@register_dataset("FeatureFolderTestNorm")
class FeatureFolderTestNorm(Dataset):

    def __init__(self, root, split="test"):
        # splitdir = Path(root) / split

        # if not splitdir.is_dir():
        #     raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in Path(root).iterdir() if (f.is_file() and f.stem[1] != '6')]
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = from_numpy(np.load(self.samples[index]))
        head_tail = path.split(self.samples[index])
        t = torch.clamp(t, min=-26.426828384399414, max=28.397470474243164)
        t = (t+26.426828384399414)/54.824298858642578
        return t, head_tail[1]
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

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

    def __init__(self, root, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float')).unsqueeze(0)
       # t = from_numpy(load(self.samples[index], allow_pickle=True))
        if self.samples[index].stem[1] == '2':   # p2
            t = interpolate(t, scale_factor=0.5, mode='bicubic')
        # elif self.samples[index].stem[1] == '4':   # p4
        #     t = interpolate(t, scale_factor=4, mode='bicubic')
        # elif self.samples[index].stem[1] == '5'   :     # p5
        #     t = interpolate(t, scale_factor=8, mode='bicubic')
        # if t.shape[2] == 256 and t.shape[3] == 256:
        #     return t.float()
        # if 64 < max(t.shape[2], t.shape[3]) <= 128:     # p3
        #     t = interpolate(t, scale_factor=2, mode='bicubic')
        # elif 32 < max(t.shape[2], t.shape[3]) <= 64:    # p4
        #     t = interpolate(t, scale_factor=4, mode='bicubic')
        # elif max(t.shape[2], t.shape[3]) <= 32:         # p5
        #     t = interpolate(t, scale_factor=8, mode='bicubic')

        # hpad, wpad = 256-t.shape[2], 256-t.shape[3]
        # padding = torch.nn.ReplicationPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
        
        return t.float()
        #print("x_hat: {}".format(x_hat[0, 1, 0, 0]))

        #return from_numpy(load(self.samples[index]))
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)

@register_dataset("FeatureFolderTiledNorm")
class FeatureFolderTiledNorm(Dataset):
    #FeatureMaps scaled & normalized
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        #self.norm = transforms.Normalize(mean, std)    

    def __getitem__(self, index):
        
        
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float')).unsqueeze(0).unsqueeze(0)
        
       # t = from_numpy(load(self.samples[index], allow_pickle=True))
        if self.samples[index].stem[1] == '2':   # p2
            t = interpolate(t, scale_factor=0.5, mode='bicubic')
        # elif self.samples[index].stem[1] == '4':   # p4
        #     t = interpolate(t, scale_factor=4, mode='bicubic')
        # elif self.samples[index].stem[1] == '5'   :     # p5
        #     t = interpolate(t, scale_factor=8, mode='bicubic')
        t = torch.clamp(t, min=-26.426828384399414, max=28.397470474243164)
        t = (t+26.426828384399414)/54.824298858642578
        t = torch.clamp(t, 0, 1)
        # normalize
        # scaling
        
        return t.float()
        #print("x_hat: {}".format(x_hat[0, 1, 0, 0]))

        #return from_numpy(load(self.samples[index]))
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)




@register_dataset("FeatureFolderNorm")
class FeatureFolderNorm(Dataset):
    #FeatureMaps scaled & normalized
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train"):
        
        self.samples = [f for f in Path(root).iterdir() if f.is_file()]
        #self.norm = transforms.Normalize(mean, std)    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float'))
       # t = from_numpy(load(self.samples[index], allow_pickle=True))
        # t = torch.clamp(t, min=-26.426828384399414, max=28.397470474243164)
        # t = (t+26.426828384399414)/54.824298858642578
        # normalize
        # scaling
        if t.shape[2] == 256 and t.shape[3] == 256:
            return t.float()
        if 64 < max(t.shape[2], t.shape[3]) <= 128:     # p3
            t = interpolate(t, scale_factor=2, mode='bicubic')
        elif 32 < max(t.shape[2], t.shape[3]) <= 64:    # p4
            t = interpolate(t, scale_factor=4, mode='bicubic')
        elif max(t.shape[2], t.shape[3]) <= 32:         # p5
            t = interpolate(t, scale_factor=8, mode='bicubic')

        hpad, wpad = 256-t.shape[2], 256-t.shape[3]
        padding = torch.nn.ReplicationPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
        
        # if torch.max(t) > 1 or torch.min(t) < 0:
        #     print("!!!!!!!!!! ERROR !!!!!!!!")
        #     print(self.samples[index])
      
        #return padding(t).type(torch.FloatTensor)
        head_tail = path.split(self.samples[index])
        return padding(t).type(torch.FloatTensor), head_tail[1]
        #print("x_hat: {}".format(x_hat[0, 1, 0, 0]))

        #return from_numpy(load(self.samples[index]))
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)


@register_dataset("FeatureFolderStd")
class FeatureFolderStd(Dataset):
    #FeatureMaps scaled & normalized
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train", mean_=-0.0961, std_=1.961142):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        #self.norm = transforms.Normalize(mean, std)    
        self.transforms = transforms.Normalize(mean_, std_)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        t = torch.as_tensor(np.load(self.samples[index], allow_pickle=True).astype('float'))
        t = self.transforms(t)
        # normalize
        # scaling
        if t.shape[2] == 256 and t.shape[3] == 256:
            return t.float()
        if 64 < max(t.shape[2], t.shape[3]) <= 128:     # p3
            t = interpolate(t, scale_factor=2, mode='bicubic')
        elif 32 < max(t.shape[2], t.shape[3]) <= 64:    # p4
            t = interpolate(t, scale_factor=4, mode='bicubic')
        elif max(t.shape[2], t.shape[3]) <= 32:         # p5
            t = interpolate(t, scale_factor=8, mode='bicubic')

        hpad, wpad = 256-t.shape[2], 256-t.shape[3]
        padding = torch.nn.ReplicationPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
        
        t = torch.clamp(t, min=0, max=1)
        if torch.max(t) > 1 or torch.min(t) < 0:
            print("!!!!!!!!!! ERROR !!!!!!!!")
            print(self.samples[index])
      
        return padding(t).float()
        #print("x_hat: {}".format(x_hat[0, 1, 0, 0]))

        #return from_numpy(load(self.samples[index]))
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)



@register_dataset("FeatureFolderGeneral")
class FeatureFolderGeneral(Dataset):

    def __init__(self, root, split="test"):
        splitdir = Path(root) / split
        # if not splitdir.is_dir():
        #     raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        x = from_numpy(np.load(self.samples[index]))
        filename = path.split(self.samples[index])
        if filename[1][1] == '3':
            x = interpolate(x, scale_factor=2, mode='bicubic')
        if filename[1][1] == '4':
            x = interpolate(x, scale_factor=4, mode='bicubic')
        if filename[1][1] == '5':
            x = interpolate(x, scale_factor=8, mode='bicubic')

        hpad, wpad = 256-x.shape[2], 256-x.shape[3]
        padding = torch.nn.ZeroPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
        x = padding(x)
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img
        return x

    def __len__(self):
        return len(self.samples)



# @register_dataset("FeatureFolderFname")
# class FeatureFolderFname(Dataset):

#     def __init__(self, root, split="test"):
#         splitdir = Path(root) / split
#         # if not splitdir.is_dir():
#         #     raise RuntimeError(f'Invalid directory "{root}"')

#         self.samples = [f for f in splitdir.iterdir() if f.is_file()]
#         # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
#         """
#         x = from_numpy(load(self.samples[index]))
#         filename = path.split(self.samples[index])
#         if filename[1][1] == '3':
#             x = interpolate(x, scale_factor=2, mode='bicubic')
#         if filename[1][1] == '4':
#             x = interpolate(x, scale_factor=4, mode='bicubic')
#         if filename[1][1] == '5':
#             x = interpolate(x, scale_factor=8, mode='bicubic')

#         hpad, wpad = 256-x.shape[2], 256-x.shape[3]
#         padding = torch.nn.ZeroPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
#         x = padding(x)
#         # img = Image.open(self.samples[index]).convert("RGB")
#         # if self.transform:
#         #     return self.transform(img)
#         # return img
#         return x, int(filename[1][1])

#     def __len__(self):
#         return len(self.samples)




# @register_dataset("YUVfolder")
# class YUVfolder(Dataset):

#     def __init__(self, root, split="test"):
#         splitdir = Path(root) / split

#         # if not splitdir.is_dir():
#         #     raise RuntimeError(f'Invalid directory "{root}"')

#         self.samples = [f for f in splitdir.iterdir() if f.is_file()]
#         # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
#         """
#         x = from_numpy(load(self.samples[index]))
#         filename = path.split(self.samples[index])
#         if filename[1][1] == '3':
#             x = interpolate(x, scale_factor=2, mode='bicubic')
#         if filename[1][1] == '4':
#             x = interpolate(x, scale_factor=4, mode='bicubic')
#         if filename[1][1] == '5':
#             x = interpolate(x, scale_factor=8, mode='bicubic')

#         hpad, wpad = 256-x.shape[2], 256-x.shape[3]
#         padding = torch.nn.ZeroPad2d((math.ceil(wpad/2),math.floor(wpad/2), math.ceil(hpad/2), math.floor(hpad/2)))
#         x = padding(x)
#         # img = Image.open(self.samples[index]).convert("RGB")
#         # if self.transform:
#         #     return self.transform(img)
#         # return img
#         return x

#     def __len__(self):
#         return len(self.samples)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor