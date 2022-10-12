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
from numpy import load
from torch import from_numpy
from torch import squeeze
from compressai.registry import register_dataset
from torchvision.transforms import transforms


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
        # print("self.samples[0]: {}, {}".format(type(self.samples[0]), self.samples[0]))

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        fmap = from_numpy(load(self.samples[index], allow_pickle=True))
        if self.transform:
           # print(self.transform(fmap).shape)
            return squeeze(self.transform(fmap),axis=0)
        return fmap
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)



@register_dataset("P5train")
class P5train(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        # import glob
        # # All files and directories ending with .txt and that don't begin with a dot:
        # print(glob.glob("/home/adam/*.txt")) 
        # # All files and directories ending with .txt with depth of 2 folders, ignoring names beginning with a dot:
        # print(glob.glob("/home/adam/*/*.txt")) 
        import glob
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        self.samples = []
        for file in glob.glob("/data/curieyoon/pseudo/train/p5_*.npy"):
            self.samples.append(file)
        #self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        print("first: {}, {}\nsecond: {}\n".format(type(self.samples[0]), self.samples[0], self.samples[1]))
        self.transforms = transforms.Compose(
              [transforms.RandomCrop(25)]
        )

    def __getitem__(self, index):
        fmap = load(self.samples[index], allow_pickle=True)
        if self.transform:
             return self.transform(fmap)
        return fmap
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)

@register_dataset("P5test")
class P5test(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        # import glob
        # # All files and directories ending with .txt and that don't begin with a dot:
        # print(glob.glob("/home/adam/*.txt")) 
        # # All files and directories ending with .txt with depth of 2 folders, ignoring names beginning with a dot:
        # print(glob.glob("/home/adam/*/*.txt")) 
        import glob
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        self.samples = []
        for file in glob.glob("/data/curieyoon/pseudo/test/p5_*.npy"):
            self.samples.append(file)

        #self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        fmap = load(self.samples[index])
        if self.transform:
             return self.transform(fmap)
        return fmap
        # img = Image.open(self.samples[index]).convert("RGB")
        # if self.transform:
        #     return self.transform(img)
        # return img

    def __len__(self):
        return len(self.samples)


