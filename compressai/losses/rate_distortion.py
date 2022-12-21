# Copyright (c) 2021-2022, InterDigital Communications, Inc, Curie Yoon
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

import math

import torch
import torch.nn as nn

from compressai.registry import register_criterion
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        mse_element = torch.square(output["x_hat"]-target)
        lambda_element = torch.sigmoid((mse_element-1)/0.05)
        mse_element = mse_element * lambda_element
        # out["mse_loss"] = self.mse(output["x_hat"], target)
        out["lambda"] = torch.mean(lambda_element)
        out["mse_loss"] = torch.mean(mse_element)
        # out["loss"] = self.lmbda * lambda_element * mse_element + out["bpp_loss"]
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        
        bins = np.linspace(0, 5, 3000)
        print("mse: {}".format(np.mean(x-y)**2))
        plt.hist(torch.flatten(mse_element), bins, alpha=0.5, label='MSE')
        plt.hist(torch.flatten(lambda_element), bins, alpha=0.5, label='lambda')
        plt.legend(loc='upper right')
        now = datetime.now()

        current_time =  now.strftime("%Y-%m-%d_%H;%M;%S")
        plt.title(current_time)
        plt.savefig('/home/porsche/curie/neural-featuremap-compressor/viz/{}.png'.format(current_time))

        return out

class WarpedRDLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        #  print("x_hat shape: {}".format(output["x_hat"].shape))
       # print("target_size: {}".format(target.shape))
       # print("mse type: {}".format(type(out["mse_loss"])))

       # squaredloss = torch.square(output["x_hat"]-target)
       # print("mse elementwise : {}\n{}".format(squaredloss.shape, squaredloss))
        # 3.4888e-04, 2.3032e-04]]]], device='cuda:0', grad_fn=<PowBackward0>)
        # x_hat shape: torch.Size([4, 256, 256, 256])
        # target_size: torch.Size([4, 256, 256, 256])
        # mse type: <class 'torch.Tensor'>
        # mse elementwise : torch.Size([4, 256, 256, 256])
        # x_hat shape: torch.Size([4, 256, 256, 256])
        # target_size: torch.Size([4, 256, 256, 256])
        # mse type: <class 'torch.Tensor'>
        # mse elementwise : torch.Size([4, 256, 256, 256])
        # tensor([[[[5.6712e+00, 1.6933e+00, 2.1460e+00,  ..., 
        squaredloss = self.mse(output["x_hat"], target)#torch.square(output["x_hat"]-target)
        # if mse < 2:
        #     #mse = 1.5011 + 20/(20+math.exp(13-5*mse))
        #     # mse = 1.80017092764013109005821404764048191+20/(80+math.exp(13-5*mse))
        out["mse_loss"] = squaredloss/(1+math.exp(50-55*squaredloss))
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out
