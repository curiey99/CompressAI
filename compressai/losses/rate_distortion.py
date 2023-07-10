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
# from datetime import datetime
# import matplotlib.pyplot as plt
# import numpy as np

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
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

@register_criterion("FusionRDLoss")
class FusionRDLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target): 
        # out_net,                                  , d
        # dict{"features"(list), "likelihoods"}     , list[p2~p5]
        out = {}
        num_pixels = 0
        
        for p in target:
            N, _, H, W = p.size()
            num_pixels += N * H * W
            

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )


        out["mse_loss"] = self.mse(output["features"][0], target[0]) + self.mse(output["features"][1], target[1]) + self.mse(output["features"][2], target[2]) + self.mse(output["features"][3], target[3])

        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


@register_criterion("FusionRDLoss_P")
class FusionRDLoss_P(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target): 
        # out_net,                                  , d
        # dict{"features"(list), "likelihoods"}     , list[p2~p5]
        out = {}
        num_pixels = 0
        
        for p in target:
            N, _, H, W = p.size()
            num_pixels += N * H * W
            

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.mse(output["features"][0], target[0]) + self.mse(output["features"][1], target[1]) + self.mse(output["features"][2], target[2]) + self.mse(output["features"][3], target[3])

        out["p2_mse"] = (torch.square(output["features"][0] - target[0])).mean().item()
        out["p3_mse"] = (torch.square(output["features"][1] - target[1])).mean().item()
        out["p4_mse"] = (torch.square(output["features"][2] - target[2])).mean().item()
        out["p5_mse"] = (torch.square(output["features"][3] - target[3])).mean().item()
      
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


@register_criterion("FusionWarpedLoss")
class FusionWarpedLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, alpha=1.0, beta=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target): 
        # out_net,                                  , d
        # dict{"features"(list), "likelihoods"}     , list[p2~p5]
        out = {}
        num_pixels = 0
        
        for p in target:
            N, _, H, W = p.size()
            num_pixels += N * H * W
            

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        p2_mse = torch.square(output["features"][0] - target[0])
        p3_mse = torch.square(output["features"][1] - target[1])
        p4_mse = torch.square(output["features"][2] - target[2])
        p5_mse = torch.square(output["features"][3] - target[3])



        out["p2_mseloss"] = p2_mse * torch.sigmoid((p2_mse-self.alpha)/self.beta)
        out["p3_mseloss"] = p3_mse * torch.sigmoid((p3_mse-self.alpha)/self.beta)
        out["p4_mseloss"] = p4_mse * torch.sigmoid((p4_mse-self.alpha)/self.beta)
        out["p5_mseloss"] = p5_mse * torch.sigmoid((p5_mse-self.alpha)/self.beta)
        


        out["p2_mse"] = p2_mse.mean().item()
        out["p3_mse"] = p3_mse.mean().item()
        out["p4_mse"] = p4_mse.mean().item()
        out["p5_mse"] = p5_mse.mean().item()
        out["mse_loss"] = torch.mean(out["p2_mseloss"]) + torch.mean(out["p3_mseloss"]) + torch.mean(out["p4_mseloss"]) + torch.mean(out["p5_mseloss"])

        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


@register_criterion("FusionWarpedLoss_Pwise")
class FusionWarpedLoss_Pwise(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, 
        p2_alpha=1.1, p2_beta=0.005,
        p3_alpha=0.48, p3_beta=0.005,
        p4_alpha=0.35, p4_beta=0.005,
        p5_alpha=1.3, p5_beta=0.005,
        alpha_w=1.0, beta_w=1.0
        ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

        self.p2_alpha, self.p2_beta = p2_alpha * alpha_w, p2_beta * beta_w
        self.p3_alpha, self.p3_beta = p3_alpha * alpha_w, p3_beta * beta_w
        self.p4_alpha, self.p4_beta = p4_alpha * alpha_w, p4_beta * beta_w
        self.p5_alpha, self.p5_beta = p5_alpha * alpha_w, p5_beta * beta_w
        

    def forward(self, output, target): 
        # out_net,                                  , d
        # dict{"features"(list), "likelihoods"}     , list[p2~p5]
        out = {}
        num_pixels = 0
        
        for p in target:
            N, _, H, W = p.size()
            num_pixels += N * H * W
            

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        p2_mse = torch.square(output["features"][0] - target[0])
        p3_mse = torch.square(output["features"][1] - target[1])
        p4_mse = torch.square(output["features"][2] - target[2])
        p5_mse = torch.square(output["features"][3] - target[3])

        # out["p2_lambda"] = torch.sigmoid((p2_mse-self.alpha)/self.beta)
        # out["p3_lambda"] = torch.sigmoid((p3_mse-self.alpha)/self.beta)
        # out["p4_lambda"] = torch.sigmoid((p4_mse-self.alpha)/self.beta)
        # out["p5_lambda"] = torch.sigmoid((p5_mse-self.alpha)/self.beta)
        


        out["p2_mseloss"] = p2_mse * torch.sigmoid((p2_mse-self.p2_alpha)/self.p2_beta)
        out["p3_mseloss"] = p3_mse * torch.sigmoid((p3_mse-self.p3_alpha)/self.p3_beta)
        out["p4_mseloss"] = p4_mse * torch.sigmoid((p4_mse-self.p4_alpha)/self.p4_beta)
        out["p5_mseloss"] = p5_mse * torch.sigmoid((p5_mse-self.p5_alpha)/self.p5_beta)
        


        out["p2_mse"] = p2_mse.mean().item()
        out["p3_mse"] = p3_mse.mean().item()
        out["p4_mse"] = p4_mse.mean().item()
        out["p5_mse"] = p5_mse.mean().item()
        out["mse_loss"] = torch.mean(out["p2_mseloss"]) + torch.mean(out["p3_mseloss"]) + torch.mean(out["p4_mseloss"]) + torch.mean(out["p5_mseloss"])

        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


@register_criterion("WarpedRDLoss")
class WarpedRDLoss(nn.Module):
    def __init__(self, lmbda=1e-2, alpha=1, beta=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.alpha = alpha
        self.beta = beta
        


    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        mse_element = torch.square(output["x_hat"]-target)
        out["lambda_element"] = torch.sigmoid((mse_element-self.alpha)/self.beta)
        out["mse_element"] = mse_element * out["lambda_element"]
        out["lambda"] = torch.mean(out["lambda_element"])
        out["mse_loss"] = torch.mean(out["mse_element"])
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


@register_criterion("SpatialMedoLoss")
class SpatialMedoLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, mask_coef=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.mask_coef = mask_coef

    def forward(self, output, target, mask): 

        out = {}
        num_pixels = 0
        
        for p in target:
            N, _, H, W = p.size()
            num_pixels += N * H * W
            

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )


        p2_mse = torch.square(output["features"][0] - target[0])
        p3_mse = torch.square(output["features"][1] - target[1])
        p4_mse = torch.square(output["features"][2] - target[2])
        p5_mse = torch.square(output["features"][3] - target[3])

        
       
        p2_mask = 1.0 - ((1.0 - mask) * self.mask_coef)

       
        p2_mask = torch.clamp(p2_mask, min=0.000000001, max=1.0)
        p2_mask = p2_mask / torch.max(p2_mask)
       

        p3_mask = torch.nn.functional.interpolate(p2_mask, scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True)
        p4_mask = torch.nn.functional.interpolate(p2_mask, scale_factor=0.25, mode='bilinear', align_corners=False, antialias=True)
        p5_mask = torch.nn.functional.interpolate(p2_mask, scale_factor=0.125, mode='bilinear', align_corners=False, antialias=True)
        
       
        p3_mask = torch.clamp(p3_mask, min=0.00000001, max=1.0)
        p3_mask = p3_mask / torch.max(p3_mask)
        # e.g (0.3, 0.8) -> (0.375, 1.0)

       
        p4_mask = torch.clamp(p4_mask, min=0.00000001, max=1.0)
        p4_mask = p4_mask / torch.max(p4_mask)
        
        p5_mask = torch.clamp(p5_mask, min=0.00000001, max=1.0)
        p5_mask = p5_mask / torch.max(p5_mask)

        

        out["p2_mseloss"] = p2_mse * p2_mask
        
        out["p3_mseloss"] = p3_mse * p3_mask
        
        out["p4_mseloss"] = p4_mse * p4_mask
        
        out["p5_mseloss"] = p5_mse * p5_mask
        
        out["mse_loss"] = torch.mean(out["p2_mseloss"]) + torch.mean(out["p3_mseloss"]) + torch.mean(out["p4_mseloss"]) + torch.mean(out["p5_mseloss"])   


        out["p2_mse"] = p2_mse.mean().item()
        out["p3_mse"] = p3_mse.mean().item()
        out["p4_mse"] = p4_mse.mean().item()
        out["p5_mse"] = p5_mse.mean().item()
        
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        return out


