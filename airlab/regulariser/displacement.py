# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pdb
import torch as th
import torch.nn.functional as F

import numpy as np

# Regulariser base class (standard from PyTorch)
class _Regulariser(th.nn.modules.Module):
    def __init__(self, pixel_spacing, size_average=True, reduce=True, weight=1, sample_weight=None):
        super(_Regulariser, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._weight = weight
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self.name = "parent"
        self._mask = None
        self._sample_weight = sample_weight

    def SetWeight(self, weight):
        print("SetWeight is deprecated. Use set_weight instead.")
        self.set_weight(weight)

    def set_weight(self, weight):
        self._weight = weight

    def set_mask(self, mask):
        self._mask = mask

    def _mask_2d(self, df):
        if not self._mask is None:
            nx, ny, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny].unsqueeze(-1).repeat(1,1,d)
        else:
            return df

    def _mask_3d(self, df):
        if not self._mask is None:
            nx, ny, nz, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny,:nz].unsqueeze(-1).repeat(1,1,1,d)
        else:
            return df

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            if self._sample_weight is not None:
                w = th.ones(tensor.shape[0])
                w[self._sample_weight] = 0.1
                w = w.to(tensor.device)
                tensor = th.mean(tensor,(1,2)) * w
            tensor = self._weight*tensor.mean()
            return tensor
        if not self._size_average and self._reduce:
            return self._weight*tensor.sum()
        if not self._reduce:
            return self._weight*tensor

"""
    Isotropic TV regularisation
"""
class IsotropicTVRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(IsotropicTVRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "isoTV"

        if self._dim == 2:
            self._regulariser = self._isotropic_TV_regulariser_2d # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._isotropic_TV_regulariser_3d # 3d regularisation

    def _isotropic_TV_regulariser_2d(self, displacement):
        dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2)*self._pixel_spacing[0]
        dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2)*self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0,1,0,1)))

    def _isotropic_TV_regulariser_3d(self, displacement):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2)*self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2)*self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2)*self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0,1,0,1,0,1)))

    def forward(self, displacement):

        # set the supgradient to zeros
        value = self._regulariser(displacement)
        mask = value > 0
        value[mask] = th.sqrt(value[mask])

        return self.return_loss(value)

"""
    TV regularisation 
"""
class TVRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(TVRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "TV"

        if self._dim == 2:
            self._regulariser = self._TV_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._TV_regulariser_3d  # 3d regularisation

    def _TV_regulariser_2d(self, displacement):
        dx = th.abs(displacement[1:, 1:, :] - displacement[:-1, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, :] - displacement[1:, :-1, :])*self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _TV_regulariser_3d(self, displacement):
        dx = th.abs(displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :])*self._pixel_spacing[0]
        dy = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :])*self._pixel_spacing[1]
        dz = th.abs(displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :])*self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)))

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))


"""
    Diffusion regularisation 
"""
class DiffusionRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(DiffusionRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "L2"

        if self._dim == 2:
            self._regulariser = self._l2_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._l2_regulariser_3d  # 3d regularisation

    def _l2_regulariser_2d(self, displacement):
        dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2) * self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _l2_regulariser_3d(self, displacement):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2) * self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2) * self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)))

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))


"""
    Sparsity regularisation 
"""
class SparsityRegulariser(_Regulariser):
    def __init__(self, size_average=True, reduce=True):
        super(SparsityRegulariser, self).__init__([0], size_average, reduce)

    def forward(self, displacement):
        return self.return_loss(th.abs(displacement))




class SmoothnesRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True, weight=1, sample_weights=None):
        super(SmoothnesRegulariser, self).__init__(pixel_spacing,
                                                   size_average, 
                                                   reduce, 
                                                   weight, 
                                                   th.tensor(sample_weights))

        self.name = "L2"
        if self._dim == 2:
            self._regulariser = self._l2_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._l2_regulariser_3d  # 3d regularisation

    def _l2_regulariser_2d(self, dp):
        dp = th.stack(dp)
        # interpolate grid between anchors
        # select pairs by index
        dp_interp = []
        for i in range(len(self._sample_weight)-1):
            idx0 = self._sample_weight[i]
            idx1 = self._sample_weight[i+1]
            g = th.stack((dp[idx0], dp[idx1])).squeeze()
            gp = g.permute(1,2,0,3) # [w,h,b,c]
            t = (idx1-idx0).cpu().detach().numpy()[0]
            g_interp = F.interpolate(gp, (t,2), mode='bilinear')
            dp_interp.append(g_interp.permute(2,0,1,3))
        dp_interp = th.cat(dp_interp)

        # shape [b, h, w, f]
        # duplicate corner points
        dx = (dp[1:,:,:,0] - dp[:-1,:,:,0]).pow(2) * self._pixel_spacing[0]
        dy = (dp[1:,:,:,1] - dp[:-1,:,:,1]).pow(2) * self._pixel_spacing[1]

        l2 = self._mask_2d(F.pad(dx + dy, (0,1, 0,1, 0,1)))
        l2 = th.sqrt(l2)
        return l2

    def _l2_regulariser_3d(self, displacement):
        raise NotImplementedError()

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))



class NeighbourSmoothnesRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True, weight=1):
        super(NeighbourSmoothnesRegulariser, self).__init__(pixel_spacing, size_average, reduce, weight)

        self.name = "L2"
        if self._dim == 2:
            self._regulariser = self._l2_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._l2_regulariser_3d  # 3d regularisation

    def _l2_regulariser_2d(self, dp):
        dp = th.stack(dp)
        
        # shape [b, h, w, f]
        dx = (dp[0:-2,:,:,0] - dp[1:-1,:,:,0]) * self._pixel_spacing[0]
        dy = (dp[0:-2,:,:,1] - dp[1:-1,:,:,1]) * self._pixel_spacing[1]


        l2 = self._mask_2d(F.pad(dx.pow(2) + dy.pow(2), (0, 1, 0, 1)))
        l2 = th.sqrt(l2)
        
        #l2 = th.mean(l2, (1,2)) * self._sample_weights.to(l2.device)
        return l2

    def _l2_regulariser_3d(self, displacement):
        raise NotImplementedError()

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))

