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

import torch as th
import torch.nn.functional as F

import numpy as np

from .. import transformation as T
from ..transformation import utils as tu
from ..utils import kernelFunction as utils
from ..utils.image import toNP, get_mask



# Loss base class (standard from PyTorch)
class _PairwiseImageLoss(th.nn.modules.Module):
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(_PairwiseImageLoss, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._name = "parent"

        self._warped_moving_image = None
        self._warped_moving_mask = None
        self._weight = 1

        self._moving_image = moving_image
        self._moving_mask = moving_mask
        self._fixed_image = fixed_image
        self._fixed_mask = fixed_mask
        self._grid = None

        assert self._moving_image != None and self._fixed_image != None
        # TODO allow different image size for each image in the future
        assert self._moving_image.size == self._fixed_image.size
        assert self._moving_image.device == self._fixed_image.device
        assert len(self._moving_image.size) == 2 or len(self._moving_image.size) == 3

        self._grid = T.utils.compute_grid(self._moving_image.size, dtype=self._moving_image.dtype,
                                     device=self._moving_image.device)

        self._dtype = self._moving_image.dtype
        self._device = self._moving_image.device

    @property
    def name(self):
        return self._name

    def GetWarpedImage(self):
        return self._warped_moving_image[0, 0, ...].detach().cpu()

    def GetCurrentMask(self, displacement):
        """
        Computes a mask defining if pixels are warped outside the image domain, or if they fall into
        a fixed image mask or a warped moving image mask.
        return (Tensor): maks array
        """
        # exclude points which are transformed outside the image domain
        mask = th.zeros_like(self._fixed_image.image, dtype=th.uint8, device=self._device)
        for dim in range(displacement.size()[-1]):
            mask += displacement[..., dim].gt(1) + displacement[..., dim].lt(-1)

        mask = mask == 0

        # and exclude points which are masked by the warped moving and the fixed mask
        if not self._moving_mask is None:
            self._warped_moving_mask = F.grid_sample(self._moving_mask.image, displacement)
            self._warped_moving_mask = self._warped_moving_mask >= 0.5

            # if either the warped moving mask or the fixed mask is zero take zero,
            # otherwise take the value of mask
            if not self._fixed_mask is None:
                mask = th.where(((self._warped_moving_mask == 0) | (self._fixed_mask == 0)), th.zeros_like(mask), mask)
            else:
                mask = th.where((self._warped_moving_mask == 0), th.zeros_like(mask), mask)

        return mask

    def set_loss_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return tensor.mean()*self._weight
        if not self._size_average and self._reduce:
            return tensor.sum()*self._weight
        if not self.reduce:
            return tensor*self._weight


class MSE(_PairwiseImageLoss):
    r""" The mean square error loss is a simple and fast to compute point-wise measure
    which is well suited for monomodal image registration.

    .. math::
         \mathcal{S}_{\text{MSE}} := \frac{1}{\vert \mathcal{X} \vert}\sum_{x\in\mathcal{X}}
          \Big(I_M\big(x+f(x)\big) - I_F\big(x\big)\Big)^2

    Args:
        fixed_image (Image): Fixed image for the registration
        moving_image (Image): Moving image for the registration
        size_average (bool): Average loss function
        reduce (bool): Reduce loss function to a single value

    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(MSE, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._name = "mse"

        self.warped_moving_image = None

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(MSE, self).GetCurrentMask(displacement)

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        # compute squared differences
        value = (self.warped_moving_image - self._fixed_image.image).pow(2)

        # mask values
        value = th.masked_select(value, mask)

        return self.return_loss(value)


class NCC(_PairwiseImageLoss):
    r""" The normalized cross correlation loss is a measure for image pairs with a linear
         intensity relation.

        .. math::
            \mathcal{S}_{\text{NCC}} := \frac{\sum I_F\cdot (I_M\circ f)
                   - \sum\text{E}(I_F)\text{E}(I_M\circ f)}
                   {\vert\mathcal{X}\vert\cdot\sum\text{Var}(I_F)\text{Var}(I_M\circ f)}


        Args:
            fixed_image (Image): Fixed image for the registration
            moving_image (Image): Moving image for the registration

    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None):
        super(NCC, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, False, False)

        self._name = "ncc"

        self.warped_moving_image = th.empty_like(self._moving_image.image, dtype=self._dtype, device=self._device)

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(NCC, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        moving_image_valid = th.masked_select(self._warped_moving_image, mask)
        fixed_image_valid = th.masked_select(self._fixed_image.image, mask)


        value = -1.*th.sum((fixed_image_valid - th.mean(fixed_image_valid))*(moving_image_valid - th.mean(moving_image_valid)))\
                /th.sqrt(th.sum((fixed_image_valid - th.mean(fixed_image_valid))**2)*th.sum((moving_image_valid - th.mean(moving_image_valid))**2) + 1e-10)

        return value


"""
    Local Normaliced Cross Corelation Image Loss
"""
class LCC(_PairwiseImageLoss):
    def __init__(self, fixed_image, moving_image,fixed_mask=None, moving_mask=None, sigma=3, kernel_type="box", size_average=True, reduce=True):
        super(LCC, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask,  size_average, reduce)

        self._name = "lcc"
        self.warped_moving_image = th.empty_like(self._moving_image.image, dtype=self._dtype, device=self._device)
        self._kernel = None

        dim = len(self._moving_image.size)
        sigma = np.array(sigma)

        if sigma.size != dim:
            sigma_app = sigma[-1]
            while sigma.size != dim:
                sigma = np.append(sigma, sigma_app)

        if kernel_type == "box":
            kernel_size = sigma*2 + 1
            self._kernel = th.ones(*kernel_size.tolist(), dtype=self._dtype, device=self._device) \
                           / float(np.product(kernel_size)**2)
        elif kernel_type == "gaussian":
            self._kernel = utils.gaussian_kernel(sigma, dim, asTensor=True, dtype=self._dtype, device=self._device)

        self._kernel.unsqueeze_(0).unsqueeze_(0)

        if dim == 2:
            self._lcc_loss = self._lcc_loss_2d  # 2d lcc

            self._mean_fixed_image = F.conv2d(self._fixed_image.image, self._kernel)
            self._variance_fixed_image = F.conv2d(self._fixed_image.image.pow(2), self._kernel) \
                                         - (self._mean_fixed_image.pow(2))
        elif dim == 3:
            self._lcc_loss = self._lcc_loss_3d  # 3d lcc

            self._mean_fixed_image = F.conv3d(self._fixed_image.image, self._kernel)
            self._variance_fixed_image = F.conv3d(self._fixed_image.image.pow(2), self._kernel) \
                                         - (self._mean_fixed_image.pow(2))


    def _lcc_loss_2d(self, warped_image, mask):


        mean_moving_image = F.conv2d(warped_image, self._kernel)
        variance_moving_image = F.conv2d(warped_image.pow(2), self._kernel) - (mean_moving_image.pow(2))

        mean_fixed_moving_image = F.conv2d(self._fixed_image.image * warped_image, self._kernel)

        cc = (mean_fixed_moving_image - mean_moving_image*self._mean_fixed_image)**2 \
             / (variance_moving_image*self._variance_fixed_image + 1e-10)

        mask = F.conv2d(mask, self._kernel)
        mask = mask == 0

        return -1.0*th.masked_select(cc, mask)

    def _lcc_loss_3d(self, warped_image, mask):

        mean_moving_image = F.conv3d(warped_image, self._kernel)
        variance_moving_image = F.conv3d(warped_image.pow(2), self._kernel) - (mean_moving_image.pow(2))

        mean_fixed_moving_image = F.conv3d(self._fixed_image.image * warped_image, self._kernel)

        cc = (mean_fixed_moving_image - mean_moving_image*self._mean_fixed_image)**2\
             /(variance_moving_image*self._variance_fixed_image + 1e-10)

        mask = F.conv3d(mask, self._kernel)
        mask = mask == 0

        return -1.0 * th.masked_select(cc, mask)

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(LCC, self).GetCurrentMask(displacement)
        mask = 1-mask
        mask = mask.to(dtype=self._dtype, device=self._device)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        return self.return_loss(self._lcc_loss(self._warped_moving_image, mask))


class MI(_PairwiseImageLoss):
    r""" Implementation of the Mutual Information image loss.

         .. math::
            \mathcal{S}_{\text{MI}} := H(F, M) - H(F|M) - H(M|F)

        Args:
            fixed_image (Image): Fixed image for the registration
            moving_image (Image): Moving image for the registration
            bins (int): Number of bins for the intensity distribution
            sigma (float): Kernel sigma for the intensity distribution approximation
            spatial_samples (float): Percentage of pixels used for the intensity distribution approximation
            background: Method to handle background pixels. None: Set background to the min value of image
                                                            "mean": Set the background to the mean value of the image
                                                            float: Set the background value to the input value
            size_average (bool): Average loss function
            reduce (bool): Reduce loss function to a single value

    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, bins=64, sigma=3,
                 spatial_samples=0.1, background=None, size_average=True, reduce=True):
        super(MI, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._name = "mi"

        self._dim = fixed_image.ndim
        self._bins = bins
        self._sigma = 2*sigma**2
        self._normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
        self._normalizer_2d = 2.0 * np.pi*sigma**2

        if background is None:
            self._background_fixed = th.min(fixed_image.image)
            self._background_moving = th.min(moving_image.image)
        elif background == "mean":
            self._background_fixed = th.mean(fixed_image.image)
            self._background_moving = th.mean(moving_image.image)
        else:
            self._background_fixed = background
            self._background_moving = background

        self._max_f = th.max(fixed_image.image)
        self._max_m = th.max(moving_image.image)

        self._spatial_samples = spatial_samples

        self._bins_fixed_image = th.linspace(self._background_fixed, self._max_f, self.bins,
                                             device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

        self._bins_moving_image = th.linspace(self._background_moving, self._max_m, self.bins,
                                              device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

    @property
    def sigma(self):
        return self._sigma

    @property
    def bins(self):
        return self._bins

    @property
    def bins_fixed_image(self):
        return self._bins_fixed_image

    def _compute_marginal_entropy(self, values, bins):

        p = th.exp(-((values - bins).pow(2).div(self._sigma))).div(self._normalizer_1d)
        p_n = p.mean(dim=1)
        p_n = p_n/(th.sum(p_n) + 1e-10)

        return -(p_n * th.log2(p_n + 1e-10)).sum(), p

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(MI, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        moving_image_valid = th.masked_select(self._warped_moving_image, mask)
        fixed_image_valid = th.masked_select(self._fixed_image.image, mask)

        mask = (fixed_image_valid > self._background_fixed) & (moving_image_valid > self._background_moving)

        fixed_image_valid = th.masked_select(fixed_image_valid, mask)
        moving_image_valid = th.masked_select(moving_image_valid, mask)

        number_of_pixel = moving_image_valid.shape[0]

        sample = th.zeros(number_of_pixel, device=self._fixed_image.device,
                          dtype=self._fixed_image.dtype).uniform_() < self._spatial_samples

        # compute marginal entropy fixed image
        image_samples_fixed = th.masked_select(fixed_image_valid.view(-1), sample)

        ent_fixed_image, p_f = self._compute_marginal_entropy(image_samples_fixed, self._bins_fixed_image)

        # compute marginal entropy moving image
        image_samples_moving = th.masked_select(moving_image_valid.view(-1), sample)

        ent_moving_image, p_m = self._compute_marginal_entropy(image_samples_moving, self._bins_moving_image)

        # compute joint entropy
        p_joint = th.mm(p_f, p_m.transpose(0, 1)).div(self._normalizer_2d)
        p_joint = p_joint / (th.sum(p_joint) + 1e-10)

        ent_joint = -(p_joint * th.log2(p_joint + 1e-10)).sum()

        return -(ent_fixed_image + ent_moving_image - ent_joint)

class NGF(_PairwiseImageLoss):
    r""" Implementation of the Normalized Gradient Fields image loss.

            Args:
                fixed_image (Image): Fixed image for the registration
                moving_image (Image): Moving image for the registration
                fixed_mask (Tensor): Mask for the fixed image
                moving_mask (Tensor): Mask for the moving image
                epsilon (float): Regulariser for the gradient amplitude
                size_average (bool): Average loss function
                reduce (bool): Reduce loss function to a single value

    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, epsilon=1e-5,
                 size_average=True,
                 reduce=True):
        super(NGF, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._name = "ngf"

        self._dim = fixed_image.ndim
        self._epsilon = epsilon

        if self._dim == 2:
            dx = (fixed_image.image[..., 1:, 1:] - fixed_image.image[..., :-1, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:] - fixed_image.image[..., 1:, :-1]) * fixed_image.spacing[1]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_2d
        else:
            dx = (fixed_image.image[..., 1:, 1:, 1:] - fixed_image.image[..., :-1, 1:, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:, 1:] - fixed_image.image[..., 1:, :-1, 1:]) * fixed_image.spacing[1]
            dz = (fixed_image.image[..., 1:, 1:, 1:] - fixed_image.image[..., 1:, 1:, :-1]) * fixed_image.spacing[2]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy) + th.abs(dz))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_3d

    def _ngf_loss_2d(self, warped_image):

        dx = (warped_image[..., 1:, 1:] - warped_image[..., :-1, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:] - warped_image[..., 1:, :-1]) * self._moving_image.spacing[1]

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

    def _ngf_loss_3d(self, warped_image):

        dx = (warped_image[..., 1:, 1:, 1:] - warped_image[..., :-1, 1:, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, :-1, 1:]) * self._moving_image.spacing[1]
        dz = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, 1:, :-1]) * self._moving_image.spacing[2]

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(NGF, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        # compute the gradient of the warped image
        ng_warped_image = self._ngf_loss(self._warped_moving_image)

        value = 0
        for dim in range(self._dim):
            value = value + ng_warped_image[:, dim, ...] * self._ng_fixed_image[:, dim, ...]

        value = 0.5 * th.masked_select(-value.pow(2), mask)

        return self.return_loss(value)


class SSIM(_PairwiseImageLoss):
    r""" Implementation of the Structual Similarity Image Measure loss.

        Args:
                fixed_image (Image): Fixed image for the registration
                moving_image (Image): Moving image for the registration
                fixed_mask (Tensor): Mask for the fixed image
                moving_mask (Tensor): Mask for the moving image
                sigma (float): Sigma for the kernel
                kernel_type (string): Type of kernel i.e. gaussian, box
                alpha (float): Controls the influence of the luminance value
                beta (float): Controls the influence of the contrast value
                gamma (float): Controls the influence of the structure value
                c1 (float): Numerical constant for the luminance value
                c2 (float): Numerical constant for the contrast value
                c3 (float): Numerical constant for the structure value
                size_average (bool): Average loss function
                reduce (bool): Reduce loss function to a single value
    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None,
                 sigma=[3], dim=2, kernel_type="box", alpha=1, beta=1, gamma=1, c1=0.00001, c2=0.00001,
                 c3=0.00001, size_average=True, reduce=True, ):
        super(SSIM, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        self._c1 = c1
        self._c2 = c2
        self._c3 = c3

        self._name = "sim"
        self._kernel = None

        dim = dim
        sigma = np.array(sigma)

        if sigma.size != dim:
            sigma_app = sigma[-1]
            while sigma.size != dim:
                sigma = np.append(sigma, sigma_app)

        if kernel_type == "box":
            kernel_size = sigma * 2 + 1
            self._kernel = th.ones(*kernel_size.tolist()) \
                           / float(np.product(kernel_size) ** 2)
        elif kernel_type == "gaussian":
            self._kernel = utils.gaussian_kernel(sigma, dim, asTensor=True)

        self._kernel.unsqueeze_(0).unsqueeze_(0)

        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)

        # calculate mean and variance of the fixed image
        self._mean_fixed_image = F.conv2d(self._fixed_image.image, self._kernel)
        self._variance_fixed_image = F.conv2d(self._fixed_image.image.pow(2), self._kernel) \
                                     - (self._mean_fixed_image.pow(2))

    def forward(self, displacement):
        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(SSIM, self).GetCurrentMask(displacement)
        mask = 1 - mask
        mask = mask.to(dtype=self._dtype, device=self._device)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        mask = F.conv2d(mask, self._kernel)
        mask = mask == 0

        mean_moving_image = F.conv2d(self._warped_moving_image, self._kernel)

        variance_moving_image = F.conv2d(self._warped_moving_image.pow(2), self._kernel) - (
            mean_moving_image.pow(2))

        mean_fixed_moving_image = F.conv2d(self._fixed_image.image * self._warped_moving_image, self._kernel)

        covariance_fixed_moving = (mean_fixed_moving_image - mean_moving_image * self._mean_fixed_image)

        luminance = (2 * self._mean_fixed_image * mean_moving_image + self._c1) / \
                    (self._mean_fixed_image.pow(2) + mean_moving_image.pow(2) + self._c1)

        contrast = (2 * th.sqrt(self._variance_fixed_image + 1e-10) * th.sqrt(
            variance_moving_image + 1e-10) + self._c2) / \
                   (self._variance_fixed_image + variance_moving_image + self._c2)

        structure = (covariance_fixed_moving + self._c3) / \
                    (th.sqrt(self._variance_fixed_image + 1e-10) * th.sqrt(
                        variance_moving_image + 1e-10) + self._c3)

        sim = luminance.pow(self._alpha) * contrast.pow(self._beta) * structure.pow(self._gamma)

        value = -1.0 * th.masked_select(sim, mask)

        return self.return_loss(value)


class kernelNTG(_PairwiseImageLoss):
    def __init__(self,
                 fixed_image,
                 moving_image,
                 fixed_mask=None,
                 moving_mask=None,
                 size_average=True,
                 reduce=True,
                 kernel='sobel',
                 percent=0.02,
                 size=3,
                 weight=False,
                 debug=False):

        super(kernelNTG, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)
        self._name = "kernelNTG"
        self.warped_moving_image = None
        self.m = np.array([0])
        self.debug = debug
        self.kernel=kernel
        self.size=size
        self.weight=weight
        self.percent=percent

        # definition of different kernels to be used:
        self.sobel3_x = th.Tensor([[-1,0,1],
                             [-2,0,2],
                             [-1,0,1]]).float().to(self._device).view(1,1,3,3)

        self.sobel3_y = th.Tensor([[-1,-2,-1],
                             [0,0,0],
                             [1,2,1]]).float().to(self._device).view(1,1,3,3)

        self.scharr3_x = th.Tensor([[-47,0,47],
                             [-162,0,162],
                             [-47,0,47]]).float().to(self._device).view(1,1,3,3)

        self.scharr3_y = th.Tensor([[-47,-162,-47],
                             [0,0,0],
                             [47,162,47]]).float().to(self._device).view(1,1,3,3)

        self.sobel5_x = th.Tensor([[-1, -2, 0, 2, 1],
                              [-4, -8, 0, 8, 4],
                              [-6, -12,0,12, 6],
                              [-4, -8, 0, 8, 4],
                              [-1, -2, 0, 2, 1]]).float().to(self._device).view(1,1,5,5)

        self.sobel5_y = th.Tensor([[-1,-4, -6, -4,-1],
                              [-2, -8, -12, -8, -2],
                              [0,0,0,0,0],
                              [2, 8, 12, 8, 2],
                              [1,4, 6, 4, 1]]).float().to(self._device).view(1,1,5,5)
        if self.size==3:
            padding=1
        elif self.size==5:
            padding=2

        self.dx = th.nn.functional.conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=(self.size,self.size),
                          stride=1,
                          padding=padding,
                          groups=1,
                          bias=False)
        self.dy = th.nn.Conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=(self.size,self.size),
                          stride=1,
                          padding=padding,
                          groups=1,
                          bias=False)

        if self.kernel=='sobel' and self.size==3:
            self.dx.weight.data = self.sobel3_x
            self.dy.weight.data = self.sobel3_y
        elif self.kernel=='sobel' and self.size==5:
            self.dx.weight.data = self.sobel5_x
            self.dy.weight.data = self.sobel5_y
        elif self.kernel=='scharr' and self.size==3:
            self.dx.weight.data = self.scharr3_x
            self.dy.weight.data = self.scharr3_y

        self.dx.weight.requires_grad = True
        self.dy.weight.requires_grad = True


        if self.debug:
            from  matplotlib import pyplot as plt
            img1 = self._moving_image.image.cpu().detach().numpy()[0, 0]
            img2 = self._fixed_image.image.cpu().detach().numpy()[0, 0]
            self.plt = plt.imshow(np.dstack((img1,img2,img1)))
            plt.ion()
            plt.show()


    def GetCurrentMask(self, displacement):
        import skimage
        """
        Computes a mask defining if pixels are warped outside the image domain, or if they fall into
        a fixed image mask or a warped moving image mask.
        return (Tensor): maks array
        """
        # exclude points which are transformed outside the image domain

        mask = get_mask(toNP(self._fixed_image.image)[0,0], toNP(self.warped_moving_image)[0,0])

        mask2 = mask
        for _ in range(3):
            mask2 = skimage.morphology.binary_dilation(mask2)

        mask2 = th.from_numpy(mask2)
        mask2 = mask2[None,None,:,:]

        mask = th.from_numpy(mask)
        mask = mask[None,None,:,:]

        return mask.to(self._device), mask2.to(self._device)


    def forward(self, displacement):
        from matplotlib import pyplot as plt

        def weight_lower_percent(mat,mask):
            """
            Weighting that suppresses the lower percent gradients
            :param mat:     gradient matrix
            :param mask:    mask matrix
            :return:        matrix with weights for each pixel
            """
            mat_=mat.masked_select(~mask)
            m = th.max(mat_)
            weight = th.zeros(mat.shape).to(self._device)
            weight[mat>(self.percent*m)] = 1.0
            return weight

        # compute displacement field
        displacement = self._grid + displacement

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        # compute current mask
        mask,mask2 = self.GetCurrentMask(displacement)

        mov = self.warped_moving_image.masked_fill(mask,0)
        fix = self._fixed_image.image.masked_fill(mask,0)

        i1x = th.abs(self.dx(mov))
        i1y = th.abs(self.dy(mov))
        i2x = th.abs(self.dx(fix))
        i2y = th.abs(self.dy(fix))

        #weight
        if self.weight:
            i1x = i1x*weight_lower_percent(i1x,mask2)
            i1y = i1y*weight_lower_percent(i1y,mask2)
            i2x = i2x*weight_lower_percent(i2x,mask2)
            i2y = i2y*weight_lower_percent(i2y,mask2)

        dx = th.abs(i1x-i2x)
        dy = th.abs(i1y-i2y)

        # add narrower mask
        m  = (dx+dy).masked_fill(mask2,0).sum()
        n1 = (i1x+i1y).masked_fill(mask2,0).sum()
        n2 = (i2x+i2y).masked_fill(mask2,0).sum()

        metric = th.div(m, th.add(n1,n2))


        if self.debug:
            mov = (self.warped_moving_image).cpu().detach().numpy()[0, 0]
            fix = (self._fixed_image.image).cpu().detach().numpy()[0, 0]

            self.plt.set_data(np.dstack((mov,fix,mov)))
            plt.draw()
            plt.pause(0.0001)
        return metric


class NTG(_PairwiseImageLoss):
    def __init__(self,
                 fixed_image,
                 moving_image,
                 fixed_mask=None,
                 moving_mask=None,
                 size_average=True,
                 reduce=True,
                 debug=False):
        super(NTG, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)
        self._name = "NTG"
        self.warped_moving_image = None
        self.m = np.array([0])
        self.debug = debug


        self.dx = th.nn.Conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=(1,2),
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=False)

        kernel = th.tensor([1.0, -1.0]).to(self._device)
        self.dx.weight.data = kernel.view(1,1,1,2)

        self.dy = th.nn.Conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=(2,1),
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=False)

        self.dy.weight.data = kernel.view(1,1,2,1)
        if debug:
            from  matplotlib import pyplot as plt
            img1 = self._moving_image.image.cpu().detach().numpy()[0, 0]
            img2 = self._fixed_image.image.cpu().detach().numpy()[0, 0]
            self.plt = plt.imshow(np.dstack((img1,img2,img1)))
            plt.ion()
            plt.show()


    def GetCurrentMask(self, displacement):
        import skimage
        """
        Computes a mask defining if pixels are warped outside the image domain, or if they fall into
        a fixed image mask or a warped moving image mask.
        return (Tensor): maks array
        """
        # exclude points which are transformed outside the image domain
        mask = get_mask(toNP(self._fixed_image.image)[0,0], toNP(self.warped_moving_image)[0,0])

        mask2 = mask
        for _ in range(3):
            mask2 = skimage.morphology.binary_dilation(mask2)

        mask2 = th.from_numpy(mask2)
        mask2 = mask2[None,None,:,:]

        mask = th.from_numpy(mask)
        mask = mask[None,None,:,:]


        return mask.to(self._device),mask2.to(self._device)


    def forward(self, displacement):
        from matplotlib import pyplot as plt

        # compute displacement field
        displacement = self._grid + displacement

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        # compute current mask
        mask,mask2 = self.GetCurrentMask(displacement)

        mov = self.warped_moving_image.masked_fill(mask.bool(),0)
        fix = self._fixed_image.image.masked_fill(mask.bool(),0)

        i1x = th.abs(self.dx(mov))
        i1y = th.abs(self.dy(mov))
        i2x = th.abs(self.dx(fix))
        i2y = th.abs(self.dy(fix))

        delta = mov-fix
        dx = th.abs(self.dx(delta))
        dy = th.abs(self.dy(delta))

        xpad = th.nn.ZeroPad2d((0,1,0,0))
        ypad = th.nn.ZeroPad2d((0,0,0,1))
        i1x = xpad(i1x)
        i2x = xpad(i2x)
        dx = xpad(dx)
        i1y = ypad(i1y)
        i2y = ypad(i2y)
        dy  = ypad(dy)

        # add mask
        m  = (dx+dy).masked_fill(mask2.bool(),0).sum()
        n1 = (i1x+i1y).masked_fill(mask2.bool(),0).sum()
        n2 = (i2x+i2y).masked_fill(mask2.bool(),0).sum()


        metric = th.div(m, th.add(n1,n2))
        if self.debug:
            img1 = (delta).cpu().detach().numpy()[0, 0]
            img2 = (i1x-i2x).cpu().detach().numpy()[0, 0]
            img3 = (i1y-i2y).cpu().detach().numpy()[0, 0]
            mov = (self.warped_moving_image).cpu().detach().numpy()[0, 0]/255
            fix = (self._fixed_image.image).cpu().detach().numpy()[0, 0]/255

            self.plt.set_data(np.dstack((mov,fix,mov)))
            plt.draw()
            plt.pause(0.0001)
        return metric