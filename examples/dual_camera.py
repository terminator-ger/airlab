#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import sys
import cv2
from Image import imgCVShape, toFloatImage
import os
import time
import numpy as np

import matplotlib.pyplot as plt
import torch as th
from scipy.spatial.transform import Rotation
from airlab import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al
from airlab.utils.image import image_from_numpy

def calc_h(r1,r2, t12, k1, k2, d):
    Rb = Rotation.from_euler('xyz', r1, degrees=True).as_matrix() 
    Ra = Rotation.from_euler('xyz', r2, degrees=True).as_matrix() 

    Rot = Ra @ np.linalg.inv(Rb)
    H = Rot
    tmp = (-Rot @ t12)/d
    H[:,2] += tmp

    H = k2 @ H @ np.linalg.inv(k1)
    H = H/H[2,2]
    return H

def toRad(r):
    return r * np.pi / 180

def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # load the image data and normalize to [0, 1]

    r1 = np.array([-4.44, -5.38, 2.66]).astype(np.float32)
    r2 = np.array([-1.27, -5.57, 0.57]).astype(np.float32)
    t12 = np.array([153, 8, 0]).astype(np.float32)

    k1= np.array([[2.79521733e+03,       0.00000000e+00,     2.02009334e+03],
                 [0.00000000e+00,      2.90937103e+03, 1.07732724e+03],
                 [0.00000000e+00,      0.00000000e+00,     1.00000000e+00]]).astype(np.float32)

    k2 =    np.array([[901.8698481,  0.,              368.26672249],
                      [  0.,           810.62167476,    248.48349116],
                      [  0.,           0.,                1.        ]]).astype(np.float32)

    d = 7500.0
    H = calc_h(r1, r2, t12, k1, k2, d)
    print(H)
    img2 = cv2.imread('./data/dual_camera_test_image_2d_moving.png')
    img1 = cv2.imread('./data/dual_camera_test_image_2d_fixed.png')

    img2 = cv2.warpPerspective(img2, np.linalg.inv(H), imgCVShape(img1))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    plt.imshow(np.dstack((img1,img2,img1))) 
    plt.show()
    dtype = th.float32
    moving_image = image_from_numpy(img2,
                                    pixel_spacing=(1,1),
                                    image_origin=(0,0),
                                    dtype=dtype,
                                    device=device)
    fixed_image  = image_from_numpy(img1, pixel_spacing=(1,1),
                                    image_origin=(0,0),
                                    dtype=dtype,
                                    device=device)

    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object
#    fixed_image.image = 1 - fixed_image.image
#    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    transformation = al.transformation.pairwise.DualCameraRegistration(moving_image, opt_cm=False)
    # initialize the translation with the center of mass of the fixed image
    #transformation.init_translation(fixed_image)
    transformation.set_parameters(7500)

    registration.set_transformation(transformation)
    
    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.dNTG(fixed_image, moving_image, debug=True)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(1000)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")
    print("Result parameters:")
    transformation.print()

    # plot the results
    plt.subplot(131)
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(132)
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(133)
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.show()


if __name__ == '__main__':
    main()
