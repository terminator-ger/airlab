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
import sys
import os
#import time
import signal
import matplotlib.pyplot as plt
import torch as th

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al
import cv2
import numpy as np
from create_test_image_data import create_C_2_O_test_images
import json


def main():
    signal.signal(signal.SIGABRT, signal.SIG_IGN)
#    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    #device = th.device("cuda:1")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:1")

    # create test image data
    #	fixed_image, moving_image, shaded_image = create_C_2_O_test_images(256, dtype=dtype, device=device)
    img1 = cv2.imread('/home/lechnerml/storage/historical/a/00000001.tif')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('/home/lechnerml/storage/historical/b/00000001.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    moving_image = al.image_from_numpy(img1,
                                       pixel_spacing=(1, 1),
                                       image_origin=(0, 0),
                                       dtype=dtype,
                                       device=device)
    fixed_image = al.image_from_numpy(img2,
                                      pixel_spacing=(1, 1),
                                      image_origin=(0, 0),
                                      dtype=dtype,
                                      device=device)
    # create image pyramide size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(fixed_image, [ [8,8], [4,4], [2, 2]])
    moving_image_pyramid = al.create_image_pyramid(moving_image, [ [8,8], [4,4], [2, 2]])

    # initial registration search with projective matrix:
    # define parameter sets:
    constant_displacement = None

#    regularisation_weight = [1, 5, 50]
#    number_of_iterations = [120, 200, 200]
#    sigma = [[None,None], [11, 11], [3, 3]]

#    step_size = [ 7e-3, 1e-3, 1e-3]
    os.makedirs('experiments', exist_ok=True)

    parameters = {  'Transform' : [['Perspective', 'BSpline','BSpline', 'BSpline']],

                    'regularization_weight': [[1,100, 1000,1000],
                                              [1,100,  200, 500],
                                              [1,100,  500,1000],
                                              [1,10,  100, 1000]],

                    #'number_of_iterations': [[70, 100, 100, 100]],
                    'number_of_iterations': [[1, 1, 1, 1]],

                    'sigma': [[[None, None], [75,75], [100,100], [150,150]],
                              [[None, None], [75,75], [100,100], [150,150]],
                              [[None, None], [50,50], [100,100], [200,200]],
                              [[None, None], [50,50], [100,100], [150,150]],
                              [[None, None], [75,75], [150,150], [300,300]],
                              [[None, None], [75,75], [150,150], [150,150]]],

                    'order': [[None,1,1,1]],

                    'step_size' : [[ 7e-3, 7e-3, 7e-3, 7e-3]]
                  }
    def at(key):
        return list(parameters.keys()).index(key)

    import itertools
    param_combinations = list(itertools.product(*parameters.values()))
    for iteration, param in enumerate(param_combinations):



        os.makedirs('experiments/{}'.format(iteration), exist_ok=True)
        with open('experiments/{}/config.json'.format(iteration),'w+') as file:
            json.dump(param, file)

        for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

            registration = al.PairwiseRegistration(verbose=True)
            if param[at('Transform')][level] == 'Perspective':
                transformation = al.transformation.pairwise.PerspectiveTransformation(mov_im_level)
            elif param[at('Transform')][level] == 'BSpline':
                transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                                  sigma=param[at('sigma')][level],
                                                                                  order=param[at('order')][level],
                                                                                  dtype=dtype,
                                                                                  device=device,
                                                                                  diffeomorphic=True)

            if level > 0:
                constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                              mov_im_level.size,
                                                                              interpolation="linear")
                transformation.set_constant_flow(constant_flow)

            registration.set_transformation(transformation)

            image_loss = al.loss.pairwise.NTG(fix_im_level, mov_im_level, debug=False)

            registration.set_image_loss([image_loss])

            # define the regulariser for the displacement
            if param[at('Transform')][level] == 'BSpline':
                regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
                regulariser.SetWeight(param[at('regularization_weight')][level])
                registration.set_regulariser_displacement([regulariser])

            # define the optimizer
            optimizer = th.optim.Adam(transformation.parameters(),
                                      lr=param[at('step_size')][level])

            registration.set_optimizer(optimizer)
            registration.set_number_of_iterations(param[at('number_of_iterations')][level])

            registration.start()

            if param[at('Transform')][level] == 'Perspective':
                constant_flow = transformation.get_displacement()
            elif param[at('Transform')][level] == 'BSpline':
                constant_flow = transformation.get_flow()

        # create final result
        displacement = transformation.get_displacement()
        warped_image = al.transformation.utils.warp_image(moving_image, displacement)
        displacement = al.create_displacement_image_from_image(displacement, moving_image)

        # create inverse displacement field
        inverse_displacement = transformation.get_inverse_displacement()
        inverse_warped_image = al.transformation.utils.warp_image(warped_image, inverse_displacement)
        inverse_displacement = al.create_displacement_image_from_image(inverse_displacement, moving_image)


        warped = warped_image.image.cpu().detach().numpy()[0, 0]
        fixed = fixed_image.image.cpu().detach().numpy()[0, 0]
        cv2.imwrite('experiments/{}/moving.png'.format(iteration), warped)
        cv2.imwrite('experiments/{}/fixed.png'.format(iteration), fixed)
        plt.imshow(displacement.magnitude().image.cpu().detach().numpy()[0, 0])
        plt.savefig('experiments/{}/displacement.png'.format(iteration))
        cv2.imwrite('experiments/{}/diff.png'.format(iteration), np.dstack((warped,fixed,warped)))
#        end = time.time()

#    print("=================================================================")
#
#    print("Registration done in: ", end - start)
#    print("Result parameters:")
#
#    # plot the results
#    plt.subplot(241)
#    plt.imshow(fixed_image.numpy(), cmap='gray')
#    plt.title('Fixed Image')
#
#    plt.subplot(242)
#    plt.imshow(moving_image.numpy(), cmap='gray')
#    plt.title('Moving Image')
#
#    plt.subplot(243)
#    plt.imshow(warped_image.numpy(), cmap='gray')
#    plt.title('Warped Shaded Moving Image')
#
#    plt.subplot(244)
#    plt.imshow(displacement.magnitude().numpy(), cmap='jet')
#    plt.title('Magnitude Displacement')
#
#    # plot the results
#    plt.subplot(245)
#    plt.imshow(warped_image.numpy(), cmap='gray')
#    plt.title('Warped Shaded Moving Image')
#
#    plt.subplot(246)
#    plt.imshow(np.dstack((warped_image.image.cpu().detach().numpy()[0, 0],
#                          fixed_image.image.cpu().detach().numpy()[0, 0],
#                          warped_image.image.cpu().detach().numpy()[0, 0])), cmap='gray')
#    plt.title('Shaded Moving Image')
#
#    plt.subplot(247)
#    plt.imshow(inverse_warped_image.numpy(), cmap='gray')
#    plt.title('Inverse Warped Shaded Moving Image')
#
#    plt.subplot(248)
#    plt.imshow(inverse_displacement.magnitude().numpy(), cmap='jet')
#    plt.title('Magnitude Inverse Displacement')
#
#    plt.show()


if __name__ == '__main__':
    main()
