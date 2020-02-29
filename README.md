# hybrid_image

Hybrid images are created by adding the low pass filtered image with the high pass filtered image. In this project, we implement a convolution function, a function to create a gaussian kernel, and the generation of the hybrid image.

## Overview

### Filtering
The implementation of filtering can be straightforwardly implemented using for-loops; however, writing loops in python is not efficient and slows down our program. For that we tried utilizing NumPy's API to avoid writing for-loops and hope for a faster code running-time. We also used reflection padding instead of zero padding for the convolution.

### Hybrid image generation
To create a hybrid image, we remove the high frequency components of one image, and the low frequency components of the other image and then add both filtered images together. To remove the high frequencies, we convolve a gaussian kernel of any odd kernel size with the image. To remove the low frequencies, we subtract the gaussian filtered image from the original image.

## Example result
