# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy.signal import gaussian
# import matplotlib.pyplot as plt
def my_imfilter(image: np.ndarray, filter: np.ndarray):
  """
      Your function should meet the requirements laid out on the project webpage.
      Apply a filter to an image. Return the filtered image.
      Inputs:
      - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
      - filter -> numpy nd-array of odd dim (k, l)
      Returns
      - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
      Errors if:
      - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  # filtered_image = np.asarray([0])
  filter = np.flip(np.flip(filter,0),1)
  assert len(filter.shape) ==2," Filter must have shape (k,l)"
  assert filter.shape[0]%2 ==1, "filter dimensions have to be odd"
  assert filter.shape[1]%2 ==1, "filter dimensions have to be odd"
  new_shape = list(image.shape)
  r_pad, c_pad = ((filter.shape[0] - 1)//2 , (filter.shape[1] - 1)//2) #row and column padding
  new_shape[0] += r_pad*2
  new_shape[1] += c_pad*2
  new_image = np.zeros(new_shape, dtype=image.dtype)
  new_image[r_pad:r_pad+image.shape[0], c_pad:c_pad+image.shape[1]] = image
  # pad with reflected image
  # horizontal up and down strips:
  new_image[:r_pad,c_pad:c_pad+image.shape[1]] = image[:r_pad,:][::-1,:] # upper h strip
  new_image[-r_pad:,c_pad:c_pad+image.shape[1]] = image[-r_pad:,:][::-1,:] # lower h strip
  # vertical up and down strips:
  new_image[r_pad:r_pad+image.shape[0],:c_pad] = image[:,:c_pad][:,::-1] # left v strip
  new_image[r_pad:r_pad+image.shape[0],-c_pad:] = image[:,-c_pad:][:,::-1] # right v strip
  # corners: by flipping the already flipped edges from previous step.
  new_image[:r_pad, :c_pad] = new_image[:r_pad, c_pad:2*c_pad][:,::-1] # top left
  new_image[:r_pad, -c_pad:] = new_image[:r_pad, -2*c_pad:-c_pad][:,::-1] # top right
  new_image[-r_pad:, :c_pad] = new_image[-r_pad:, c_pad:2*c_pad][:,::-1] # bottom left
  new_image[-r_pad:, -c_pad:] = new_image[-r_pad:, -2*c_pad:-c_pad][:,::-1] # bottom right
  # plt.imshow(new_image)
  # plt.show()
  if len(new_shape) ==2: new_image = np.expand_dims(new_image,2)

  convolver = np.vectorize(lambda i,j,k: np.sum(filter*new_image[i:i+filter.shape[0],
                                                                 j:j+filter.shape[1],k]))

  filtered_image = convolver(np.asarray(range(0,image.shape[0])).reshape(-1,1,1), # is
                             np.asarray(range(0,image.shape[1])).reshape(1,-1,1), # js
                             np.asarray(range(0,new_image.shape[2])).reshape(1,1,-1)) #ks

  return filtered_image.squeeze()

def create_gaussian_filter(ksize, sigma):
    # create gaussian filter of an arbitrary MxN dimensions
    kernel1 = gaussian(ksize[0],sigma)
    kernel2 = gaussian(ksize[1],sigma)
    kernel2D = np.outer(kernel1,kernel2)
    # Normalize the kernel values
    return (kernel2D)/(np.sum(kernel2D))

# lets leave that here for now til the actual imfilter is implemented
def im(img,kernel):
    my_blur[:,:,0] = correlate2d(img[:,:,0], kernel, 'same')
    my_blur[:,:,1] = correlate2d(img[:,:,1], kernel, 'same')
    my_blur[:,:,2] = correlate2d(img[:,:,2], kernel, 'same')
    return filtered_image

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
  ksize = (19,19)  
  kernel = create_gaussian_filter(ksize,cutoff_frequency)
  
  # Your code here:
  #low_frequencies = my_imfilter(image1,kernel) # Replace with your implementation
  low_frequencies = im(image1,kernel) 
  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  kernel = create_gaussian_filter(ksize,cutoff_frequency*2)
  high = image2 - my_imfilter(image2,kernel)  
  #when visualizing, make sure you use np.uint8()
  high_frequencies = (high+128.0)%255 # Replace with your implementation

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  hybrid_image = np.clip((high_frequencies+low_frequencies-128),0,255) # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!

  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
      Visualize a hybrid image by progressively downsampling the image and
      concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect')
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
