###########################################################
###########################################################
####                                                   #### 
####       LIBRARY OF PREPROCESSING FUNCTIONS FOR      ####
####              COLOR FUNDUS PHOTOGRAPHY             ####
####                                                   ####
###########################################################
###########################################################


##  Python modules
from __future__ import division, print_function
import sys
import numpy as np
import cv2
import os

from sklearn import preprocessing
from skimage.color import rgb2grey, grey2rgb
from skimage import exposure
from scipy.stats import norm
from scipy.ndimage import filters
from scipy import interpolate as intp
from scipy.signal import argrelextrema
from scipy.ndimage import morphology as mph
from scipy import misc

##  My variable format
myfloat = np.float32
path = "./finalDataset/val/1/"
outpath = "./claheImages/"


##  Constants
EPS = 1e-7


###########################################################
###########################################################
####                                                   #### 
####                   RGB TO GRAYSCALE                ####
####                                                   ####
###########################################################
###########################################################


def rgb2gray(rgb):
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


###########################################################
###########################################################
####                                                   #### 
####                  CLAHE EQUALIZATION               ####
####                                                   ####
###########################################################
###########################################################

def clahe(img, clip_limit=2.0, grid=(8, 8)):

    clh = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_planes = cv2.split(img)
    img_planes[0] = clh.apply(img_planes[0])
    img = cv2.merge(img_planes)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


###########################################################
###########################################################
####                                                   #### 
####                    GAMMA CORRECTION               ####
####                                                   ####
###########################################################
###########################################################

def adjust_gamma(img, gamma=1.2):

    ##  Build a lookup table mapping the pixel values [0,255]
    ##  to their gamma-adjusted values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')

    ##  Apply gamma correction using the lookup table
    new_imgs = cv2.LUT(np.array(img, dtype=np.uint8), table)

    return new_imgs


###########################################################
###########################################################
####                                                   #### 
####       CREATE CUSTOMIZED 2D GAUSSIAN KERNEL        ####
####                                                   ####
###########################################################
###########################################################

def gauss_kernel_2d(shape=(3, 3), sigma=0.5):

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


###########################################################
###########################################################
####                                                   #### 
####        GET BACKGROUND MASK OF FUNDUS IMAGES       ####
####                                                   ####
###########################################################
###########################################################

####  The output mask is 0 for the background pixels and 1
####  for the pixel of the retinal scan.

def get_mask(img_in):
    ##  Number of bins
    nbins = 256

    ##  Copy input image and convert it to floating precision
    img = img_in.copy().astype(myfloat)

    ##  Make sure it is either a grey level image or a channel
    ##  of a RGB image
    if img.ndim == 3:
        img = rgb2grey(img)

    ##  Normalize image to the interval [0,255]
    img[:] = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

    ##  Get histogram profile
    x = np.arange(nbins)
    hist = np.histogram(img, bins=nbins)[0]
    ii = np.argwhere(hist != 0).flatten()
    f = intp.interp1d(x[ii], hist[ii])(x)

    ##  Analyze the local minima of the histogram profile:
    ##  the optimal threshold corresponds to the grey level,
    ##  where is located the first local minimum following
    ##  the highest peak (being of dark pixels)
    x_min_local = argrelextrema(f, np.less)[0].flatten()
    x_max_global = np.argwhere(f == np.max(f))[0].flatten()
    ind = np.argwhere(x_min_local > x_max_global)[0].flatten()
    thres = x_min_local[ind]

    ##  Create approximated mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
    mask[img > thres] = 1.0

    ##  Clean mask by removing all black pixels inside it
    mask = mph.binary_fill_holes(mask)

    return mask


###########################################################
###########################################################
####                                                   #### 
####          NORMALIZE LUMINOSITY -- METHOD 1         ####
####                                                   ####
###########################################################
###########################################################

def normalize_luminosity_m1(img, sigma1=50, sigma2=50):
    if img.ndim == 3:
        img = rgb2grey(img)

    img = img.astype(myfloat)
    eps = 1e-1

    hs1 = np.int(np.ceil(-norm.ppf(0.5 * eps, 0, sigma1)))
    size1 = 2 * hs1 + 1

    hs2 = np.int(np.ceil(-norm.ppf(0.5 * eps, 0, sigma2)))
    size2 = 2 * hs2 + 1

    gauss1 = gauss_kernel_2d(shape=(size1, size1), sigma=sigma1)
    gauss2 = gauss_kernel_2d(shape=(size2, size2), sigma=sigma2)

    num = img - filters.convolve(img, gauss1)
    den = np.sqrt(filters.convolve(num * num, gauss2))

    return num / (den + EPS)


###########################################################
###########################################################
####                                                   #### 
####          NORMALIZE LUMINOSITY -- METHOD 2         ####
####                                                   ####
###########################################################
###########################################################

def normalize_luminosity_m2(img, scale):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30),
                           -4, 128)


###########################################################
###########################################################
####                                                   #### 
####                      SCALE RADIUS                 ####
####                                                   ####
###########################################################
###########################################################

def scale_radius(img, scale):
    x = img[np.int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10.0).sum() / 2.0

    if r < 1:
        r = np.int(img.shape[0] / 2)

    s = scale * 1.0 / r
    print('scale', s)
    return cv2.resize(img, (0, 0), fx=s, fy=s)


###########################################################
###########################################################
####                                                   #### 
####           CONTRAST STRETCHING THROUGH CLAHE       ####
####                                                   ####
###########################################################
###########################################################

def equalize_contrast(img_in, clip_limit=0.0, nbins=20):
    img = (img_in - np.min(img_in)) / (np.max(img_in) - np.min(img_in)) * 2 - 1
    img[:] = exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=nbins)

    return img


###########################################################
###########################################################
####                                                   #### 
####           RADIAL PADDING OF FUNDUS IMAGES         ####
####                                                   ####
###########################################################
###########################################################

def radial_padding(img_in, mask, nn=1):
    ##  Copy input image and convert it to floating precision    
    img = img_in.copy().astype(myfloat)

    ##  Mark all background pixels with 0    
    img[mask == 0] = 0

    ##  Initialize image to store the "contour pixels" to process 
    diff = mask.copy()

    ##  Radial padding by iterative expansion of the binary mask "diff"
    while (np.sum(diff) != 0):
        diff[:] = mph.binary_dilation(mask) - mask

        ii = np.argwhere(diff != 0)

        for i in range(len(ii)):
            ##  Get neighborhood of the selected pixel
            jmin = np.max([0, ii[i, 0] - nn])
            jmax = ii[i, 0] + nn + 1
            kmin = np.max([0, ii[i, 1] - nn])
            kmax = ii[i, 1] + nn + 1
            neigh = img[jmin:jmax, kmin:kmax]

            ##  Get numbers of contributing neighbors
            n = np.argwhere(neigh != 0).shape[0]

            ##  Assign to selected pixel the mean value of its non-zero neighbors
            if n != 0 and img[ii[i, 0], ii[i, 1]] == 0:
                img[ii[i, 0], ii[i, 1]] = 1.0 / n * np.sum(neigh)

                ##  Expand mask by 1 pixel in every direction
        mask[:] += diff

    return img


###########################################################
###########################################################
####                                                   #### 
####            LINE PADDING OF FUNDUS IMAGES          ####
####                                                   ####
###########################################################
###########################################################

def line_padding(img_in, mask):
    ##  Copy input image and convert it to floating precision
    img = img_in.copy().astype(myfloat)

    ##  Get number of rows and columns
    nr, nc = img.shape

    ##  Mark all background pixels with 0
    img[mask == 0] = 0

    ##  Line padding along the rows
    for i in range(nr):
        ii = np.argwhere(img[i, :] != 0).flatten()
        if ii.shape[0] != 0:
            i1 = ii[0];
            v1 = img[i, i1]
            i2 = ii[len(ii) - 1];
            v2 = img[i, i2]
            img[i, :i1] = v1
            img[i, i2 + 1:] = v2

    ##  Line padding along the columns
    for i in range(nc):
        ii = np.argwhere(img[:, i] != 0).flatten()
        if ii.shape[0] != 0:
            i1 = ii[0];
            v1 = img[i1, i]
            i2 = ii[len(ii) - 1];
            v2 = img[i2, i]
            img[:i1, i] = v1
            img[i2 + 1:, i] = v2

    return img


###########################################################
###########################################################
####                                                   #### 
####        STRETCH RETINAL SCAN TO A RECTANGLE        ####
####                                                   ####
###########################################################
###########################################################

def stretch_scan(img_in, mask, margin=0):
    ##  Get extension of the rectangular image to be
    ones = np.argwhere(mask == 1)
    imin = np.min(ones[:, 0]) + margin;
    imax = np.max(ones[:, 0]) - margin
    jmin = np.min(ones[:, 1]) + margin;
    jmax = np.max(ones[:, 1]) - margin

    ##  Copy input image and convert it to floating precision
    img = img_in.copy().astype(myfloat)

    ##  Stretch image along the rows
    for i in range(imin, imax + 1):
        ind = np.argwhere(mask[i, :] == 1)
        arr = img_in[i, ind].reshape(-1, 1)
        arr = misc.imresize(arr, (jmax + 1 - jmin, 1), interp='bilinear')
        img[i, jmin:jmax + 1] = arr.reshape(-1)

    ##  Stretch image along the columns
    for j in range(jmin, jmax + 1):
        ind = np.argwhere(mask[:, j] == 1)
        arr = img_in[ind, j].reshape(-1, 1)
        arr = misc.imresize(arr, (imax + 1 - imin, 1), interp='bilinear')
        img[i, imin:imax + 1] = arr.reshape(-1)

        ##  Crop rectangular ROI
    img = img[imin:imax + 1, jmin:jmax + 1]

    return img


###########################################################
###########################################################
####                                                   #### 
####                   IMAGE NORMALIZATION             ####
####                                                   ####
###########################################################
###########################################################

def normalization(img, mean, std):
    return (img - mean) / std


###########################################################
###########################################################
####                                                   #### 
####                   IMAGE WHITENING                 ####
####                                                   ####
###########################################################
###########################################################

def whitening(img):
    ##  Convert array to integer type
    if img.ndim == 3:
        img = np.asarray(img, dtype='uint8').transpose(2, 0, 1)
    else:
        img = np.asarray(img, dtype='uint8')

    ##  Whitening for color images
    if img.ndim == 3:
        whitened = []

        for ii in range(img.shape[0]):
            a = img[ii] - img[ii].mean()
            aa = np.fft.fft2(a)
            spectr = np.sqrt(np.mean(np.dot(np.transpose(np.abs(aa)), np.abs(aa))))
            out = np.fft.ifft2(np.dot(aa, 1. / (spectr + EPS)))
            out = preprocessing.scale(np.abs(out))
            whitened.append(out)
        whitened = np.asarray(whitened)
        whitened = np.swapaxes(whitened, 0, 2)

    ##  Whitening of gray images
    else:
        a = img - img.mean()
        aa = np.fft.fft2(a)
        spectr = np.sqrt(np.mean(np.dot(np.transpose(np.abs(aa)), np.abs(aa))))
        out = np.fft.ifft2(np.dot(aa, 1. / (spectr + EPS)))
        whitened = preprocessing.scale(np.abs(out))

    return whitened


if __name__ == '__main__':

    # img = cv2.imread('./finalDataset/val/1/42780_right.jpeg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for file in os.listdir(path):
        inp = path + file
        out = outpath + file
        img = cv2.imread(inp)
        img_clahe = clahe(img)
        cv2.imwrite(out, img_clahe)
        # print(inp, out)
        # preproc._run(inp, out)
    # img_clahe =clahe(img)                   # contrast limited histogram equalisation *** best results with (5, 5) grid
    # img_gamma = adjust_gamma(img_clahe)     # gamma adjusted image

    # cv2.imshow('original_image', img)
    # cv2.imshow('clahe', img_clahe)
    # cv2.imshow('gamma_image', img_gamma)
    # cv2.waitKey(0)


