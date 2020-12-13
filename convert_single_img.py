"""Resize and crop images to square, save as tiff."""
from __future__ import division, print_function
import os
import click
import numpy as np
from PIL import Image, ImageFilter
import cv2


def clahe(img, clip_limit=2.0, grid=(8, 8)):

    clh = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_planes = cv2.split(img)
    img_planes[0] = clh.apply(img_planes[0])
    img = cv2.merge(img_planes)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


def convert(fname, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original 
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpeg', extension).replace(directory, 
                                                    convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory, 
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname) 


def save(img, fname):
    img.save(fname, quality=97)

# @click.command()
# @click.option('--input_path',
#               help="Input image.", )
# @click.option('--output_path',
#               help="Output image")
# @click.option('--crop_size', default=512, show_default=True,
#               help="Size of converted images.")
# @click.option('--extension', default='png', show_default=True,
#               help="Filetype of converted images.")
def main(input_path, output_path, crop_size, extension):
    for img in os.listdir(input_path):
        inp = input_path+img
        out = output_path+img[:-4] + 'jpg'
        image = convert(inp, crop_size)
        save(image, out)
        image = cv2.imread(out)
        image = clahe(image)
        cv2.imwrite(out, image)

if __name__ == '__main__':
    input_path = './Train/'
    output_path = './dataset/'
    main(input_path, output_path, 300, '.jpg')
