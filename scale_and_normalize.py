###########################################################
###########################################################
####                                                   ####
####             LUMINOSITY NORMALIZATION &            ####
####       PREPROCESSING RECIPE OF THE WINNER OF       ####
####       KAGGLE DIABETIC RETINOPATHY CHALLENGE       #### 
####                                                   ####
###########################################################
###########################################################

##  This class performs both standard luminosity normalization 
##  and the preprocessing steps used by the winner of the KaggleDR
##  challenge for color fundus images.
##
##  The procedure of the KaggleDR winner includes the following steps:
##  	1) Re-scaling the radius of the FOV.
##    2) Luminosity normalization by subtraction of a blurred version
##       of the image from the image itself.
##    3) Removal of the bright circle characterizing the boundary of
##       the FOV to smoothen up the transition from the background inside
##       the FOV.
##
##  More information about this preprocessing recipe on:
##  http://blog.kaggle.com/2015/09/09/diabetic-retinopathy-winners-interview-1st-place-ben-graham/ 


##  Python standard Python modules
import argparse
import os, sys
import numpy as np
import cv2

##  My Python modules
import preprocessing_fundus as pf

##  Format variable
myfloat = np.float32
myint = np.int
path = "./claheImages/val/1/"
outpath = "./finalDataset/"

################################################
################################################
####                                        #### 
####            GET INPUT ARGUMENTS         ####
####                                        ####
################################################
################################################

def _examples():
    print(
        '\n\nEXAMPLES\n\nScale and normalize color fundus images according to the the recipe of the KaggleDR winner:\n' \
        '"""\npython scale_and_normalize.py -i /pstore/data/pio/Tasks/PIO-233/data/RIDE_and_RISE_csme_qc_filt_orig_nopre/CF-58703-2009-04-15-M3-RE-F2-LS.png -o ./CF-58703-2009-04-15-M3-RE-F2-LS.png -s 30\n"""\n\n'
    )


def _get_args():
    parser = argparse.ArgumentParser(
        prog='scale_and_normalize',
        description='Scale and normalize color fundus',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        , add_help=False
    )

    parser.add_argument('-i', '--file_in', dest='file_in',
                        help='Select input image', default='./43923_left.jpeg')

    parser.add_argument('-o', '--file_out', dest='file_out',
                        help='Select output image name', default='./43923_left_norm.jpeg')

    parser.add_argument('-s', '--scale', dest='scale', type=myfloat, default=-1,
                        help='Select rescaling factor (output dimension in px = 2x rescaling factor in each dimension)')

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples', default=False)

    args = parser.parse_args()

    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.file_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input image not specified!\n')

    if os.path.isfile(args.file_in) is False:
        parser.print_help()
        sys.exit('\nERROR: Input image does not exist!\n')

    if args.file_out is None:
        parser.print_help()
        sys.exit('\nERROR: Output image name not specified!\n')

    return args


################################################
################################################
####                                        #### 
####             CLASS DEFINITION           ####
####                                        ####
################################################
################################################

class ScaleAndNormalize():

    ################################################
    ################################################
    ####                                        ####
    ####               INITIALIZATION           ####
    ####                                        ####
    ################################################
    ################################################

    def __init__(self, scale):
        self._scale = scale

    ################################################
    ################################################
    ####                                        ####
    ####         RUN PREPROCESSING RECIPE       ####
    ####         OF THE KAGGLE-DR WINNER        ####
    ####                                        ####
    ################################################
    ################################################

    def _run(self, file_in, file_out):
        ##  Read image
        # print(file_in)
        img = cv2.imread(file_in)

        ##  Preprocessing of KaggleDR
        if self._scale:
            ##  Set scale to default
            if self._scale == -1:
                self._scale = 0.5 * img.shape[0]

            ##  Rescale radius
            img = pf.scale_radius(img, self._scale)

            ##  Subtract locally average color
            img = pf.normalize_luminosity_m2(img, self._scale)

            ##  Remove outer 10% of the image
            img2 = np.zeros(img.shape)
            cv2.circle(img2, (np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)),
                       np.int(self._scale * 0.9), (1, 1, 1), -1, 8, 0)
            img = img * img2 + 128 * (1 - img2)



        ##  Only luminosity normalization
        else:
            ##  Subtract locally average color)
            img = pf.normalize_luminosity_m1(img)

        ##  Create output file name
        # file_out = self._create_output_filename(file_out)

        ##  Write preprocessed image to output file
        cv2.imwrite(file_out, img)

    ################################################
    ################################################
    ####                                        ####
    ####      DETERMINE FILENAME IDENTIFIER     ####
    ####                                        ####
    ################################################
    ################################################

    def _filename_identifier(self):
        if self._scale > 0:
            identifier = '_preproc-kaggledr'

        else:
            identifier = '_lumnorm'

        return identifier

    ################################################
    ################################################
    ####                                        ####
    ####         CREATE OUTPUT FILE NAME        ####
    ####                                        ####
    ################################################
    ################################################

    def _create_output_filename(self, fileOut):
        identifier = self._filename_identifier()
        head, ext = os.path.splitext(fileOut)
        fileOut = head + identifier + ext
        return fileOut

# def auc_roc(prediction):


################################################
################################################
####                                        #### 
####             	   MAIN                 ####
####                                        ####
################################################
################################################

if __name__ == '__main__':
    ##  Get inputs
    args = _get_args()

    ##  Initialize class
    preproc = ScaleAndNormalize(scale=args.scale)
    preproc._run(args.file_in, args.file_out)


    ##  Run preprocessing
    for file in os.listdir(path):
        inp = path + file
        out = outpath + file
        # print(inp, out)
        preproc._run(inp, out)
