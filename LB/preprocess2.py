"""
Script to preprocess the training images 

Usage:
    preprocess.py [-e] [--in=<directory>] [--out=<directory>] 
    preprocess.py (-h | --help)
    preprocess.py --version

Options:
    -h --help     Show this screen.
    --version     Show version.
    -e  Run Histogram equalization 
    -i --in=<directory>   Directory to find the raw images [default: /home/brian/Downloads/raw_train_data]
    -o --out=<directory>  Directory to put the processed images [default: /home/brian/Downloads/train_data/Images]

"""
from docopt import docopt
import cv2 as cv
import numpy as np
import sys, os


def equalize(im):
    gray = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
    # create a CLAHE object
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cli = clahe.apply(gray)
    im2 = cv.cvtColor(cli,cv.COLOR_GRAY2RGB)
    return im2
def load_images_from_folder(in_folder,out_folder,CLAHE):
    for filename in os.listdir(in_folder):
        img = cv.imread(os.path.join(in_folder,filename))
        assert img is not None,"File:"+filename+" not loaded"
        print(filename)
        if CLAHE:
            img2 = equalize(img)
            print("CLAHE")
        else:
            img2 = img
            print("No CLAHE")
        crop_img = img2[230:550, 130:770].copy()
        cv.imwrite(os.path.join(out_folder,filename),crop_img) 

if __name__ == '__main__':

    args = docopt(__doc__,version='Preprocess version 1.1')
    # print(args)
    CLAHE = False
    if args['-e']:
        CLAHE = True
    load_images_from_folder(args['--in'],args['--out'],CLAHE)