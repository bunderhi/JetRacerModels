"""
Script to convert images to tiff format

Usage:
    converttotiff.py [-e] [--in=<directory>] [--out=<directory>] 

Options:
    -h --help     Show this screen.
    --version     Show version.
    -e  Run Histogram equalization 
    -i --in=<directory>   Directory to find the raw images [default: /home/brian/Downloads/train_data/Masks]
    -o --out=<directory>  Directory to put the processed images [default: /home/brian/Downloads/train_data/Masks]

"""
from docopt import docopt
import cv2 as cv
import numpy as np
import sys, os


def load_images_from_folder(in_folder,out_folder):
    for filename in os.listdir(in_folder):
        img = cv.imread(os.path.join(in_folder,filename))
        assert img is not None,"File:"+filename+" not loaded"
        pre, ext = os.path.splitext(filename)
        newfilename = pre+".tiff"
        print(filename,newfilename)
        cv.imwrite(os.path.join(out_folder,newfilename),img) 

if __name__ == '__main__':

    args = docopt(__doc__,version='converttotiff version 1.1')
    # print(args)
    load_images_from_folder(args['--in'],args['--out'])