"""
Script to convert images to tiff format

Usage:
    decompress.py [-e] [--in=<directory>] [--out=<directory>] 

Options:
    -h --help     Show this screen.
    --version     Show version.
    -e  Run Histogram equalization 
    -i --in=<directory>   Directory to find the raw images [default: /home/brian/Downloads/train_data/Masks]
    -o --out=<directory>  Directory to put the processed images [default: /home/brian/Downloads/train_data/NewMasks]

"""
from docopt import docopt
from PIL import Image
import numpy as np
import sys, os


def load_images_from_folder(in_folder,out_folder):
    for filename in os.listdir(in_folder):
        #img = cv.imread(os.path.join(in_folder,filename))
        img = Image.open(os.path.join(in_folder,filename))
        assert img is not None,"File:"+filename+" not loaded"
        print(filename)
        #cv.imwrite(os.path.join(out_folder,newfilename),img) 
        img.save(os.path.join(out_folder,filename), compression='deflate')

if __name__ == '__main__':

    args = docopt(__doc__,version='decompress version 1.1')
    # print(args)
    load_images_from_folder(args['--in'],args['--out'])