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
    -i --in=<directory>   Directory to find the raw images [default: /home/brian/Downloads/labels]
    -o --out=<directory>  Directory to put the processed images [default: /home/brian/Downloads/train_labels]

"""
from docopt import docopt
import cv2 as cv
import numpy as np
import sys, os

def mask(im):
    height,width,depth = im.shape
    circle_img = np.zeros((height,width), np.uint8)
    cv.circle(circle_img,(int(width/2),int(height/2)),int(height/2),1,-1)
    # Mask with circle
    masked_img = cv.bitwise_and(im, im, mask=circle_img)
    # mask off an additional rectangle 
    mask = np.zeros((height,width), np.uint8)
    mask[:,:] = 1 
    clip = int(height * .75)
    cv.rectangle(mask,(350,clip),(650,height),(0,0,0),-1)
    masked_img2 = cv.bitwise_and(masked_img,masked_img,mask = mask)
    return masked_img2

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
        else:
            img2 = img
        masked_img = mask(img2)
        cv.imwrite(os.path.join(out_folder,filename),masked_img) 
        print("Done")
if __name__ == '__main__':

    args = docopt(__doc__,version='Preprocess version 1.1')
    # print(args)
    CLAHE = False
    if args['-e']:
        CLAHE = True
    load_images_from_folder(args['--in'],args['--out'],CLAHE)