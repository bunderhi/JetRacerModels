import cv2 as cv
import numpy as np
import sys, os

def mask(filename):
    # Mask with circle
    im = cv.imread(filename)
    height,width,depth = im.shape
    print (height,width,depth)
    cv.imshow("image", im)
    #gray = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
    #height,width = gray.shape
    #print (height,width,depth)
    #cv.imshow("gray", gray)
    #cv.waitKey(0)
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cli = clahe.apply(gray)
    #cv.imshow("gray", gray)
    #cv.imshow("cli", cli)
    #cv.waitKey(0)
    #im2 = cv.cvtColor(cli,cv.COLOR_GRAY2RGB)
    cv.rectangle(im,(75,200),(775,600),(0,0,0),3)
    cv.imshow("im", im)
    cv.waitKey(0)
    crop_img = im[200:600, 75:775].copy()
    cv.imshow("crop_img", crop_img)
    height,width,depth = crop_img.shape
    print (height,width,depth)
    cv.waitKey(0)

if __name__ == '__main__':
    # to run: python3 mask_demo.py <FILENAME>
    mask(sys.argv[1])