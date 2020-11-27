import cv2 as cv
import numpy as np
import sys, os

def mask(filename):
    # Mask with circle
    im = cv.imread(filename)
    height,width,depth = im.shape
    print (height,width,depth)
    cv.imshow("image", im)
    gray = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
    height,width = gray.shape
    print (height,width,depth)
    cv.imshow("gray", gray)
    cv.waitKey(0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cli = clahe.apply(gray)
    cv.imshow("gray", gray)
    cv.imshow("cli", cli)
    cv.waitKey(0)
    im2 = cv.cvtColor(cli,cv.COLOR_GRAY2RGB)
    circle_img = np.zeros((height,width), np.uint8)
    cv.circle(circle_img,(int(width/2),int(height/2)),int(height/2),1,-1)
    masked_img = cv.bitwise_and(im2, im2, mask=circle_img)
    

    # mask off an additional rectangle 
    mask = np.zeros((height,width), np.uint8)
    mask[:,:] = 1 
    clip = int(height * .75)
    cv.rectangle(mask,(350,clip),(650,height),(0,0,0),-1)
    masked_img2 = cv.bitwise_and(masked_img,masked_img,mask = mask)
    cv.imshow("masked2", masked_img2)
    cv.waitKey(0)

if __name__ == '__main__':
    # to run: python3 mask_demo.py <FILENAME>
    mask(sys.argv[1])