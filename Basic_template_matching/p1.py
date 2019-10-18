#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:50:38 2019

@author: wickedshaman
"""
import cv2
import numpy as np
# load images

# add noise loop

# step 2 gaussian blur
################################
#noisy - modified from Shubham Pachori on stackoverflow
def noisy(image, noise_type, sigma):
    if noise_type == "gauss":
        row,col = image.shape
        mean = 0
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy


def main():
    
   # load images 
   input_img = cv2.imread(r'/home/wickedshaman/Fall_2019/ECE_5554/Computer_Vision/hw2/motherboard-gray.png', cv2.IMREAD_GRAYSCALE)
   temp = cv2.imread(r'/home/wickedshaman/Fall_2019/ECE_5554/Computer_Vision/hw2/template.png', cv2.IMREAD_GRAYSCALE)
   cv2.imshow('Original',input_img)
   cv2.imshow('Template',temp)
   cv2.waitKey(0)
    
   l = ["gauss"]
   k = 0
   # noise loop
   for i in range(1,5):
       print (" for i ="+ str(i))
       for j in range(1,5):
           smoothed_img = cv2.GaussianBlur(input_img,(5,5),float(j))
           noisy_img = np.uint8(noisy(input_img, l[k], float(i)))
           print ("\tj="+ str(j))
           output_img = cv2.matchTemplate(input_img,temp,cv2.TM_CCOEFF_NORMED)
           cv2.imshow('output',output_img)
           cv2.waitKey(0) 
           #j = (j+1) % len(l)
   cv2.destroyAllWindows()
          
if __name__== "__main__":
  main()