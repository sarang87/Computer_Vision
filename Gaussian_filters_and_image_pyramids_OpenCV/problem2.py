#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:16:17 2019

@author: Sarang Joshi
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
def gauss(img):
    midImage = cv2.GaussianBlur(img,(5,5),cv2.BORDER_REPLICATE)
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    subImage= cv2.resize (midImage,dim, interpolation = cv2.INTER_AREA)
  
    return subImage
"""
def get_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def display_image(input_matrix, captions = "Original image"):
    cv2.imshow(captions,input_matrix)

"""
apply a gaussian filter and resize the image by a factor
"""
def gaussian_blur(input_image, resize_factor):    
    blurred_input_image = cv2.GaussianBlur(input_image,(5,5),cv2.BORDER_REPLICATE)
    columns = int(input_image.shape[1] * resize_factor)
    rows = int(input_image.shape[0] * resize_factor) 
    image_dimensions    = (columns, rows)
    resized_image = cv2.resize (blurred_input_image, image_dimensions, interpolation = cv2.INTER_AREA)
    return resized_image
    

def laplace_transform(input_image):
    blurred_input_image = cv2.GaussianBlur(input_image,(5,5),cv2.BORDER_REPLICATE)
    laplace_image=cv2.subtract(input_image,blurred_input_image)
    return laplace_image 

def fft(img):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

def plot_graphs(img_magnitude, laplace_magnitude, levels):
    plt.subplot(121),plt.imshow(img_magnitude, cmap = 'gray')
    plt.title('Magnitude Gaussian{!s}'.format(levels)), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(laplace_magnitude, cmap = 'gray')
    plt.title('Magnitude Laplacian{!s}'.format(levels)), plt.xticks([]), plt.yticks([])
    levels+=1
    plt.show() 
    
def write_image(list_of_images, image_name):       
    level = 0 
    for image in list_of_images:
        image_name = image_name +  '_' + str(level) + ".JPEG"
        cv2.imwrite(image_name, image)
        level+=1
   

def transform_image(input_image, iterations):
    list_gaussians = []
    list_laplacians = []
    levels = 0    
    for i in range (iterations):      

        outImage = gaussian_blur(input_image, 0.5)
        list_gaussians.append(outImage)
        lImage= laplace_transform(input_image)
        list_laplacians.append(lImage)
        image_magnitude=fft(input_image)
        laplace_magnitude=fft(lImage)
        input_image = outImage      
        cv2.imshow('Image after applying gaussian ',outImage)
        cv2.imshow('mage after applying laplace transform ',lImage)
        plot_graphs(image_magnitude,  laplace_magnitude, levels)
        levels+=1
        cv2.waitKey(0) # waits until a key is pressed
        write_image(list_gaussians, 'gaussian_level')
        write_image(list_laplacians, 'laplacian_level')


def main():    
    # Load an image in grayscale  
    image_path =  '/home/wickedshaman/Fall_2019/ECE_5554/Computer_Vision/hw1/elephant.png'
    input_image = get_image(image_path)
    display_image(input_image)
    # Apply gaussian blur, apply laplace transform and fft on image while resizing
    transform_image(input_image, 5)       
    cv2.destroyAllWindows()

        
if __name__== "__main__":
  main()
