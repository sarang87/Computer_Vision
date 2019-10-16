import numpy as np
import cv2
from matplotlib import pyplot as plt


def gauss(img):
    midImage = cv2.GaussianBlur(img,(5,5),cv2.BORDER_REPLICATE)
    scale_percent = 80 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    subImage= cv2.resize (midImage,dim, interpolation = cv2.INTER_AREA)
    return subImage



def laplace(img):
    midImage1 = cv2.GaussianBlur(img,(5,5),cv2.BORDER_REPLICATE)
    laplaceImage=cv2.subtract(img,midImage1)
    return laplaceImage 
def main():
    plt.plot([1,2,3],[4,5,1])

    # Load an image in grayscale
    #first Apply gaussion blur and resize to 50% for for every level
    img = cv2.imread(r'/home/wickedshaman/Fall_2019/ECE_5554/Computer_Vision/hw1/elephant.png', cv2.IMREAD_GRAYSCALE)
    lImage= laplace(img)
    cv2.imshow('Original',img) 
    
    for i in range(6):
        print ("Running")     
        subImage = gauss(img)
        img = subImage
        #cv2.namedWindow('oOriginal', flags=cv2.WINDOW_FULLSCREEN)
            
        cv2.imshow('Gaussion Levels',subImage)
        cv2.imshow('Laplace Level 0',lImage)
        cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows()   
    
    
    
    #cv2.namedWindow('INPUT', flags=cv2.WINDOW_NORMAL)
   
        
if __name__== "__main__":
  main()