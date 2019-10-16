import numpy as np
import cv2

#import image 
# Load an image in grayscale
img = cv2.imread(r'/home/wickedshaman/Fall_2019/ECE_5554/hw1/einstein.bmp')
height = len(img)
width = len(img[0])
imgarray= np.asarray(img)
#fiter and output

testMatrix = imgarray
filter = np.full([3,3],1/9)
outputMatrix = np.zeros((height, width, 1), dtype = "uint8")
  
# works for 3*3 only
def extractSubMatrix(inputMatrix, i, j):

      return inputMatrix[i-1:i+2,j-1:j+2]
         

def handleBoundary():
  return 0

def applyFilter():

  # This works only for a 3*# generalize in other cases
  for i in range(1,len(testMatrix)-1):
    for j in range(1,len(testMatrix[0])-1):
      handleBoundary()
      sm = extractSubMatrix(testMatrix, i,j)
      value = weightedSum(filter,len(filter),len(filter[0]),sm, len(sm),len(sm[0]))
      outputMatrix[i][j] = value
      outputImage = cv.CreateMat( outputMatrix, )
      # do the main operation here
      #val = filter.temp
 



def weightedSum(filter, f_row, f_col, subMatrix, sm_row, sm_col):
  if (f_row != sm_row or f_col != sm_col):
    print (exit)
  else:
    list_filter = list (filter.reshape(f_row*f_col))
    list_sm = list (subMatrix.reshape(sm_row*sm_col))
    total = 0
    for i in range(len(list_filter)):
      total += list_filter[i] * list_sm[i]
    return total




  # move i and j for one row and column ahead of 0 to one row and column behind last row,column

def updateMatrix(smMatrix, index_i, index_j):
 return 0
  
# show image at 1/2 scale
"""
cv2.namedWindow('INPUT', flags=cv2.WINDOW_NORMAL)
cv2.imshow('INPUT',img)
cv2.resizeWindow('INPUT', (int(width/2), int(height/2)))
cv2.namedWindow('RES', flags=cv2.WINDOW_NORMAL)
cv2.imshow('RES', outputMatrix)
cv2.resizeWindow('RES', (int(width/2), int(height/2))) 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


