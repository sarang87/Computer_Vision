import numpy as np 
import cv2


# works for 3*3 only
def extractSubMatrix(inputMatrix, i, j, br_row, br_col):

      return inputMatrix[i-br_row:i+br_row+1,j-br_col:j+br_col+1]
         

def handleBoundary():
  return 0


def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h




def applyFilter(testMatrix):
  #testMatrix = np.arange(100).reshape(10,10)
  height = len(testMatrix)
  width = len (testMatrix[0])

  filter = np.array([(0.111108,0.111113,0.111108), (0.111113,0.111118,0.111113), (0.111108,0.111113,0.111108)])
  
  filter2= gauss2D((11,11),1.5)


  outputMatrix = np.zeros((height,width),dtype="uint8")
  
  ht_filter = int(len(filter2)/2)
  wd_filter = int(len(filter2[0])/2)
  # This works only for a 3*# generalize in other cases
  for i in range(ht_filter,len(testMatrix)-ht_filter):
    for j in range(wd_filter,len(testMatrix[0])-wd_filter):
      handleBoundary()
      sm = extractSubMatrix(testMatrix, i,j,ht_filter,wd_filter)
      value = weightedSum(filter2,len(filter2),len(filter2[0]),sm, len(sm),len(sm[0]))
      outputMatrix[i][j] = value
      # do the main operation here
      #val = filter.temp
  #print (outputMatrix)
  return (outputMatrix)


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

def weightedSum_1D(kernel, inputMat):
    n = len(kernel)
    value = 0
    #print (inputMat)
    for i in range(n):       
        value+= kernel[i] * inputMat[i]
        #print(str(kernel[i]) +"::"+ str(inputMat[i]))
        #print (value)
    #print("\n")
    return value
    
def extract1D_Matrix_RowWise(inputMatrix, i, j, filter_window ):
    #print("\n")
    #print (inputMatrix[:i,j-filter_window :j+filter_window +1])
    return (inputMatrix[:i,j-filter_window :j+filter_window +1])
    
    
def extract1D_Matrix_ColWise(inputMatrix, i,j,filter_window ):   
    return (inputMatrix[i-filter_window :i+filter_window +1,j])
    
    
def apply_1DFilter_row(inputMatrix, kernel_1D):
    kernel_rows = len(kernel_1D)
    filter_window = int(kernel_rows/2)
    height = len(inputMatrix)
    width = len(inputMatrix[0])
    outputMatrix = np.zeros((height,width),dtype="uint8")
    for i in range(1,len(inputMatrix)+1):
        for j in range(filter_window, len(inputMatrix[0])-filter_window):
            subMatrix_1D = inputMatrix[i-1:i,j - filter_window: j+filter_window+1]
            m=len(subMatrix_1D[0])
            value = weightedSum_1D(kernel_1D, subMatrix_1D.reshape(m))
            outputMatrix[i-1][j] = value
    return outputMatrix

#### code here
def apply_1DFilter_col(inputMatrix, kernel_1D):
    kernel_rows = len(kernel_1D)
    filter_window = int(kernel_rows/2)
    height = len(inputMatrix)
    width = len(inputMatrix[0])
    outputMatrix = np.zeros((height,width),dtype="uint8")
    for i in range(1,len(inputMatrix[0])+1):
        for j in range(filter_window, len(inputMatrix)-filter_window):     
            #print(str(j) +"::"+ str(i-1))
            subMatrix_1D = inputMatrix[j - filter_window: j+filter_window+1,i-1:i]
            m=len(subMatrix_1D)        
            value = weightedSum_1D(kernel_1D, subMatrix_1D.reshape(m))            
            outputMatrix[j][i-1] = value
    return outputMatrix
    

def get1Dkernel_row():
    return np.array([0.000003,0.000229,0.005977,0.060598,0.24173,0.382925,0.24173,0.060598, 0.005977, 0.000229, 0.000003])


def get1Dkernel_col():
    b = np.array([0.000003,0.000229,0.005977,0.060598,0.24173,0.382925,0.24173,0.060598, 0.005977, 0.000229, 0.000003])
    return b.transpose()

def main():
  #testfilter = np.array([1/3,1/3,1/3])
  #testmat = np.arange(60).reshape(6,10)
  #print(testmat)
  #apply_1DFilter_col(testmat, testfilter)
  kernel_1D = get1Dkernel_row()
  img = cv2.imread('/home/wickedshaman/Fall_2019/ECE_5554/Computer_Vision/hw1/elephant.png',cv2.IMREAD_GRAYSCALE)
  #print(img.shape)
  mat1 = apply_1DFilter_col(img, kernel_1D)
  #mat2= apply_1DFilter_row(img, kernel_1D)
  cv2.imshow('input_1d_col',img)
  #for i in range(len(mat1)):
  #    for j in range(len(mat1[0])):
  #        print (mat1[i][j])
  cv2.imshow('output_1d_col',mat1)
  cv2.waitKey(0)
  #cv2.namedWindow('window_output',flags=cv2.WINDOW_FULLSCREEN)

  cv2.destroyAllWindows()
  #cv2.imshow('input',img)
  #outputMatrix = applyFilter(img)
  #cv2.imshow('output',outputMatrix)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows

  
if __name__== "__main__":
  main()

