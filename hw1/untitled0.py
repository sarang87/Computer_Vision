#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:04:41 2019

@author: Sarang Joshi
"""
import numpy as np
import cv2


"""
Class to apply gaussian filters on an image
"""
class imageBlur:   
    # Class Attribute
    input_matrix = None
    filepath = None
    gaussian_1D_row = None
    gaussian_1D_col = None
    gaussian_2D = None

    """
    Initialize
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.input_matrix = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        self.gaussian_1D_row = self.get_gaussian_1D_row()
        self.gaussian_1D_col = self.get_gaussian_1D_col()
        self.gaussian_2D = self.get_gaussian_2D()
        
        
    """
    instance method to display the input image
    """
    def display_image(self,  display_matrix,captions = 'input_image',):
        if display_matrix is None:
            display_matrix = self.input_matrix      
        cv2.imshow(captions, display_matrix)
        cv2.waitKey(0)
        
        
    """
    returns the matrix for the original image    
    """
    def get_input_matrix(self):
        return self.input_matrix
    
    """
    returns a gaussian kernel 1*11
    """
    def get_gaussian_1D_row(self):
        return np.array([0.000003,0.000229,0.005977,0.060598,0.24173,0.382925,0.24173,0.060598, 0.005977, 0.000229, 0.000003])

    """
    returns a gaussian kernel 11*1
    """
    def get_gaussian_1D_col(self):
        b = np.array([0.000003,0.000229,0.005977,0.060598,0.24173,0.382925,0.24173,0.060598, 0.005977, 0.000229, 0.000003])
        return b.transpose()
    
    """
    calcluates the sum of the input kernel and the input matrix, same shape for 1D
    """
    def sum_neighbours_1D(self, input_kernel, input_matrix):
        kernel_size = len(input_kernel)
        sum = 0
        for i in range(kernel_size):       
            sum+= input_kernel[i] * input_matrix[i]
        return sum
    
    """
    applies a 1D gaussian kernel on an input matrix row major
    """
    def apply_1DGaussian_row(self, inputMatrix, kernel_1D):
        # Set default values 
        if inputMatrix is None:
            inputMatrix=self.input_matrix
        if kernel_1D is None:
            kernel_1D = self.gaussian_1D_row
        kernel_rows = len(kernel_1D)
        filter_window = int(kernel_rows/2)
        height = len(inputMatrix)
        width = len(inputMatrix[0])
        outputMatrix = np.zeros((height,width),dtype="uint8")
        for i in range(1,len(inputMatrix)+1):
            for j in range(filter_window, len(inputMatrix[0])-filter_window):
                subMatrix_1D = inputMatrix[i-1:i,j - filter_window: j+filter_window+1]
                value = self.sum_neighbours_1D(kernel_1D, subMatrix_1D.reshape(len(subMatrix_1D[0])))
                outputMatrix[i-1][j] = value
        return outputMatrix
    
    
    """
    calculates a sum of neighbouring pixels by running a filter on a sub_matrix
    """
    def sum_neighbours_2D(self, sub_matrix, sm_row, sm_col):
      filter_2D = self.gaussian_2D
      f_row = len(filter_2D)
      f_col = len(filter_2D[0])
      if (f_row != sm_row or f_col != sm_col):
        print (exit)
      else:
        list_filter = list (filter_2D.reshape(f_row*f_col))
        list_sm = list (sub_matrix.reshape(sm_row*sm_col))
        sum = 0
        for i in range(len(list_filter)):
          sum += list_filter[i] * list_sm[i]
      return sum
    
    
    
    """
    applies a 2D gaussian filter on the input matrix and returns an output matrix
    """
    def apply_2D_filter(self):
      #testMatrix = np.arange(100).reshape(10,10)
      height = len(self.input_matrix)
      width = len (self.input_matrix[0])
    
      #filter = np.array([(0.111108,0.111113,0.111108), (0.111113,0.111118,0.111113), (0.111108,0.111113,0.111108)])    
      filter_2D = self.gaussian_2D  
      output_matrix = np.zeros((height,width),dtype="uint8")
     
      ht_filter = int(len(filter_2D)/2)
      wd_filter = int(len(filter_2D[0])/2)
      # This works only for a 3*# generalize in other cases
      for i in range(ht_filter,len(self.input_matrix)-ht_filter):
        for j in range(wd_filter,len(self.input_matrix[0])-wd_filter):
          #handleBoundary()
          sub_matrix = self.extract_sub_matrix(self.input_matrix, i,j,ht_filter,wd_filter)
          avg_sum = self.sum_neighbours_2D(sub_matrix, len(sub_matrix),len(sub_matrix[0]))
          output_matrix[i][j] = avg_sum
          # do the main operation here
          #val = filter.temp
      #print (outputMatrix)
      return (output_matrix)
    
    """
    extracts a sub matrix of br_row and br_column from the i, j coordinates in in an input matrix
    """
    def extract_sub_matrix(self, input_matrix, i, j, br_row, br_col):
      return input_matrix[i-br_row:i+br_row+1,j-br_col:j+br_col+1]
    
    
    """
    2D gaussian mask - method to create a 2D gaussian filter
    """
    def get_gaussian_2D(self, shape=(11,11),sigma=1.5): 
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    
    """
    Code to debug by counting common values in two matrices and common zero values
    """
    def count_common_pixels(self, mat1, mat2):
        common = 0
        common_0 = 0
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                if mat1[i][j] == mat2[i][j]:
                    common+=1
                    if mat1[i][j] == 0:
                        common_0 +=1
        print (common)
        print (common_0)

    
    """
    applies a 1D gaussian kernel on an input matrix col major
    """
    def apply_1DGaussian_col(self, inputMatrix, kernel_1D):
        # Set default values 
        if inputMatrix is None:
            inputMatrix=self.input_matrix
        if kernel_1D is None:
            kernel_1D = self.gaussian_1D_row
        kernel_rows = len(kernel_1D)
        filter_window = int(kernel_rows/2)
        height = len(inputMatrix)
        width = len(inputMatrix[0])
        outputMatrix = np.zeros((height,width),dtype="uint8")
        for i in range(1,len(inputMatrix[0])+1):
            for j in range(filter_window, len(inputMatrix)-filter_window):     
                #print(str(j) +"::"+ str(i-1))
                subMatrix_1D = inputMatrix[j - filter_window: j+filter_window+1,i-1:i]    
                value = self.sum_neighbours_1D(kernel_1D, subMatrix_1D.reshape(len(subMatrix_1D)))            
                outputMatrix[j][i-1] = value
        return outputMatrix
    
    
def main():
  filepath = '/home/wickedshaman/Fall_2019/ECE_5554/Computer_Vision/hw1/lighthouse.bmp'
  i1 = imageBlur(filepath)
  input_matrix = i1.get_input_matrix()
  
  
  # show input image  
  cv2.imshow('Original image lighthouse',input_matrix)
  cv2.waitKey(0)
  
  #apply 1D filter row-wise and display image
  kernel_1D_row = i1.get_gaussian_1D_row()
  output_matrix_row = i1.apply_1DGaussian_row(input_matrix, kernel_1D_row)
  #i1.display_image(output_matrix_row, "output 1D_gaussian_row")
  
  #apply 1D filter col-wise and display image
  kernel_1D_col = i1.get_gaussian_1D_col()
  output_matrix_col = i1.apply_1DGaussian_col(input_matrix, kernel_1D_col)
  #i1.display_image(output_matrix_col, "output 1D_gaussian_col")
 
  # apply both column and row filters
  output_matrix_two_1D_filters = i1.apply_1DGaussian_col(output_matrix_row, kernel_1D_col)
  i1.display_image(output_matrix_two_1D_filters, "Output image 2*1D filters")
  cv2.imwrite("lighthouse_1D_filter.png",output_matrix_two_1D_filters)
  
  # apply a 2D filter
  output_matrix_2D = i1.apply_2D_filter()
  i1.display_image(output_matrix_2D, "Output image 2D filter")
  cv2.imwrite("lighthouse_2D_filter.png",output_matrix_2D)
  
  # calculate and display the difference matrix
  i1.count_common_pixels (output_matrix_two_1D_filters,output_matrix_2D )
  diff_matrix = np.subtract(output_matrix_two_1D_filters, output_matrix_2D)
  i1.display_image(diff_matrix, "Difference image")
  cv2.imwrite("lighthouse_diff.png",diff_matrix)
  cv2.destroyAllWindows()
  
  # Print statistics for difference matrix from the two filters
  print("\nStatistics for difference matrix")
  print(np.mean(diff_matrix))
  print(np.median(diff_matrix))
  print(np.var(diff_matrix))
  
  print("\nStatistics for 2*1D filters output matrix")
  print(np.mean(output_matrix_two_1D_filters))
  print(np.median(output_matrix_two_1D_filters))
  print(np.var(output_matrix_two_1D_filters))
  
  print("\nStatistics for 2D filter output matrix")
  print(np.mean(output_matrix_2D))
  print(np.median(output_matrix_2D))
  print(np.var(output_matrix_2D))

  
if __name__== "__main__":
  main()
