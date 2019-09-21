import numpy as np 

# works for 3*3 only
def extractSubMatrix(inputMatrix, i, j):

      return inputMatrix[i-1:i+2,j-1:j+2]
         

def handleBoundary():
  return 0

def applyFilter():
  testMatrix = np.arange(100).reshape(10,10)
  filter = np.array([(1,0,1), (0,1,1), (2,0,1)])
  outputMatrix = np.full([10,10],-1)
  # This works only for a 3*# generalize in other cases
  for i in range(1,len(testMatrix)-1):
    for j in range(1,len(testMatrix[0])-1):
      handleBoundary()
      sm = extractSubMatrix(testMatrix, i,j)
      value = weightedSum(filter,len(filter),len(filter[0]),sm, len(sm),len(sm[0]))
      outputMatrix[i][j] = value
      # do the main operation here
      #val = filter.temp
  print (outputMatrix)

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
  


def main():

  #createMatrix()
  tm1 = np.array([(1,0,1), (0,1,1), (2,0,1)])
  tm2 = np.arange(9).reshape(3,3)
  #print (weightedSum(tm1,len(tm1),len(tm1[0]), tm2,len(tm2),len(tm2[0])))
  applyFilter()

  
if __name__== "__main__":
  main()

