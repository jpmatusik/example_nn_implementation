
# when you use matrix notation on paper, you have to be careful to keep in mind the dimensions of the object you are using. There are no 1-D vectors, instead a vector is considered as a single column of a matrix. So now your notation depends on if you are dealing with a column vector or a matrix.

# For example, if you have a single observation A with d features, this is a column vector with dimensions (d, 1). Now if you have a transformation matrix, W, of dimentions (d, w), you would write the transformation as W^(T)xA and the result would be a column vector of dimensions (w, 1).

# However if A consisted of n samples, then A would now have dimensions (n, d). Now the transformation is written as AxW and the output would be a matrix with dimensions (n, w).

# Mathematical notation on paper is not strictly how the operation is implemented in a computer language. This script is to test how numpy handles these cases, because while numpy does allow you to create column/row vectors, it's preferred method is to create a 1-D vector instead.

# If you are really into mathematical notation, numpy 1-D vectors may trip you up because np.dot elegantly handles the two examples above without having to specifically handle the case where A is a column vector or a matrix.

# It does this by considering A to be a 1-D vector instead of a column vector and handling it 'how you would want it to be done.'

# numpy.dot
import numpy as np

# matrix dot matrix
# matrix multiplication
# returns a matrix
a = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1, 2],[2, 3],[3, 4]])
a.dot(w)

# vector dot matrix
# treats the vector as a single row and does matrix multiplication
# returns a vector of length equal to the number of columns in the matrix
a = np.array([1, 2, 3])
w = np.array([[1, 2],[2, 3],[3, 4]])

a = np.array([4, 5, 6])
w = np.array([[1, 2],[2, 3],[3, 4]])
a.dot(w)

# matrix dot vector
# this treats the vector as a single column and does matrix multiplication
# returns a vector of length equal to the number of rows in the matrix
a = np.array([[1, 2, 3],[2, 3, 4]])
w = np.array([1, 2, 3])
a.dot(w)

a = np.array([[1, 2, 3],[2, 3, 4]])
w = np.array([4,5, 6])
a.dot(w)

# vector dot vector
# typical inner product
# returns a single value
a = np.array([1, 2, 3])
w = np.array([4,5, 6])
a.dot(w)
