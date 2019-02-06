#########################################################################
# Matrix Comparisons.py
# Julie Butler
# July 12th, 2018
# Version 1.0
#
# A collection of methods that allow for the values contained in two matrices to be eassily compared.
# Currently only implemented for square matrices.
#########################################################################

#########################################################################
# Function Outline:
# compare_square_matrices (x, y, dim): Calcualte the differencce between two matrices using 
#    delta = SUM_{i = 0}^d SUM_{j=0}^d ||xij|-|yij||
# compare_square_matrices_by_element (x, y, dim): Calcualtes the difference between two square 
#   matrices by element.
# ratio_matrix (matrix): Return a matrix such that each element has been divided by the global maximum of 
#   the matrix
#########################################################################

#################################################
#
#                      compare_square_matrices
#
#################################################
def compare_square_matrices (x, y, dim):
    """
    Calcualte the differencce between two matrices using 
    delta = SUM_{i = 0}^d SUM_{j=0}^d ||xij|-|yij||.
    Input:
        x (a matrix): a square matrix of dimension dim
        y (a matrix): a square matrix of dimension dim.  Must be the same size as x.
        dim (an integer):  The size of one side of one of the matrices.
    Output:
        total_difference (a double): The difference between the matrices.
    """
    total_difference = 0
    for i in range (0, dim):
        for j in range (0, dim):
                total_difference = total_difference + abs(abs(x[i][j]) - abs(y[i][j]))
    return total_difference

#################################################
#
#               compare_square_matrices_by_element
#
#################################################
def compare_square_matrices_by_element (x, y, dim):
    """
    Calcualtes the difference between two square matrices by element.
    Input:
        x (a matrix): a square matrix of dimension dim
        y (a matrix): a square matrix of dimension dim.  Must be the same size as x.
        dim (an integer):  The size of one side of one of the matrices.
    Output:
        total_difference (a matrix): The difference between the two matrices by element.
    """
    total_difference = []
    for i in range (0, dim):
        row_difference = []
        for j in range (0, dim):
            row_difference.append (abs (abs (x[i][j]) - abs (y[i][j])))
        total_difference.append (row_difference)
    return total_difference

#################################################
#
#                         ratio_matrix
#
#################################################
def ratio_matrix (matrix):
    """
    Return a matrix such that each element has been divided by the global maximum of 
    the matrix.
    Input:
        matrix (a matrix): The starting matrix.
    Output:
        matrix (a matrix): The matrix after the ratio calcualtions have been performed
    """
    # Find the maximum value
    local_maxes = [max(sub_array) for sub_array in matrix]
    global_max = max (local_maxes)
    # Convert the elements of the matrix to ratios
    for i in range (0, 6):
    	for j in range (0, 6):
    		matrix[i][j] = matrix[i][j] / global_max
    return matrix
				
