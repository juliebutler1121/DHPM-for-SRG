#########################################################################
# LinearAlgebra.py
# Julie Butler
# Version 1.1
# July 9th, 2018
#
# A colection of methods that perform linear algebra related calculations.  The methods are organized
# alphabetically, with an outline of all methods given below.
#########################################################################

#########################################################################
# Function Outline:
# commutator (x, y): computes the commutator of the numpy arrays x and y.
# nested_commutator (x, y, n): computes the nested commutator to the nth degree of the 
# numpy arrays x and y
#########################################################################
##################################################
#
#                 IMPORTS
#
##################################################
# Third-Party Imports
import numpy as np 
from numpy import dot

##################################################
#
#                  commutator
#
##################################################
def commutator (x, y):
	"""	
	 Performs a commutator on the numpy arrays x and y.
	Input:
		x (a numpy array): The first array that the nested commutator is to be performed on.
		y (a numpy array): Thesecond array that the nested commutator is to be performed on.
	Output:
		Unnamed (depends on input):  The results of performing the commutator on x and y  
	"""
	return dot (x, y) - dot (y, x)

##################################################
#
#              nested_commutator
#
##################################################
def nested_commutator (x, y, n):
	"""
    Performs a nested commutator with n iterations on the numpy arrays x and y.
	Input:
		x (a numpy array):  The first array that the nested commutator is to be performed
			on.
		y (a numpy array): The second array that the nested commutator is to be performed
			on.
		n (an integer): the number of times the nested commutator routine
			is to be performed.
	Output:
		Unnamed (depends on input): The result of performing the nested commutator n
			times on class level numpy arrays x and y.
	"""
	if n == 0:
		return y
	else:
		return commutator (x, nested_commutator (x, y, n-1))
		
