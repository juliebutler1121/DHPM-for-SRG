#########################################################################
# Magnus-Butler.py
# Julie Butler
# Version 1.0
# July 9th, 2018
#
# A code that applies the Magnus Expansion to the In-Medium Similarity Renormalization
# Group Theory for nuclear physics. The specific case this Magnus Expansion is written for is the 
# pairing model containing eight single particle states and four particles.
#########################################################################

########################################################################
# Function Outline:
# bernoulli (n): Calcualates the nth Bernoulli number
# factorial (n): Calculates the factorial of n (n! = n*(n-1)*(n-2)*...*2*1).
# hamiltonian (d, g): Return the pairing model Hamiltonian.
# calculate_h_from_omega (omega, H0); Finds H(s) from H(0) and omega(s).
# magnus_expansion (omega, s, threshold, H0):  Calculates omega (s) by solving the differential 
#   equation for domega/ds.
# main omega, s, threshold, H0): Calcualtes H(s) from omega(s) by solving for domega/ds.
########################################################################

##################################################
#
#                 IMPORTS
#
##################################################
# System Imports
from fractions import Fraction as Fr
from sys import exit 

# Third-Party Imports
import numpy as np
from numpy import array, dot, diag, reshape, arange
from scipy.integrate import odeint
from scipy.linalg import eigvalsh

# Local Imports
from LinearAlgebra import commutator, nested_commutator 
from MatrixComparisons import compare_square_matrices
# Writes the solutions to a file in a formatted way
from ReadAndWriteSRGMatricesToFiles import write 

##################################################
#
#                      GLOBAL VARIABLES
#
##################################################
# Sets the maximum number of iterations that a loop can perform. 
MAX_ITERATIONS = 100

##################################################
#
#                                 bernoulli
#
##################################################
def bernoulli(n):
    """
    Calculate the nth Bernoulli number.
    Input:
        n (an integer):  The index of the Bernoulli number to be calculated.
    Output:
        A[0] (a fraction): The nth Bernoulli number.
    Algorithm taken from the following website, unmodified:
    https://rosettacode.org/wiki/Bernoulli_numbers#Python
    """
    A= [0] * (n+1)
    for m in range(n+1):
        A[m] = Fr(1, m+1)
        for j in range(m, 0, -1):
            A[j-1] = j*(A[j-1] - A[j])
    return A[0] # (which is Bn)

##################################################
#
#                                 factorial 
#
##################################################
def factorial (n):
    """
    Calculate the factorial of n (n! = n*(n-1)*(n-2)*...*2*1).
    Input:
        n (an integer): The number the factorial will be calcualted for.
    Output:
        Unnamed (an integer): The nth factorial.
    """
    if n == 0:
        return 1
    else:
        return n * factorial (n-1)

##################################################
#
#                                 hamiltonian
#
##################################################
def hamiltonian (d, g):
    """
    Return the pairing model Hamiltonian.
    Input:
        d (a double): The energy  level spacing.
        g (a double): The interaction strength.
    Output:
        Unnamed (a 6x6 numpy matrix): The initial Hamiltonian.
    """
    return array([
            [2*d-g, -g/2, -g/2, -g/2, -g/2, 0],
            [-g/2, 4*d-g, -g/2, -g/2, 0, -g/2],
            [-g/2, -g/2, 6*d-g, 0, -g/2, -g/2],
            [-g/2, -g/2, 0, 6*d-g, -g/2, -g/2],
            [-g/2, 0, -g/2, -g/2, 8*d-g, -g/2],
            [0, -g/2, -g/2, -g/2, -g/2, 10*d-g]
    ])

#################################################
#
#                    calculate_h_from_omega
#
#################################################
def calculate_h_from_omega (omega, H0, threshold):
    """
    Finds H(s) from H(0) and omega(s)
    Input:
        omega (a matrix): The value of omega(s)
        H0 (a matrix): The initial Hamiltonian
        threshold (a double): the terminating condition for the loop
    Output
        Hs (a matrix): the value of H(s) calculated using omega(s) and H(0)
    """
    # Calculates H(s) using the formula H(s) = SUM 1/k! [omega, H(0)]^(k)
    previous_term = (1/factorial (0)) * nested_commutator (omega, H0, 0)
    Hs = previous_term
    k = 1
    delta = 10000
    while (delta > threshold):
        if (k == MAX_ITERATIONS):
            print ("Summation failed to converge to required threshold.")
            print ("Program will now terminate")
            exit ()
        current_term = (1/factorial (k)) * nested_commutator (omega, H0, k)
        Hs = Hs + current_term
        delta = compare_square_matrices (current_term, previous_term, 6)
        previous_term = current_term
        k = k + 1
    return Hs

##################################################
#
#                              mangusExpansion
#
##################################################
def magnus_expansion (omega, s, threshold, H0):
    """
    Calculates omega (s) by solving the differential equation for domega/ds.
    Input:
        omega (a one-dimensional numpy array):  The current value of omega.
        s (a double):  A flow parameter, needed for the ODE solver.
        threshold (a double):  Used for terminating the loops. 
        H0 (a 6x6 numpy matrix): The matrix H(0), which is used to calculate H(s)
    Output:
        domega_ds (a one-dimensional numpy array):  The result of finding domega/ds, 
            reshaped into a one-dimensional array to satisfy the ODE solver.
    """
    # Reshapes the inputed omega array from a single dimension to a 6x6 matrix
    omega = reshape (omega, (6, 6))

    # Calculates H(s) using the formula H(s) = SUM 1/k! [omega, H(0)]^(k)
    Hs = calculate_h_from_omega (omega, H0, threshold)

    # Calculates eta from H(s) using eta = [Hd, Hod]
    Hd = diag (diag (Hs))
    Hod = Hs - Hd
    eta = commutator(Hd, Hod)

    # Computes the sum domega/dt = SUM Bn/n! [omega, eta]^(n)
    previous_term = (bernoulli (0) / factorial (0)) * nested_commutator (omega, eta, 0)
    domega_ds = previous_term
    delta = 10000
    n = 1
    while (delta > threshold):
        if (n == MAX_ITERATIONS):
            print ("Summation failed to converge to required threshold.")
            print ("Program will now terminate")
            exit ()
        current_term = (bernoulli (n) / factorial (n)) * nested_commutator (omega, eta, n)
        domega_ds = domega_ds + current_term
        delta = compare_square_matrices (current_term, previous_term, 6)
        previous_term = current_term 
        n = n + 1

    # needs to be returned as a single dimension array for ODE solver
    domega_ds = reshape (domega_ds, -1)
    return domega_ds.tolist()
		
#################################################
#
#                                   main
#
#################################################


# interaction coefficient
g     = 0.5
# energy level spacing 
d = 1

# initial Hamiltonian for the given values of g and d
H0    =  hamiltonian(d, g)
# the dimension of one side of the Hamiltonian (its square)
dim   = H0.shape[0]

# calculate exact eigenvalues
eigenvalues = eigvalsh(H0)

# turn initial Hamiltonian into a linear array
y0  = reshape(H0, -1)                 

# flow parameter step
ds = 1e-4

# flow parameters for snapshot images
flowparams = arange (0, 10, ds)

# threshold to end summation
threshold = 1e-4

def generate_data (flow_parameters, threshold, d, g):
    """
    Calcualtes H(s) from omega(s) by solving for domega/ds.
    Input:
        flow_parameters (a numpy array):  The values of s for which Hamiltonians need to
             be calculated for.
        threshold (a double):   Used for terminating the loops.  
        d (a double):  The energy level spacing
        g (a double): The interaction strength
    Output:
        H_list (a numpy array): a list of the Hamiltonians calculated for each flow
            parameter
    """

    # The initial value of omega for the ODE solver, reshaped into a one-dimensional array
    omega0 = array ([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    omega0 = reshape (omega0, -1)

    #Caclulation of H(0)
    H0 = hamiltonian (d, g)

    # Calculates omega(s) using the ODE solver, returns one matrix per flow parameter
    omegas = odeint (magnus_expansion, omega0, flow_parameters, args=(threshold,H0,))

    # Converts the values of omega(s) to H(s) using the formula 
    # H(s) = SUM 1/k! [omega, H(0)]^(k).  Does so for each flow parameter.
    Hs_list = []
    for omega, s in zip(omegas, flow_parameters):
        omega = reshape (omega, (6, 6))
        Hs = calculate_h_from_omega (omega, H0, threshold)
        Hs_list.append(Hs)
	
    return Hs_list
	

def main ():
    print ("Generating data")
    Hs_list = generate_data (flowparams, threshold, d, g)
    print ("Writing to file")
    write ("SRG_Training_Data_Magnus_1_e_-4.txt", flowparams, Hs_list)


#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()



