
#!/usr/bin/env python

#------------------------------------------------------------------------------
# TrainingDataGenerator.py
#
# author:   Julie Butler
# version:  1.0.0
# date:     
# 
# Solves the pairing model for four particles in a basis of four doubly 
# degenerate states by means of a Similarity Renormalization Group (SRG)
# flow.  Writes to a file the Hamiltonian for values of the flow parameter
# from 0 to 10, using a flow parameter step of 1e-6.
#
# Adapted from the code used for Lecture Notes in Physics 936.  Original Code
# written by H. Hergert,
#
#------------------------------------------------------------------------------

#################################################
# IMPORTS
#################################################
import numpy as np
# Matrix Calculations
from numpy import array, dot, diag, reshape, arange
from scipy.linalg import eigvalsh
# ODE Solver
from scipy.integrate import odeint
# Writes the solutions to a file in a formatted way
from ReadAndWriteSRGMatricesToFiles import write 

#################################################
# FUNCTIONS
#################################################

# HAMILTONIAN
def Hamiltonian(delta,g):
    """
        Inputs: 
            delta (an int): the energy level spacing the the pairing model
            g (an int): the interaction coefficient in the pairing model
        Returns:
            H (a matrix): the initial Hamiltonian for the pairing model
        Generates and returns the initial Hamiltonian for the pairing model consisting 
        of four particles and four energy levels.
    """

    H = array(
        [[2*delta-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,          0.],
         [   -0.5*g, 4*delta-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ], 
         [   -0.5*g,    -0.5*g,  6*delta-g,         0.,    -0.5*g,     -0.5*g ], 
         [   -0.5*g,    -0.5*g,         0.,  6*delta-g,    -0.5*g,     -0.5*g ], 
         [   -0.5*g,        0.,     -0.5*g,     -0.5*g, 8*delta-g,     -0.5*g ], 
         [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g, 10*delta-g ]]
        )

    return H

# COMMUTATOR
def commutator(a,b):
    """
        Inputs:
            a, b (varied, but must be the same type):  The two elements of which the 
                commutator is to be taken.  The order they are passed matters.
        Returns:
            (varies, same type as a and b): The commutator of a and b
        Takes the commutator of a and b (order matters):
            [a,b] = ab - ba
    """
    return dot(a,b) - dot(b,a)

# FLOW_EQUATION
def flow_equation(y, t, dim):
    """
        Inputs:
            y (a square matrix): the initial Hamilitonian of the pairing model
            t (an array of ints): the values of the flow parameter at which values of 
                the Hamiltonian for the pairing model need to be found.
            dim (an int): the dimension of the square matrix y
        Returns:
            dy/dy (a matrix): the solution to the SRG flow equation at a flow parameter
                value(s) given by t
        Solves the SRG flow equation for the pairing model at various values of the flow
        parameter 
    """

    # reshape the solution vector into a dim x dim matrix
    H = reshape(y, (dim, dim))

    # extract diagonal Hamiltonian...
    Hd  = diag(diag(H))

    # ... and construct off-diagonal the Hamiltonian
    Hod = H-Hd

    # calculate the generator
    eta = commutator(Hd, Hod)

    # dH is the derivative in matrix form 
    dH  = commutator(eta, H)

    # convert dH into a linear array for the ODE solver
    dydt = reshape(dH, -1)
    
    return dydt

#################################################
# MAIN PROGRAM
#################################################

# interaction coefficient
g     = 0.5
# energy level spacing 
delta = 1

# initial Hamiltonian for the given values of g and d
H0    = Hamiltonian(delta, g)
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

# GENERATE_DATA
def generate_data(initial_y, s_values):
    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys  = odeint(flow_equation, initial_y, s_values, args=(dim,))

    # reshape individual solution vectors into dim x dim Hamiltonian
    # matrices
    Hs  = reshape(ys, (-1, dim,dim))

    return Hs
    

def main():
    print ("Generating data")
    Hs = generate_data(y0, flowparams)
    print ("Writing to file")
    write ("SRG_Training_Data_1_e_-4.txt", flowparams, Hs)
 
#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()



