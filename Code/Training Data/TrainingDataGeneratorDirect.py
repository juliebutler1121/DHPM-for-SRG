#########################################################################################
# TrainingDataGeneratorDirect.py
# Julie Butler
# 2-5-19
#########################################################################################

from math import sin, pi
from numpy import arange

def wave_equation (nx, ny, nz, L, x, y, z):
    return (2/L)**(3/2)*sin(nx*pi*x/L)*sin(ny*pi*y/L)*sin(nz*pi*z/L)

def main ():
    nx = 1
    ny = 1
    nz = 1

    L = 1

    space_values = arange (0, L, 1e-2)

    F = open("Schrodinger_Equation_Training_Data_1_e_-2.txt", 'w')

    for x in space_values:
        for y in space_values:
            for z in space_values:
                to_file = str(x) + "," + str(y) + "," + str(z) + ","
                wave_value = wave_equation (nx, ny, nx, L, x, y, z)
                to_file = to_file + str(wave_value) + "\n"
                F.write(to_file)

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()


                
