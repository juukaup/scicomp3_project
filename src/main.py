#!/usr/bin/env python3

import scipy.io
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
from project_funcs import *

def main():
    
    test_data_size = 200
   
    # Compute the svd:s for all the training data
    print("Computing SVD:s of all digits...")
    U1,_,_ = get_svd_of_digit(1, test_data_size)
    U2,_,_ = get_svd_of_digit(2, test_data_size)
    U3,_,_ = get_svd_of_digit(3, test_data_size)
    U4,_,_ = get_svd_of_digit(4, test_data_size)
    U5,_,_ = get_svd_of_digit(5, test_data_size)
    U6,_,_ = get_svd_of_digit(6, test_data_size)
    U7,_,_ = get_svd_of_digit(7, test_data_size)
    U8,_,_ = get_svd_of_digit(8, test_data_size)  
    U9,_,_ = get_svd_of_digit(9, test_data_size)
    U0,_,_ = get_svd_of_digit(0, test_data_size)

    #----------------------------------------------------------
    # Compare residuals for 2 self made test digits with 
    # different k values
    #----------------------------------------------------------
    
    test_digit_2 = np.array(Image.open(get_path("../test_digit_2.png"))).ravel()
    test_digit_9 = np.array(Image.open(get_path("../test_digit_9.png"))).ravel()

    print("Computing the residuals...")
    fig2 = plt.figure(figsize=(14,3))
    fig9 = plt.figure(figsize=(14,3))
    for i,k in enumerate((10,20,30),1):
        
        U= (U0[:,:k],U1[:,:k],U2[:,:k],U3[:,:k],
            U4[:,:k],U5[:,:k],U6[:,:k],U7[:,:k],U8[:,:k],U9[:,:k])
        
        res2 = compute_residuals(U, test_digit_2)
        res9 = compute_residuals(U, test_digit_9)
        
        ax = fig2.add_subplot(1,3,i)
        ax.plot(range(10), res2, 'bo-')
        ax2 = fig9.add_subplot(1,3,i)
        ax2.plot(range(10), res9, 'ro-')
    
    fig2.savefig(get_path("../residuals_test_2.png"))
    fig9.savefig(get_path("../residuals_test_9.png"))

      
    #--------------------------------------------------------
    # Test the program recognition algorithm for N=test_data_size 
    # images of each digit
    #--------------------------------------------------------
    k = 10
    U= (U0[:,:k],U1[:,:k],U2[:,:k],U3[:,:k],
        U4[:,:k],U5[:,:k],U6[:,:k],U7[:,:k],U8[:,:k],U9[:,:k])
    
    test_algorithm(U, test_data_size, tuple(range(10)))
    
    #----------------------------------------------------------
    # Show eigen patterns and test data images
    #----------------------------------------------------------
    
    # Plot some of the handwritten figures for visualizing the data
    show_digits(5, load_test_data(9, test_data_size), get_path("../example_9.png"))

    show_eigen_patterns(5, U2, get_path("../eigen_patterns_2.png"))
    show_eigen_patterns(5, U9, get_path("../eigen_patterns_9.png"))
    
    
if __name__=="__main__":
    main()
    