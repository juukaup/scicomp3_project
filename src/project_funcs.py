#!/usr/bin/env python3

import scipy.io
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import svd

def get_path(filename):
    """ Appends path to the input data directory 
    (run/nist_digits) to the given filename. """
    import sys
    import os
    return os.path.join(os.path.dirname(sys.argv[0]), "..", "run/nist_digits", filename)

def get_svd_of_digit(dig, test_data_size):
    """ This function calculates the SVD for 
    the given digit using NIST data sets. """
    
    file_name = get_path("digit"+str(dig)+".mat")
    D = scipy.io.loadmat(file_name)
    digits = np.array(D["D"])
    
    # Leave the last N=test_data_size images as testing data
    # since there's plenty of training data.
    U,S,V = svd(digits[:-test_data_size].T)
    return U,S,V

def compute_residuals(U,digit):
    """ Calculates the residuals for given digit 
    from SVD:s of every digit """
    res = np.zeros((10,))
    for i,u in enumerate(U):
        r = (u @ u.T @ digit) - digit
        res[i] = np.linalg.norm(r,2)
    return res

def recognize_digit(U,digit):
    """ Recognizes the given digit (vector) using the 
    given U-matrices from SVD of all the other digits """
    res = compute_residuals(U,digit)
    return (np.argmin(res))

def test_algorithm(U, test_data_size, numbers):
    for d in numbers:
        test_data = load_test_data(d, test_data_size)
    
        result = np.zeros((test_data_size,))
        for i in range(test_data_size):
            result[i] = recognize_digit(U,test_data[i,:])
        print("Recognizing images of digit: " + str(d))
        print("Done. Digits recognized correctly: %d/%d" % (sum(result==d), test_data_size))
        print("")

def show_digits(n, data, filename):
    fig, axes = plt.subplots(1, n, subplot_kw=dict(xticks=[], yticks=[]), figsize=(10,3))
    for ax, eig_pattern in zip(axes.flat[:n], data[:n]):
        ax.imshow(eig_pattern.reshape(28,28), cmap="gray")
    fig.savefig(filename)
    
def show_eigen_patterns(n, data, filename):
    fig, axes = plt.subplots(1, n, subplot_kw=dict(xticks=[], yticks=[]), figsize=(10,3))
    for ax, eig_pattern in zip(axes.flat[:n], data[:,:n].T):
        ax.imshow(eig_pattern.reshape(28,28))
    fig.savefig(filename)
    
def load_test_data(dig, test_data_size):
    D = scipy.io.loadmat(get_path("digit"+str(dig)+".mat"))
    digits = np.array(D["D"])
    return digits[-test_data_size:]