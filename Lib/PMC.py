import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *
import time

PATH_TO_SHARED_LIBRARY = "Lib/MachineLearningLib/x64/Debug/MachineLearningLib.dll"


def test(lib):
    lib.test.argtypes = []
    lib.test.restype = c_int
    return lib.test()


def create_pmc(lib, npl):

    d = npl.copy()
    npl = (c_int * len(npl))(* npl)
    d = (c_int * len(d))(* d)
    W = (c_int * len(W))(* W)
    X = (c_float * len(X))(* X)
    deltas = (c_float * len(deltas))(* deltas)

    lib.createPMC.argtypes = [
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float)
    ]

    lib.createPMC.restype = None
    lib.createPMC(npl, d, X, deltas, W)


if __name__ == "__main__":
    # load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    # call function
    create_pmc(lib, [2, 1])
