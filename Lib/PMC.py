import numpy
import math
import matplotlib.pyplot as plt
from ctypes import *

PATH_TO_SHARED_LIBRARY = "E:/Projects/Unity Projects/ProjetMachineLearning/Lib/MachineLearningLib/x64/Debug/MachineLearningLib.dll"


def test(lib, number):
    lib.test.argtypes = [POINTER(c_int)]
    lib.test.restype = c_int
    return lib.test(number)


def create_pmc(lib, npl, d, X, W, deltas):
    print("Init PMC")

    lib.createPMC.argtypes = [
        POINTER(c_int),
        POINTER(c_int),
        POINTER(POINTER(c_float)),
        POINTER(POINTER(c_float)),
        POINTER(POINTER(POINTER(c_float)))
    ]

    lib.createPMC.restype = None
    lib.createPMC(npl, d, X, deltas, W)


if __name__ == "__main__":
    # load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)

    # Init data
    npl = [2, 1]
    W = []
    d = []
    X = []
    deltas = []

    # cast data in c_type

    # call function
    create_pmc(lib, npl, W, d, X, deltas)
