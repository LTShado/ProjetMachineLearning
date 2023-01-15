import numpy
import math
import matplotlib.pyplot as plt
from ctypes import *

PATH_TO_SHARED_LIBRARY = "E:/Projects/Unity Projects/ProjetMachineLearning/Lib/MachineLearningLib/x64/Debug/MachineLearningLib.dll"


def test(lib):
    lib.test.argtypes = []
    lib.test.restype = c_int
    return lib.test()


def create_pmc(lib, npl):

    npl = (c_int * len(npl))(* npl)
    d = npl.copy()
    d = (c_int * len(d))(* d)

    W = []
    X = []
    deltas = []

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
    # call function
    create_pmc(lib, [2, 1])
