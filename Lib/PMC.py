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


def create_model_pmc(lib, npl):

    sizeNpl = len(npl)
    maxN = max(npl) + 1
    d = (c_int * len(npl))(* npl)    
    X = (c_float * (sizeNpl*maxN))()
    deltas = (c_float * (sizeNpl*maxN))()
    W = (c_float * (sizeNpl*maxN*maxN))()

    lib.createModelPMC.argtypes = [
        POINTER(c_int), #npl
        c_int, #sizeNpl
        c_int, #maxN
        POINTER(c_float), #X
        POINTER(c_float), #deltas
        POINTER(c_float) #W
    ]
    lib.createModelPMC.restype = None
    lib.createModelPMC(d, sizeNpl, maxN, X, deltas, W)
    resultArray = {"d":d, "sizeNpl":sizeNpl, "maxN":maxN, "X":X, "deltas":deltas, "W":W}
    return resultArray


if __name__ == "__main__":
    # load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    # call function
    r = create_model_pmc(lib, [2, 1])
    print(r['d'][0])
