import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *
import time

PATH_TO_SHARED_LIBRARY = "Lib/MachineLearningLib/x64/Debug/MachineLearningLib.dll"

################################## TESTS ##################################
def test(lib):
    lib.test.argtypes = []
    lib.test.restype = c_int
    return lib.test()

def simpleTestLinearDataset(lib):
    points = np.array([
        [1, 1],
        [2, 1],
        [2, 2]])
    classes = np.array([
        1,
        1,
        -1
    ])

    colors = ['blue' if c == 1 else 'red' for c in classes]
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()
    plt.clf()

    model = createModelPMC(lib,[2, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True)

    displayPredict(lib, model, points, classes)

    print("linear simple")


################################ FUNCTIONS ################################

def createModelPMC(lib, npl):

    sizeNpl = len(npl)
    maxN = max(npl) + 1
    d = (c_int * len(npl))(* npl)    
    X = (c_float * (sizeNpl*maxN))()
    deltas = (c_float * (sizeNpl*maxN))()
    W = (c_float * (sizeNpl*maxN*maxN))()

    lib.createModelPMC.argtypes = [
        POINTER(c_int),   #npl
        c_int,            #sizeNpl
        c_int,            #maxN
        POINTER(c_float), #X
        POINTER(c_float), #deltas
        POINTER(c_float)  #W
    ]
    lib.createModelPMC.restype = None

    lib.createModelPMC(d, sizeNpl, maxN, X, deltas, W)
    resultArray = {"d":d, "sizeNpl":sizeNpl, "maxN":maxN, "X":X, "deltas":deltas, "W":W}
    return resultArray

def trainPMC(lib, model, xTrain, yTrain, isClassification, alpha = 0.01, nbIter = 1000):
    print('train')

    sizeT = len(xTrain)
    #Flatten xTrain and yTrain
    sizeDataXTrain = len(xTrain[0])
    xTrainFlat = []
    for arr in xTrain:
        for value in arr:
            xTrainFlat.append(value)
    xTrainFlat = (c_float * len(xTrainFlat))(*xTrainFlat)

    sizeDataYTrain = len(yTrain[0])
    yTrainFlat = []
    for arr in yTrain:
        for value in arr:
            yTrainFlat.append(value)
    yTrainFlat = (c_float * len(yTrainFlat))(*yTrainFlat)

    lib.trainPMC.argtypes = [
        c_int,            #sizeT
        POINTER(c_float), #xTrain
        c_int,            #sizeDataXtrain
        POINTER(c_float), #yTrain
        c_int,            #sizeDataYTrain
        c_bool,           #isClassification
        c_float,          #alpha
        c_int,            #nbIter
        POINTER(c_int),   #d
        c_int,            #sizeNpl
        c_int,            #maxN
        POINTER(c_float), #X
        POINTER(c_float), #deltas
        POINTER(c_float)  #W
    ]

    lib.trainPMC.restype = None
    lib.trainPMC(
        sizeT,xTrainFlat,sizeDataXTrain,yTrainFlat,sizeDataYTrain,
        isClassification, alpha, nbIter,
        model['d'],model['sizeNpl'],model['maxN'], model['X'], model['deltas'], model['W']
        )

    print('finish')

def predictPMC(lib, model, inputs, isClassification):

    inputs = (c_float * len(inputs))(*inputs)
    lib.predictPMC.argtypes = [
        POINTER(c_float), #inputs
        c_bool,           #isClassification
        POINTER(c_int),   #d
        c_int,            #sizeNpl
        c_int,            #maxN
        POINTER(c_float), #X
        POINTER(c_float)  #W
    ]
    lib.predictPMC.restype = POINTER(c_float)
    resArr = lib.predictPMC(
        inputs,
        isClassification,
        model['d'],model['sizeNpl'],model['maxN'], model['X'], model['W']
        )
    return resArr

def displayPredict(lib, model, xTrain, yTrain):
    print('resultat')
    test_points = []
    test_colors = []
    colors = ['blue' if c == 1 else 'red' for c in yTrain]

    for row in range(0, 300):
        for col in range(0, 300):
            p = np.array([col / 100, row / 100])
            c = 'lightcyan' if predictPMC(lib,model, p, True)[0] >= 0 else 'pink'
            test_points.append(p)
            test_colors.append(c)
    test_points = np.array(test_points)
    test_colors = np.array(test_colors)

    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
    plt.scatter(xTrain[:, 0], xTrain[:, 1], c=colors)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    # call function
    simpleTestLinearDataset(lib)
    
