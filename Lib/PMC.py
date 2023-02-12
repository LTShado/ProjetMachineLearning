import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *
import time

PATH_TO_SHARED_LIBRARY = "Lib/MachineLearningLib/x64/Debug/MachineLearningLib.dll"
#PATH_TO_SHARED_LIBRARY = "MachineLearningLib/x64/Debug/MachineLearningLib.dll"
################################## TESTS ##################################
############### SIMPLE TEST ###############

def runAllSimpleTest(lib):
    simpleTestLinearDataset(lib)
    simpleTestXOR(lib)

def simpleTestLinearDataset(lib):
    print("start simpleLinear")
    points = np.array([
        [1, 1],
        [2, 1],
        [2, 2]])
    classes = np.array([
        1,
        1,
        -1
    ])
    pColor = lambda yArr: ['blue' if c == 1 else 'red' for c in yArr]
    cColor = lambda pred: 'lightcyan' if pred[0] >= 0 else 'pink'

    model = createModelPMC(lib,[2, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True)
    saveModelPMC(lib, model)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor)
    print("end simpleLinear")

def simpleTestXOR(lib):
    print("start simpleXOR")
    points = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ])
    classes = np.array([
        -1,
        -1,
        1,
        1,
    ])

    pColor = lambda yArr: ['blue' if c == 1 else 'red' for c in yArr]
    cColor = lambda pred: 'lightcyan' if pred[0] >= 0 else 'pink'

    model = createModelPMC(lib,[2, 2, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True, 0.1, 100000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor,-1)
    print("end simpleXOR")



########### CLASSIFICATION TEST ###########
def runAllClassificationTest(lib):
    testLinearSimple(lib)
    testLinearMultiple(lib)
    testXOR(lib)
    testCross(lib)
    testMultiLinear3Classes(lib)
    testMultiCross(lib)

def testLinearSimple(lib):
    print("start testLinearSimple")
    points = np.array([[1, 1],[2, 3],[3, 3]])
    classes = np.array([1,-1,-1])

    pColor = lambda yArr: ['blue' if c == 1 else 'red' for c in yArr]
    cColor = lambda pred: 'lightcyan' if pred[0] >= 0 else 'pink'

    model = createModelPMC(lib,[2, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True, 0.01, 10000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor, 0.5)
    print("end testLinearSimple")

def testLinearMultiple(lib):
    print("start testLinearMultiple")
    points = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
    classes = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    pColor = lambda yArr: ['blue' if c == 1 else 'red' for c in yArr]
    cColor = lambda pred: 'lightcyan' if pred[0] >= 0 else 'pink'

    model = createModelPMC(lib,[2, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True, 0.01, 10000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor, 0.5)
    print("end testLinearMultiple")

def testXOR(lib):
    print("start testXOR")
    points = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    classes = np.array([1, 1, -1, -1])

    pColor = lambda yArr: ['blue' if c == 1 else 'red' for c in yArr]
    cColor = lambda pred: 'lightcyan' if pred[0] >= 0 else 'pink'

    model = createModelPMC(lib,[2, 2, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True, 0.05, 10000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor, -1)
    print("end testXOR")

def testCross(lib):
    print("start testCross")
    points = np.random.random((500, 2)) * 2.0 - 1.0
    classes = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in points])

    pColor = lambda yArr: ['blue' if c == 1 else 'red' for c in yArr]
    cColor = lambda pred: 'lightcyan' if pred[0] >= 0 else 'pink'

    model = createModelPMC(lib,[2, 4, 1])
    trainPMC(lib, model, points, [[c] for c in classes], True, 0.01, 100000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor, -1.5)
    print("end testCross")

def testMultiLinear3Classes(lib):
    print("start testMultiLinear3Classes")
    points = np.random.random((500, 2)) * 2.0 - 1.0
    classes = np.array([
                [1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
                [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
                [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
                [0, 0, 0]for p in points])

    points = points[[not np.all(arr == [0, 0, 0]) for arr in classes]]
    classes = classes[[not np.all(arr == [0, 0, 0]) for arr in classes]]

    def pColor3(yArr):
        color = []
        for c in yArr:
            if np.array_equal(c,[1,0,0]):
                color.append('blue')
            if np.array_equal(c,[0,1,0]):
                color.append('red')
            if np.array_equal(c,[0,0,1]):
                color.append('green')
        return color

    def cColor3(pred):
        m = max(pred[0:3])
        if pred[0]>=0 and pred[0]==m:
            return 'lightcyan'
        elif pred[1]>=0 and pred[1]==m:
            return 'pink'
        elif pred[2]>=0 and pred[2]==m:
            return 'lightgreen'
        else:
            return 'white'

    pColor = pColor3
    cColor = cColor3

    model = createModelPMC(lib,[2, 3])
    trainPMC(lib, model, points, classes, True, 0.01, 100000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor, -1.5)
    print("end testMultiLinear3Classes")

def testMultiCross(lib):
    print("start testMultiCross")
    points = np.random.random((1000, 2)) * 2.0 - 1.0
    classes = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in points])

    def pColor3(yArr):
        color = []
        for c in yArr:
            if np.array_equal(c,[1,0,0]):
                color.append('blue')
            if np.array_equal(c,[0,1,0]):
                color.append('red')
            if np.array_equal(c,[0,0,1]):
                color.append('green')
        return color

    def cColor3(pred):
        m = max(pred[0:3])
        if pred[0]>=0 and pred[0]==m:
            return 'lightcyan'
        elif pred[1]>=0 and pred[1]==m:
            return 'pink'
        elif pred[2]>=0 and pred[2]==m:
            return 'lightgreen'
        else:
            return 'white'

    pColor = pColor3
    cColor = cColor3

    model = createModelPMC(lib,[2, 30, 30, 3])
    trainPMC(lib, model, points, classes, True, 0.1, 1000000)
    displayPredictClassif(lib, model, points, classes, True, pColor, cColor, -1.5)
    saveModelPMC(lib, model)
    print("end testMultiCross")


############# REGRESSION TEST #############

def runAllRegressionTest(lib):
    testLinearSimple2D(lib)
    testNonLinearSimple2D(lib)
    testLinearSimple3D(lib)
    testLinearTricky3D(lib)
    testNonLinearSimple3D(lib)

def testLinearSimple2D(lib):
    print("start testLinearSimple2D")
    points = np.array([[1],[2]])
    classes = np.array([2,3])

    model = createModelPMC(lib,[1, 1])
    trainPMC(lib, model, points, [[c] for c in classes], False, 0.01, 10000)
    displayPredictRegression2D(lib, model, points, classes, False, 0, 4)
    print("end testLinearSimple2D")

def testNonLinearSimple2D(lib):
    print("start testNonLinearSimple2D")
    points = np.array([[1],[2],[3]])
    classes = np.array([2,3,2.5])

    model = createModelPMC(lib,[1,4,1])
    trainPMC(lib, model, points, [[c] for c in classes], False, 0.01, 10000)
    displayPredictRegression2D(lib, model, points, classes, False, 0, 4)
    print("end testNonLinearSimple2D")

def testLinearSimple3D(lib):
    print("start testLinearSimple3D")
    points = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    classes = np.array([
        2,
        3,
        2.5
    ])

    model = createModelPMC(lib,[2,1])
    trainPMC(lib, model, points, [[c] for c in classes], False, 0.01, 10000)
    displayPredictRegression3D(lib, model, points, classes, False, 0, 4)
    print("end testLinearSimple3D")

def testLinearTricky3D(lib):
    print("start testLinearTricky3D")
    points = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    classes = np.array([
        1,
        2,
        3
    ])

    model = createModelPMC(lib,[2,1])
    trainPMC(lib, model, points, [[c] for c in classes], False, 0.01, 10000)
    displayPredictRegression3D(lib, model, points, classes, False, 0, 4)
    print("end testLinearTricky3D")

def testNonLinearSimple3D(lib):
    print("start testNonLinearSimple3D")
    points = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    classes = np.array([
        2,
        1,
        -2,
        -1
    ])

    model = createModelPMC(lib,[2,2,1])
    trainPMC(lib, model, points, [[c] for c in classes], False, 0.01, 10000)
    displayPredictRegression3D(lib, model, points, classes, False, 0, 4)
    print("end testNonLinearSimple3D")


################################ FUNCTIONS ################################

def test(lib):
    lib.test.argtypes = []
    lib.test.restype = c_int
    return lib.test()

def createModelPMC(lib, npl):

    sizeNpl = len(npl)
    maxN = max(npl) + 1
    d = (c_int * len(npl))(* npl)    
    X = (c_float * (sizeNpl*maxN))()
    deltas = (c_float * (sizeNpl*maxN))()
    W = (c_float * (sizeNpl*maxN*maxN))()
    sizeNpl = (c_int *1)(sizeNpl)
    maxN = (c_int *1)(maxN)

    lib.createModelPMC.argtypes = [
        POINTER(c_int),   #npl
        POINTER(c_int),   #sizeNpl
        POINTER(c_int),   #maxN
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
        POINTER(c_int),   #sizeNpl
        POINTER(c_int),   #maxN
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
        POINTER(c_int),   #sizeNpl
        POINTER(c_int),   #maxN
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

def saveModelPMC(lib, model, filename="pmcModel.txt"):
    tmp = []
    tmp.append(cast(model['d'],c_void_p))
    tmp.append(cast(model['sizeNpl'],c_void_p))
    tmp.append(cast(model['maxN'],c_void_p))
    tmp.append(cast(model['X'],c_void_p))
    tmp.append(cast(model['deltas'],c_void_p))
    tmp.append(cast(model['W'],c_void_p))

    filename = "SavedModel/" + filename
    
    model_void = (c_void_p * 6)(*tmp)

    byte_filename = filename.encode('utf-8')

    lib.saveModelPMC.argtypes = [
        POINTER(c_void_p), #model
        c_char_p,          #filename
    ]
    lib.saveModelPMC.restype = None
    lib.saveModelPMC(
        model_void,
        byte_filename
        )

def displayPredictClassif(lib, model, xTrain, yTrain, isClassification, pColor, cColor, offset = 0):
    test_points = []
    test_colors = []
    pointColors = pColor(yTrain)

    for row in range(0, 300):
        for col in range(0, 300):
            p = np.array([col / 100 + offset, row / 100 + offset])
            pred  = predictPMC(lib,model, p, isClassification)
            c = cColor(pred)
            test_points.append(p)
            test_colors.append(c)
    test_points = np.array(test_points)
    test_colors = np.array(test_colors)

    plt.figure()
    #plt.subplot(212)
    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
    plt.scatter(xTrain[:, 0], xTrain[:, 1], c=pointColors)
    plt.show()

def displayPredictRegression2D(lib, model, xTrain, yTrain, isClassification, start, end):
    test_X = []
    test_Y = []
    
    for x in np.arange(start,end,0.1):
        p = np.array([x])
        pred = predictPMC(lib,model, p, isClassification)
        test_X.append(x)
        test_Y.append(pred[0])


    test_Y = np.array(test_Y)

    plt.figure()
    #plt.subplot(212)
    plt.scatter(xTrain, yTrain, color='blue')
    plt.plot(test_X,test_Y, color='black')
    plt.show()

def displayPredictRegression3D(lib, model, xTrain, yTrain, isClassification, start, end):
    test_X = []
    test_Y = []
    
    for x in np.arange(start,end,0.1):
        for z in np.arange(start,end,0.1):
            p = np.array([x,z])
            pred = predictPMC(lib,model, p, isClassification)
            test_X.append([x,z])
            test_Y.append(pred[0])

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    ax = plt.figure().add_subplot(projection='3d')
    #plt.subplot(212)
    
    ax.scatter(xTrain[:,0],xTrain[:,1], yTrain, color='blue')
    ax.scatter(test_X[:,0],test_X[:,1], test_Y, s=0.5, color='black')
    plt.show()

if __name__ == "__main__":
    # load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    # call function
    #runAllSimpleTest(lib)
    #runAllClassificationTest(lib)
    #runAllRegressionTest(lib)

    simpleTestLinearDataset(lib)
    
    
