import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *

PATH_TO_SHARED_LIBRARY = "MachineLearningLib/x64/Debug/MachineLearningLib.dll"


def Linear_Simple_test(lib):
    X = np.array([
      [1, 1],
      [2, 3],
      [3, 3]
    ])
    Y = np.array([
          1,
          -1,
          -1
    ])
    
    #affichage_avant_test(X[0, 0],X[0, 1],X[1:3,0],X[1:3,1],0,0,2)
    print("linear simple")

def Linear_Multiple_test(lib):
    X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    #affichage_avant_test(X[0:50, 0],X[0:50, 1],X[50:100,0],X[50:100,1],0,0,2)
    print("linear multiple")

def XOR_test(lib):
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    #affichage_avant_test(X[0:2, 0],X[0:2, 1],X[2:4,0],X[2:4,1],0,0,2)
    print("XOR")

def Cross_test(lib):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

    #affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:,0]
     #           ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:,1]
      #          ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:,0]
       #         ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:,1]
        #                 ,0,0,2)
    print("Cross")
    
def Linear_Multiple_3_test(lib):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X])

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]


    #affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0]
     #                   ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1]
      #                  ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0]
       #                 ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1]
        #                ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0]
         #               ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1]
          #               ,3)
    print("linear multiple 3 classes")

def Multi_Cross_test(lib):
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25
    and abs(p[1] % 0.5) > 0.25 else [0, 1, 0]
    if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25
    else [0, 0, 1] for p in X])

    affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0]
                        , np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1]
                         ,3)

    print(test_lib(lib,10,2))
    print("multi cross")

def test_lib(lib,x,y):
    lib.division.argtypes = [c_int, c_int]
    lib.division.restype = c_int
    return lib.division(x,y)

def affichage_avant_test(a,b,c,d,e,f,num):
    if(num==2):
        plt.scatter(a, b, color='blue')
        plt.scatter(c, d, color='red')
    elif(num==3):
        plt.scatter(a, b, color='blue')
        plt.scatter(c, d, color='red')
        plt.scatter(e, f, color='green')
    plt.show()
    plt.clf()

if __name__ == "__main__":
    # Load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    
    #Cas_Test
    Linear_Simple_test(lib)
    Linear_Multiple_test(lib)
    XOR_test(lib)
    Cross_test(lib)
    Linear_Multiple_3_test(lib)
    Multi_Cross_test(lib)
    
    print('test')
