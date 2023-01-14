import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *

PATH_TO_SHARED_LIBRARY = "MachineLearningLib/x64/Debug/MachineLearningLib.dll"


def Linear_Simple_test(lib):
    X_arr = [
      [1, 1],
      [2, 3],
      [3, 3]
    ]
    X = np.array(X_arr)
    
    Y_arr = [
          1,
          -1,
          -1
    ]
    Y = np.array(Y_arr)

    #affichage_avant_test(X[0, 0],X[0, 1],X[1:3,0],X[1:3,1],0,0,2)
    size = 2
    
    W = create_model(lib,size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])


    D = train_rosenblatt_linear(lib,W,len(W_transfo),X_arr,len(X_arr),Y_arr,len(Y_arr),1000,0.1,len(X_arr))
    D_ptr = cast(D, POINTER(c_float))

    D_transfo = []
    for i in range(size+1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr,X,Y,2)

    print("linear simple")

def Linear_Multiple_test(lib):
    X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
    X_arr = X.tolist()

    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    print(Y)

    Y_one_dim = []
    for num in Y:
        Y_one_dim.append(num[0])

    #affichage_avant_test(X[0:50, 0],X[0:50, 1],X[50:100,0],X[50:100,1],0,0,2)

    size = 2
    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))
    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(X_arr), Y_one_dim, len(Y_one_dim), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))
    D_transfo = []
    for i in range(size + 1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr,X,Y,2)

    print("linear multiple")

def XOR_test(lib):
    X_arr =[[1, 0], [0, 1], [0, 0], [1, 1]]
    X = np.array(X_arr)

    Y_arr = [1, 1, -1, -1]
    Y = np.array(Y_arr)

    #affichage_avant_test(X[0:2, 0],X[0:2, 1],X[2:4,0],X[2:4,1],0,0,2)

    size = 2
    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))
    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))
    D_transfo = []
    for i in range(size + 1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr,X,Y,2)
    print("XOR")

def Cross_test(lib):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    X_arr = X.tolist()

    Y_arr = [1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X]
    Y = np.array(Y_arr)
    print(X_arr)

    #affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:,0]
     #           ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:,1]
      #          ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:,0]
       #         ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:,1]
        #                 ,0,0,2)

    size = 2
    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))
    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))
    D_transfo = []
    for i in range(size + 1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr,X,Y,2)
    print("Cross")
    
def Linear_Multiple_3_test(lib):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X])

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    print('X',X)
    print('Y',Y)

    affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0]
                       ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0]
                        ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1]
                         ,3)
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

    print("multi cross")


#################################################


def create_model(lib,size):
    print('Create model')
    lib.create_model_linear.argtypes = [c_int]
    lib.create_model_linear.restype = POINTER(c_float)
    return lib.create_model_linear(size)

def ReadArray(lib,arr):

    arr_t = (c_float* len(arr))(*arr)
    
    lib.ReadArrayValue.argtypes = [POINTER(c_float)]
    lib.ReadArrayValue.restype = c_int
    return lib.ReadArrayValue(arr_t)

def Testlib(lib):
    
    lib.test.argtypes = None
    lib.test.restype = c_int
    return lib.test()

def train_rosenblatt_linear(lib,model,model_size,X,Xlen,Y,Ylen,count,step,size):
    print('train')

    X_one_dim = []
    for num in X:
        X_one_dim.append(num[0])
        X_one_dim.append(num[1])
    X_one_dim = (c_float* len(X_one_dim))(*X_one_dim)
    
    Y = (c_float* Ylen)(*Y)
    
    lib.train_rosenblatt_linear.argtypes = [POINTER(c_float),c_int,POINTER(c_float),POINTER(c_float),c_int,c_float,c_int]
    
    lib.train_rosenblatt_linear.restype = POINTER(c_float)
    print('finish')
    return lib.train_rosenblatt_linear(model,model_size,X_one_dim,Y,count,step,len(X_one_dim))

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

def affichage_resultat(model,points,classes,num):
    print('resultat')
    if(num==2):
        colors = ['blue' if c == 1 else 'red' for c in classes]
        test_points = []
        test_colors = []
        for row in range(0, 300):
            for col in range(0, 300):
                p = np.array([col / 100, row / 100])
                c = 'lightcyan' if np.matmul(np.transpose(model), np.array([1.0, *p])) >= 0 else 'pink'
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
        plt.scatter(points[:, 0], points[:, 1], c=colors)
        plt.show()

    #elif(num==3):
     #   plt.scatter(a, b, color='blue')
      #  plt.scatter(c, d, color='red')
       # plt.scatter(e, f, color='green')
    plt.show()
    plt.clf()

if __name__ == "__main__":
    # Load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    
    #Cas_Test
    #Linear_Simple_test(lib)
    #Linear_Multiple_test(lib)
    #XOR_test(lib)
    #Cross_test(lib)
    Linear_Multiple_3_test(lib)
    #Multi_Cross_test(lib)
