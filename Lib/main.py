import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *
import time
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

PATH_TO_SHARED_LIBRARY = "MachineLearningLib/x64/Debug/MachineLearningLib.dll"

##Classification
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

    # affichage_avant_test(X[0, 0],X[0, 1],X[1:3,0],X[1:3,1],0,0,2)
    size = 2

    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(
        X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))

    D_transfo = []
    for i in range(size+1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr, X, Y, 2)

    print("linear simple")


def Linear_Multiple_test(lib):
    X = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]),
                       np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    X_arr = X.tolist()

    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    print(Y)

    Y_one_dim = []
    for num in Y:
        Y_one_dim.append(num[0])

    # affichage_avant_test(X[0:50, 0],X[0:50, 1],X[50:100,0],X[50:100,1],0,0,2)

    size = 2
    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))
    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(
        X_arr), Y_one_dim, len(Y_one_dim), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))
    D_transfo = []
    for i in range(size + 1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr, X, Y, 2)

    print("linear multiple")


def XOR_test(lib):
    X_arr = [[1, 0], [0, 1], [0, 0], [1, 1]]
    X = np.array(X_arr)

    Y_arr = [1, 1, -1, -1]
    Y = np.array(Y_arr)

    # affichage_avant_test(X[0:2, 0],X[0:2, 1],X[2:4,0],X[2:4,1],0,0,2)

    size = 2
    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))
    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(
        X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))
    D_transfo = []
    for i in range(size + 1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr, X, Y, 2)
    print("XOR")


def Cross_test(lib):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    X_arr = X.tolist()

    Y_arr = [1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X]
    Y = np.array(Y_arr)
    print(X_arr)

    # affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:,0]
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

    D = train_rosenblatt_linear(lib, W, len(W_transfo), X_arr, len(
        X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr))
    D_ptr = cast(D, POINTER(c_float))
    D_transfo = []
    for i in range(size + 1):
        D_transfo.append(D_ptr[i])
    D_transfo_arr = np.array(D_transfo)

    affichage_resultat(D_transfo_arr, X, Y, 2)
    print("Cross")


def Linear_Multiple_3_test(lib):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0]for p in X])
    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    X_arr = X.tolist()
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    print('Y', len(Y))
    print(len(X))

    # affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0]
    #                    ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1]
    #                      ,3)

    Y1 = np.array([1 if col[0] == 1 else -1 for col in Y])
    Y1_arr = Y1.tolist()
    Y2 = np.array([1 if col[1] == 1 else -1 for col in Y])
    Y2_arr = Y2.tolist()
    Y3 = np.array([1 if col[2] == 1 else -1 for col in Y])
    Y3_arr = Y3.tolist()

    size = 2
    W1 = create_model(lib, size)
    W1_ptr = cast(W1, POINTER(c_float))
    W1_transfo = []
    for i in range(size + 1):
        W1_transfo.append(W1_ptr[i])
    print('w1', W1_transfo)

    W2 = create_model(lib, size)
    W2_ptr = cast(W2, POINTER(c_float))
    W2_transfo = []
    for i in range(size + 1):
        W2_transfo.append(W2_ptr[i])
    print('w2', W2_transfo)

    W3 = create_model(lib, size)
    W3_ptr = cast(W3, POINTER(c_float))
    W3_transfo = []
    for i in range(size + 1):
        W3_transfo.append(W3_ptr[i])
    print('w3', W3_transfo)

    D1 = train_rosenblatt_linear(lib, W1, len(W1_transfo), X_arr, len(
        X_arr), Y1_arr, len(Y1_arr), 1000, 0.1, len(X_arr))
    D2 = train_rosenblatt_linear(lib, W2, len(W2_transfo), X_arr, len(
        X_arr), Y2_arr, len(Y2_arr), 1000, 0.1, len(X_arr))
    D3 = train_rosenblatt_linear(lib, W3, len(W3_transfo), X_arr, len(
        X_arr), Y3_arr, len(Y3_arr), 1000, 0.1, len(X_arr))

    D1_ptr = cast(D1, POINTER(c_float))
    D1_transfo = []
    for i in range(size + 1):
        D1_transfo.append(D1_ptr[i])
    D1_transfo_arr = np.array(D1_transfo)
    print('D1', D1_transfo_arr)

    D2_ptr = cast(D2, POINTER(c_float))
    D2_transfo = []
    for i in range(size + 1):
        D2_transfo.append(D2_ptr[i])
    D2_transfo_arr = np.array(D2_transfo)
    print('D2', D2_transfo_arr)

    D3_ptr = cast(D3, POINTER(c_float))
    D3_transfo = []
    for i in range(size + 1):
        D3_transfo.append(D3_ptr[i])
    D3_transfo_arr = np.array(D3_transfo)
    print('D3', D3_transfo_arr)

    predict1 = []
    predict2 = []
    predict3 = []
    points = []

    for row in range(-50, 51):
        for col in range(-50, 51):
            p = np.array([col / 50, row / 50])
            if np.matmul(np.transpose(D1_transfo_arr), np.array([1.0, *p])) >= 0:
                predict1.append(1)
            else:
                predict1.append(-1)

            if np.matmul(np.transpose(D2_transfo_arr), np.array([1.0, *p])) >= 0:
                predict2.append(1)
            else:
                predict2.append(-1)

            if np.matmul(np.transpose(D3_transfo_arr), np.array([1.0, *p])) >= 0:
                predict3.append(1)
            else:
                predict3.append(-1)
            points.append(p)

    points = np.array(points)

    predicts = predict1
    for i in range(len(predict1)):
        if predict1[i] >= 0:
            predicts[i] = 0
        if predict2[i] >= 0:
            predicts[i] = 1
        if predict3[i] >= 0:
            predicts[i] = 2
        if predict1[i] < 0 and predict2[i] < 0 and predict3[i] < 0:
            predicts[i] = 3

    colors = ['cyan' if c == 0 else ('pink' if c == 1 else (
        'lime' if c == 2 else 'black')) for c in predicts]
    colors = np.array(colors)

    plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(
                    lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(
                    lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(
                    lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.show()
    plt.clf()

    # print(predict1)

    print("linear multiple 3 classes")


def Multi_Cross_test(lib):
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    X_arr = X.tolist()
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25
                  and abs(p[1] % 0.5) > 0.25 else [0, 1, 0]
                  if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25
                  else [0, 0, 1] for p in X])

    Y1 = np.array([1 if col[0] == 1 else -1 for col in Y])
    Y1_arr = Y1.tolist()
    Y2 = np.array([1 if col[1] == 1 else -1 for col in Y])
    Y2_arr = Y2.tolist()
    Y3 = np.array([1 if col[2] == 1 else -1 for col in Y])
    Y3_arr = Y3.tolist()

    # affichage_avant_test(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1]
    #                     ,np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0]
    #                     , np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1]
    #                      ,3)

    size = 2
    W1 = create_model(lib, size)
    W1_ptr = cast(W1, POINTER(c_float))
    W1_transfo = []
    for i in range(size + 1):
        W1_transfo.append(W1_ptr[i])
    print('w1', W1_transfo)

    W2 = create_model(lib, size)
    W2_ptr = cast(W2, POINTER(c_float))
    W2_transfo = []
    for i in range(size + 1):
        W2_transfo.append(W2_ptr[i])
    print('w2', W2_transfo)

    W3 = create_model(lib, size)
    W3_ptr = cast(W3, POINTER(c_float))
    W3_transfo = []
    for i in range(size + 1):
        W3_transfo.append(W3_ptr[i])
    print('w3', W3_transfo)

    D1 = train_rosenblatt_linear(lib, W1, len(W1_transfo), X_arr, len(X_arr), Y1_arr, len(Y1_arr), 1000, 0.1,
                                 len(X_arr))
    D2 = train_rosenblatt_linear(lib, W2, len(W2_transfo), X_arr, len(X_arr), Y2_arr, len(Y2_arr), 1000, 0.1,
                                 len(X_arr))
    D3 = train_rosenblatt_linear(lib, W3, len(W3_transfo), X_arr, len(X_arr), Y3_arr, len(Y3_arr), 1000, 0.1,
                                 len(X_arr))

    D1_ptr = cast(D1, POINTER(c_float))
    D1_transfo = []
    for i in range(size + 1):
        D1_transfo.append(D1_ptr[i])
    D1_transfo_arr = np.array(D1_transfo)
    print('D1', D1_transfo_arr)

    D2_ptr = cast(D2, POINTER(c_float))
    D2_transfo = []
    for i in range(size + 1):
        D2_transfo.append(D2_ptr[i])
    D2_transfo_arr = np.array(D2_transfo)
    print('D2', D2_transfo_arr)

    D3_ptr = cast(D3, POINTER(c_float))
    D3_transfo = []
    for i in range(size + 1):
        D3_transfo.append(D3_ptr[i])
    D3_transfo_arr = np.array(D3_transfo)
    print('D3', D3_transfo_arr)

    predict1 = []
    predict2 = []
    predict3 = []
    points = []

    for row in range(-50, 51):
        for col in range(-50, 51):
            p = np.array([col / 50, row / 50])
            if np.matmul(np.transpose(D1_transfo_arr), np.array([1.0, *p])) >= 0:
                predict1.append(1)
            else:
                predict1.append(-1)

            if np.matmul(np.transpose(D2_transfo_arr), np.array([1.0, *p])) >= 0:
                predict2.append(1)
            else:
                predict2.append(-1)

            if np.matmul(np.transpose(D3_transfo_arr), np.array([1.0, *p])) >= 0:
                predict3.append(1)
            else:
                predict3.append(-1)
            points.append(p)

    points = np.array(points)

    predicts = predict1
    for i in range(len(predict1)):
        if predict1[i] >= 0:
            predicts[i] = 0
        if predict2[i] >= 0:
            predicts[i] = 1
        if predict3[i] >= 0:
            predicts[i] = 2
        if predict1[i] < 0 and predict2[i] < 0 and predict3[i] < 0:
            predicts[i] = 3

    colors = ['cyan' if c == 0 else ('pink' if c == 1 else (
        'lime' if c == 2 else 'black')) for c in predicts]
    colors = np.array(colors)

    plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(
                    lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(
                    lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(
                    lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.show()
    plt.clf()
    print("multi cross")

##Regression

def Linear_Simple_2D(lib):
    X = np.array([
          [1],
          [2]
    ])
    X_arr = X.tolist()
    Y = np.array([
          2,
          3
    ])
    Y_arr = Y.tolist()

    plt.scatter(X, Y)
    plt.show()
    plt.clf()

    size = 2;

    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_regression_linear(lib, W, len(W_transfo), X.flatten(), len(X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr),1)

    print('linear simple 2d')

def Non_Linear_Simple_2D(lib):
    X = np.array([
        [1],
        [2],
        [3]
    ])
    X_arr = X.tolist()
    Y = np.array([
        2,
        3,
        2.5
    ])
    Y_arr = Y.tolist()
    plt.scatter(X, Y)
    plt.show()
    plt.clf()

    size = 2;

    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_regression_linear(lib, W, len(W_transfo), X.flatten(), len(X_arr), Y_arr, len(Y_arr), 1000, 0.1,
                                len(X_arr),1)

    print('non linear simple 2d')

def Linear_Simple_3D(lib):
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    X_arr = X.tolist()
    Y = np.array([
        2,
        3,
        2.5
    ])
    Y_arr = Y.tolist()
    size = 2;

    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_regression_linear(lib, W, len(W_transfo), X.flatten(), len(X_arr), Y_arr, len(Y_arr), 1000, 0.1, len(X_arr),2)
    D_ptr = cast(D, POINTER(c_float))

    D_transfo = []
    for i in range(size):
        D_transfo.append(W_ptr[i])

    points_x = []
    points_y = []
    points_z = []

    for i in range(10, 31):
        for j in range(10, 31):
            points_x.append(float(i / 10))
            points_y.append(float(j / 10))
            points_z.append(float(predict_regression(lib, D,len(D_transfo), [i / 10, j / 10])))
            #print('predict ',predict_regression(lib, D,len(D_transfo), [i / 10, j / 10]))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(points_x, points_y, points_z)
    ax.scatter(X[:, 0], X[:, 1], Y, c="orange", s=100)
    plt.show()
    plt.clf()

    print('linear simple 3d')

def Linear_Tricky_3D(lib):
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    X_arr = X.tolist()
    print(len(X))
    Y = np.array([
        1,
        2,
        3
    ])
    Y_arr = Y.tolist()

    size = 2;

    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_regression_linear(lib, W, len(W_transfo), X.flatten(), len(X_arr), Y_arr, len(Y_arr), 1000, 0.1,
                                len(X_arr), 2)
    D_ptr = cast(D, POINTER(c_float))

    D_transfo = []
    for i in range(size):
        D_transfo.append(W_ptr[i])

    points_x = []
    points_y = []
    points_z = []

    for i in range(10, 31):
        for j in range(10, 31):
            points_x.append(float(i / 10))
            points_y.append(float(j / 10))
            points_z.append(float(predict_regression(lib, D, len(D_transfo), [i / 10, j / 10])))
            # print('predict ',predict_regression(lib, D,len(D_transfo), [i / 10, j / 10]))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(points_x, points_y, points_z)
    ax.scatter(X[:, 0], X[:, 1], Y, c="orange", s=100)
    plt.show()
    plt.clf()

    print('linear tricky 3d')

def Non_Linear_Simple_3D(lib):
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    X_arr = X.tolist()
    Y = np.array([
        2,
        1,
        -2,
        -1
    ])
    Y_arr = Y.tolist()

    size = 2;

    W = create_model(lib, size)
    W_ptr = cast(W, POINTER(c_float))

    W_transfo = []
    for i in range(size + 1):
        W_transfo.append(W_ptr[i])

    D = train_regression_linear(lib, W, len(W_transfo), X.flatten(), len(X_arr), Y_arr, len(Y_arr), 1000, 0.1,
                                len(X_arr), 2)
    D_ptr = cast(D, POINTER(c_float))

    D_transfo = []
    for i in range(size):
        D_transfo.append(W_ptr[i])

    points_x = []
    points_y = []
    points_z = []

    for i in range(0, 11):
        for j in range(0, 11):
            points_x.append(float(i / 10))
            points_y.append(float(j / 10))
            points_z.append(float(predict_regression(lib, D, len(D_transfo), [i / 10, j / 10])))
            # print('predict ',predict_regression(lib, D,len(D_transfo), [i / 10, j / 10]))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(points_x, points_y, points_z)
    ax.scatter(X[:, 0], X[:, 1], Y, c="orange", s=100)
    plt.show()
    plt.clf()

    print('non linear simple 3d')

#################################################


def create_model(lib, size):
    print('Create model')
    lib.create_model_linear.argtypes = [c_int]
    lib.create_model_linear.restype = POINTER(c_float)
    return lib.create_model_linear(size)


def ReadArray(lib, arr):

    arr_t = (c_float * len(arr))(*arr)

    lib.ReadArrayValue.argtypes = [POINTER(c_float)]
    lib.ReadArrayValue.restype = c_int
    return lib.ReadArrayValue(arr_t)


def Testlib(lib):

    lib.test.argtypes = None
    lib.test.restype = c_int
    return lib.test()


def train_rosenblatt_linear(lib, model, model_size, X, Xlen, Y, Ylen, count, step, size):
    print('train')

    X_one_dim = []
    for num in X:
        X_one_dim.append(num[0])
        X_one_dim.append(num[1])
    X_one_dim = (c_float * len(X_one_dim))(*X_one_dim)

    Y = (c_float * Ylen)(*Y)

    lib.train_rosenblatt_linear.argtypes = [POINTER(c_float), c_int, POINTER(
        c_float), POINTER(c_float), c_int, c_float, c_int]

    lib.train_rosenblatt_linear.restype = POINTER(c_float)
    print('finish')
    return lib.train_rosenblatt_linear(model, model_size, X_one_dim, Y, count, step, len(X_one_dim))

def train_regression_linear(lib, model, model_size, X, Xlen, Y, Ylen, count, step, size, dim):
    print('train')
    x_modif = [float(i) for i in X]

    print(x_modif)
    X = (len(X) * c_float)(*X)
    Y = (c_float * Ylen)(*Y)
    #
    lib.train_regression_linear.argtypes = [POINTER(c_float), c_int, POINTER(
        c_float), POINTER(c_float),c_int, c_int, c_float, c_int, c_int]

    lib.train_regression_linear.restype = POINTER(c_float)
    print('finish')
    return lib.train_regression_linear(model, model_size, X, Y,len(Y), count, step, len(X), dim)

def predict_regression(lib,model,size,value):
    inputs_float = [float(i) for i in value]
    inputs_type = len(inputs_float) * c_float

    lib.predict_regression.argtypes = [POINTER(c_float),c_int, inputs_type]

    lib.predict_regression.restype = c_float

    return lib.predict_regression(model,size, inputs_type(*inputs_float))

def affichage_avant_test(a, b, c, d, e, f, num):
    if (num == 2):
        plt.scatter(a, b, color='blue')
        plt.scatter(c, d, color='red')
    elif (num == 3):
        plt.scatter(a, b, color='blue')
        plt.scatter(c, d, color='red')
        plt.scatter(e, f, color='green')
    plt.show()
    plt.clf()


def affichage_resultat(model, points, classes, num):
    print('resultat')
    if (num == 2):
        colors = ['blue' if c == 1 else 'red' for c in classes]
        test_points = []
        test_colors = []
        for row in range(-100, 300):
            for col in range(-100, 300):
                p = np.array([col / 100, row / 100])
                c = 'lightcyan' if np.matmul(np.transpose(
                    model), np.array([1.0, *p])) >= 0 else 'pink'
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
        plt.scatter(points[:, 0], points[:, 1], c=colors)
        plt.show()

    # elif(num==3):
     #   plt.scatter(a, b, color='blue')
      #  plt.scatter(c, d, color='red')
       # plt.scatter(e, f, color='green')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # Load lib
    lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)

    # Cas_Test
    ##Classification
    #Linear_Simple_test(lib)
    # Linear_Multiple_test(lib)
    # XOR_test(lib)
    # Cross_test(lib)
    # Linear_Multiple_3_test(lib)
    # Multi_Cross_test(lib)

    ##Regression
    #Linear_Simple_2D(lib)
    #Non_Linear_Simple_2D(lib)
    #Linear_Simple_3D(lib)
    #Linear_Tricky_3D(lib)
    Non_Linear_Simple_3D(lib)
