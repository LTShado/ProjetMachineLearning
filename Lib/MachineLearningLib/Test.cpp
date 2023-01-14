#include "pch.h"
#include "Test.h"
#include <stdlib.h>
#include<list>
#include <random>
#include<iostream>
using namespace std;

extern "C" void destroy_array(double *arr, int arr_size)
{
    delete[] (arr);
}

extern "C" float* create_model_linear(int size) {

    float* model = new float[size];

    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (int i = 0; i < size+1; i++)
        model[i] = distribution(generator);

    return model;
}

extern "C" float* train_rosenblatt_linear(float* W, int W_size, float* X, float* Y, int count, float step, int X_flatten_size)
{    
    float* W_modif = new float[(W_size - 1)];
    cout << "x_SIZE :" << X_flatten_size << endl;
    cout << "W_SIZE :" << W_size << endl;

    for (int iter = 0; iter < count; iter++) {
        int k = rand() % (X_flatten_size / (W_size - 1));
        float yk = Y[k];
        //cout << "k :" << k << endl; //pas de pb ici

        float* Xk = new float[X_flatten_size / (W_size - 1)];
        Xk[0] = 1;
        for (int i = 0; i < (W_size - 1); i++) {
            Xk[i+1] = X[k * 2 + i];
        }
        /*for (int X_tab = 0; X_tab < X_flatten_size / (W_size - 1); X_tab++) {
            cout << Xk[X_tab] << endl;
        }*///pas de pb ici

        float sum = 0;
        for (int j = 0; j < (W_size); j++) {
            sum += (W[j] * Xk[j]);
        }
        //cout << "sum :" << sum << endl;
        float gXk;
        if (sum>=0) {
            gXk = (float)1;
        }
        else {
            gXk = (float)-1;
        }
        //cout << "gXk :" << gXk << endl;
        for (int l = 0; l <(W_size); l++) {
            W[l] = step * (yk - gXk) * Xk[l] + W[l];
        }
        
    }
    for (int i = 0; i < W_size; i++) {
        cout <<"W :"<< W[i] << endl;
    }
    return W;
}

extern "C" int ReadArrayValue(float* arr) {
    return sizeof(arr);
}

extern "C" int test() {
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(0, 10);

    int t = distribution(generator);

    return t;
}


