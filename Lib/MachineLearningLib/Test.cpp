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

extern "C" float* train_rosenblatt_linear(float* W, float* X, float* Y, int count, float step, int size)
{
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(0, size/(sizeof(W)-1));
    
    float* W_modif = new float[(sizeof(W) - 1)];


    for (int iter = 0; iter < count; iter++) {
        int k = distribution(generator);
        float yk = Y[k];

        float Xk[sizeof(W)] {0};
        Xk[0] = 1;
        for (int i = 0; i < (sizeof(W) - 1); i++) {
            Xk[i+1] = X[k * 2 + i];
        }
        float sum = 0;
        for (int j = 0; j < (sizeof(W) - 1); j++) {
            sum += (W[j] * Xk[j]);
        }

        float gXk = 0;
        if (sum>=0) {
            gXk = 1.0;
        }
        else {
            gXk = -1.0;
        }

        for (int l = 0; l <(sizeof(W) - 1); l++) {
            W[l] = step * (yk - gXk) * Xk[l] + W[l];
        }
        
    }
    
    return W;
}

extern "C" float ReadArrayValue(float* arr) {
    return sizeof(arr);
}

extern "C" int test(int a, int b, int c, int d, int e, int f) {
    return 32;
}


