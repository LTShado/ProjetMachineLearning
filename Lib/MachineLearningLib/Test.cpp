#include "pch.h"
#include "Test.h"
#include <stdlib.h>
#include <list>
#include <random>
#include <iostream>

using namespace std;

// Array method

void destroy_array(float *arr, int arr_size)
{
    delete[] (arr);
}

float* create_array(int arr_size)
{
    auto tab = new float[arr_size];
    for (auto i = 0; i < arr_size; ++i)
        tab[i] = i;
    return tab;
}

// Linear Model

extern "C" float* create_model_linear(int size)
{
    float *model = new float[size];

    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (int i = 0; i < size + 1; i++)
        model[i] = distribution(generator);

    return model;
}

extern "C" float* train_rosenblatt_linear(float *W, int W_size, float *X, float *Y, int count, float step, int X_flatten_size)
{
    // cout << "x_SIZE :" << X_flatten_size << endl;
    // cout << "W_SIZE :" << W_size << endl;

    for (int iter = 0; iter < count; iter++)
    {
        int k = rand() % (X_flatten_size / (W_size - 1));
        float yk = Y[k];
        // cout << "k :" << k << endl; //pas de pb ici

        float *Xk = new float[W_size];
        Xk[0] = 1;
        for (int i = 0; i < (W_size - 1); i++)
        {
            Xk[i + 1] = X[k * 2 + i];
        }
        /*for (int X_tab = 0; X_tab < X_flatten_size / (W_size - 1); X_tab++) {
            cout << Xk[X_tab] << endl;
        }*/
        // pas de pb ici

        float sum = 0;
        for (int j = 0; j < (W_size); j++)
        {
            sum += (W[j] * Xk[j]);
        }
        // cout << "sum :" << sum << endl;
        float gXk;
        if (sum >= 0)
        {
            gXk = (float)1;
        }
        else
        {
            gXk = (float)-1;
        }
        // cout << "gXk :" << gXk << endl;
        for (int l = 0; l < (W_size); l++)
        {
            W[l] = step * (yk - gXk) * Xk[l] + W[l];
        }
    }
    for (int i = 0; i < W_size; i++)
    {
        cout << "W :" << W[i] << endl;
    }
    return W;
}

extern "C" int ReadArrayValue(float *arr)
{
    return sizeof(arr);
}

extern "C" int test()
{
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(0, 10);
    int t = distribution(generator);
    return t;
}

// PMC method

// Init datas from pointers
extern "C" void create_model_pmc(int* npl, int sizeNpl, int maxN, float* X, float* deltas, float* W)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1, 1);
    
    // Alocate Arrays
    //W = new float[sizeNpl * maxN * maxN];
    //X = new float[sizeNpl*maxN];
    //deltas = new float[sizeNpl*maxN];
    
    // Init W:
    
    for (int l = 0; l < sizeNpl; l++)
    {
        if (l == 0)
            continue;
        for (int i = 0; i < npl[l - 1] + 1; i++)
        {
            for (int j = 0; j < npl[l] + 1; j++)
            {
                W[l*maxN*maxN + i*maxN + j] = j == 0 ? 0.f : distribution(generator);
            }
        }
    }

    // Init X and deltas:
    for (int l = 0; l < sizeNpl; l++)
    {
        for (int j = 0; j < npl[l] + 1; j++)
        {
            deltas[l*maxN + j] = 0.f;
            X[l*maxN + j] = j == 0 ? 1.0f : 0.f;
        }
    }

}

void propagate(float *inputs, bool isClassification, int *d, int sizeNpl, int L, int maxN, float *X, float *W)
{
    for (int j = 1; j < d[0] + 1; j++)
        X[0*maxN + j] = inputs[j - 1];

    for (int l = 1; l < sizeNpl; l++)
    {
        for (int j = 1; j < d[l] + 1; j++)
        {
            float total = 0.f;
            for (int i = 0; i < d[l - 1] + 1; i++)
                total += W[l*maxN*maxN +i*maxN +j] * X[(l-1)*maxN + i];

            X[l*maxN + j] = total;
            if (isClassification || l < L)
                X[l*maxN + j] = tanh(total);
        }
    }
}

extern "C" float* predict(float *inputs, bool isClassification, int* d, int sizeNpl, int maxN, float* X, float* W)
{
    int L = sizeNpl-1;
    float* new_arr = new float[d[L]];

    propagate(inputs, isClassification, d, sizeNpl, L, maxN, X, W);
    memcpy(new_arr, &X[L*maxN + 1], d[L] * sizeof(float));
    return new_arr;
}


/*
xTrain, yTrain: array<array of size d[O]>
*/
extern "C" void train(float *xTrain, int sizeXTrain, float *yTrain, int sizeYTrain, bool isClassification, float alpha, int nbIter, int* d, int sizeNpl, int maxN, float* X, float* deltas, float* W)
{
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, sizeXTrain);
    
    int L = sizeNpl - 1;
    

    for (int it = 0; it < nbIter; ++it)
    {
        int k = distribution(generator);
        float* Xk = &xTrain[k];
        float* Yk = &yTrain[k];

        propagate(Xk, isClassification, d, sizeNpl, L, maxN, X, W);
        for (int j = 1; j < d[L]+1; ++j)
        {
            deltas[L*maxN + j] = X[L*maxN + j] - Yk[j - 1];
            if (isClassification)
                deltas[L*maxN + j] = deltas[L*maxN + j] * (1 - (X[L*maxN + j] * X[L*maxN + j]));
        }

        for (int l = sizeNpl-1 ; l >= 2; --l)
        {
            for (int i = 1; i < d[l - 1]+1; ++i)
            {
                float total = 0.f;
                for (int j = 1; j < d[l]+1; ++j)
                    total += W[l*maxN*maxN + i*maxN + j] * deltas[l*maxN + j];
                deltas[(l - 1)*maxN + i] = (1 - (X[(l - 1)*maxN + i] * X[(l - 1)*maxN + i])) * total;
            }
        }

        for (int l = 1; l < sizeNpl; ++l)
        {
            for (int i = 0; i < d[l - 1]+1; ++i)
            {
                for (int j = 1; j <= d[l]; ++j)
                    W[l*maxN*maxN + i*maxN + j] += -alpha * X[(l - 1)*maxN + i] * deltas[l*maxN + j];
            }
        }
    }
}
