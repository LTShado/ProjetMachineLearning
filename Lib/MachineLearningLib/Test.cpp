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

float *create_array(int arr_size)
{
    auto tab = new float[arr_size];
    for (auto i = 0; i < arr_size; ++i)
        tab[i] = i;
    return tab;
}

// Linear Model

extern "C" float *create_model_linear(int size)
{
    float *model = new float[size];

    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (int i = 0; i < size + 1; i++)
        model[i] = distribution(generator);

    return model;
}

extern "C" float *train_rosenblatt_linear(float *W, int W_size, float *X, float *Y, int count, float step, int X_flatten_size)
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
extern "C" void createPMC(int *npl, int *d, float **X, float **deltas, float ***W)
{
    int sizeNpl = sizeof(npl) / sizeof(npl[0]);
    int L = sizeNpl - 1;

    // CPY npl in d
    for (int i = 0; i < sizeNpl; i++)
        d[i] = npl[i];

    // Init W:
    for (int l = 0; l < sizeNpl; l++)
    {
        W[l] = new float *;
        if (l == 0)
            continue;
        for (int i = 0; i < d[l - 1] + 1; i++)
        {
            W[l][i] = new float;
            for (int j = 0; j < d[l] + 1; j++)
            {
                uniform_real_distribution<double> distribution(-1, 1);
                unsigned seed = time(nullptr);
                default_random_engine generator(seed);
                float r = distribution(generator);
                W[l][i][j] = j == 0 ? 1 : r;
            }
        }
    }

    // Init X and deltas:
    for (int l = 0; l < sizeNpl; l++)
    {
        X[l] = new float;
        deltas[l] = new float;
        for (int j = 0; j < d[l] + 1; j++)
        {
            deltas[l][j] = 0.f;
            X[l][j] = j == 0 ? 1 : 0;
        }
    }
}

extern "C" void propagate(float *inputs, bool isClassification, int L, int *d, int **X, float ***W)
{
    int dSize = sizeof(d) / sizeof(d[0]);

    for (int j = 1; j < d[0] + 1; j++)
        X[0][j] = inputs[j - 1];

    for (int l = 1; l < dSize; l++)
    {
        for (int j = 1; j < d[l] + 1; j++)
        {
            int total = 0;
            for (int i = 0; i < d[l - 1] + 1; i++)
                total += W[l][i][j] * X[l - 1][i];

            X[l][j] = total;
            if (isClassification || l < L)
                X[l][j] = tanh(total);
        }
    }
}

extern "C" float *predict(float *inputs, bool isClassification, int L, int *d, int **X, float ***W)
{
    float *new_arr = new float;
    int size = sizeof(X[L]) / sizeof(X[L][0]);
    propagate(inputs, isClassification, L, d, X, W);
    memcpy(new_arr, &X[L][1], (size - 1) * sizeof(float));
    return new_arr;
}

extern "C" void train(float **xTrain, float **yTrain, bool isClassification, float alpha, int nbIter, int L, int *d, int **X, int **deltas, float ***W)
{
    int dSize = sizeof(d) / sizeof(d[0]);
    int xTrainSize = sizeof(xTrain) / sizeof(xTrain[0]);
    uniform_int_distribution<int> distribution(0, xTrainSize - 1);
    mt19937 generator;

    for (int it = 0; it < nbIter; ++it)
    {
        int k = distribution(generator);
        float *Xk = xTrain[k];
        float *Yk = yTrain[k];

        propagate(Xk, isClassification, L, d, X, W);
        for (int j = 1; j <= d[L]; ++j)
        {
            deltas[L][j] = X[L][j] - Yk[j - 1];
            if (isClassification)
                deltas[L][j] = deltas[L][j] * (1 - X[L][j] * X[L][j]);
        }

        for (int l = L - 1; l >= 1; --l)
        {
            for (int i = 1; i <= d[l - 1]; ++i)
            {
                double total = 0.0;
                for (int j = 1; j <= d[l]; ++j)
                    total += W[l][i][j] * deltas[l][j];
                deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * total;
            }
        }

        for (int l = 1; l < dSize; ++l)
        {
            for (int i = 0; i <= d[l - 1]; ++i)
            {
                for (int j = 1; j <= d[l]; ++j)
                    W[l][i][j] += -alpha * X[l - 1][i] * deltas[l][j];
            }
        }
    }
}
