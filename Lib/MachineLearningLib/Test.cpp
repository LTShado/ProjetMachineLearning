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

extern "C" float* train_regression_linear(float* W, int W_size, float* X, float* Y, int count, float step, int X_flatten_size) {
    

    int col = X_flatten_size / (W_size - 1);
    int row = (W_size - 1);

    float** x = new float*[col];

    for (int i = 0; i < col; i++) {
        x[i] = new float[row];
    }
    //cout << "col " << col << " row " << row << endl;
    int x_count = 0;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            x[i][j] = X[x_count];
            x_count++;
        }
    }
    /*for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << "i "<< i << " j " << j << " value " << x[i][j] << endl;
        }
    }*/
    float** xt = TransposeMat(x,col,row);
    float** xtx = DotMat(x,col,row ,xt, row, col);

    float** xtx_inv = InverseMat(xtx,col,col);
    //W= (Inverse(transpose(X)*X)*transpose(X))*Y
    
    return W;
}

float** InverseMat(float** mat, int col, int row) {

    float** inv = new float* [col];
    for (int i = 0; i < col; i++) {
        inv[i] = new float[row];
    }

    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            if (i == j) {
                inv[i][j] = 1;
            }
            else {
                inv[i][j] = 0;
            }
        }
    }
    cout << "inv " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << inv[i][j];
        }
        cout << endl;
    }

    int l = 0;
    float lval = 0;
    for (int k = 0; k < col; k++) {
        lval = mat[k][k];
        for (l = 0; l < col; l++) {
            mat[k][l] /= lval;
            inv[k][l] /= lval;
        }
        for (int m = 0; m < col; m++) {
            lval = mat[m][k];
            for (int n = 0; n < col; n++) {
                if (m == k) {
                    break;
                }
                mat[m][n] -= mat[k][l] * lval;
                inv[m][n] -= inv[k][l] * lval;
            }
        }
    }
    cout << " inv " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << inv[i][j];
        }
        cout << endl;
    }
    cout << " mat " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << mat[i][j];
        }
        cout << endl;
    }



    float d = 0;

    /*for (int i = 0; i < 3; i++) {
        d = d + (mat[0][i] * (mat[1][(i + 1) % 3] * mat[2][(i + 2) % 3] - mat[1][(i + 2) % 3] * mat[2][(i + 1) % 3]));
    }
    cout<<"d " << d;
    if (d > 0)
    {
        cout << "\nInverse of the matrix is: \n";
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                std::cout << ((mat[(j + 1) % 3][(i + 1) % 3] * mat[(j + 2) % 3][(i + 2) % 3]) - (mat[(j + 1) % 3][(i + 2) % 3] * mat[(j + 2) % 3][(i + 1) % 3])) / d << "\t";
            std::cout << "\n";
        }
    }
    else std::cout << "Inverse does'nt exist for this matrix";*/

    return mat;
}

float** DotMat(float** mat1, int col1, int row1, float** mat2, int col2, int row2) {


    cout << "row1 " << row1 << " row2 " << row2 << endl;
    cout << "col1 " << col1 << " col2 " << col2 << endl;
    float** mat_dot = new float* [col1];
    for (int i = 0; i < col1; i++) {
        mat_dot[i] = new float[row2];
    }

    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row2; j++) {
            mat_dot[i][j] = 0;
        }
    }

    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row2; j++) {
            for (int k = 0; k < row1; k++)
            {
                mat_dot[i][j] = mat_dot[i][j] + mat1[i][k] * mat2[k][j];
                //cout << "mat1 dot " << mat1[i][k] << " mat2 " << mat2[k][j] << endl;
            }
        }
    }

    /*for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row1; j++) {
            cout << "Mat1 " << mat1[i][j] << endl;
        }
    }

    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            cout << "Mat2 " << mat2[i][j] << endl;
        }
    }*/

    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row2; j++) {
            cout << "i " << i << " j " << j << " value " << mat_dot[i][j] << endl;
        }
    }

    return mat_dot;
}

float** TransposeMat(float** mat, int col, int row) {

    float** mat_t = new float*[row];
    for (int i = 0; i < row; i++) {
        mat_t[i] = new float[col];
    }

    int test = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            mat_t[i][j] = mat[j][i];
        }
    }
    /*for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << "i " << i << " j " << j << " value " << mat_t[i][j] << endl;
        }
    }*/
    return mat_t;
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
extern "C" void createModelPMC(int* npl, int sizeNpl, int maxN, float* X, float* deltas, float* W)
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

void propagatePMC(float *inputs, bool isClassification, int *d, int sizeNpl, int L, int maxN, float *X, float *W)
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

extern "C" float* predictPMC(float *inputs, bool isClassification, int* d, int sizeNpl, int maxN, float* X, float* W)
{
    int L = sizeNpl-1;
    float* new_arr = new float[d[L]];

    propagatePMC(inputs, isClassification, d, sizeNpl, L, maxN, X, W);
    memcpy(new_arr, &X[L*maxN + 1], d[L] * sizeof(float));
    return new_arr;
}


/*
xTrain, yTrain: array<array of size d[O]>
*/
extern "C" void trainPMC(int sizeT, float *xTrain, int sizeDataXTrain, float *yTrain, int sizeDataYTrain, bool isClassification, float alpha, int nbIter, int* d, int sizeNpl, int maxN, float* X, float* deltas, float* W)
{
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, sizeT);
    
    int L = sizeNpl - 1;
    

    for (int it = 0; it < nbIter; ++it)
    {
        int k = distribution(generator);
        float* Xk = &xTrain[k*sizeDataXTrain];
        float* Yk = &yTrain[k*sizeDataYTrain];

        propagatePMC(Xk, isClassification, d, sizeNpl, L, maxN, X, W);
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
