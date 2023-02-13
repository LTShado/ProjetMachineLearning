#include "pch.h"
#include "Test.h"
#include <stdlib.h>
#include <list>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

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

extern "C" void saveModelLinear(float* W, int W_size, char* Name) {

    ofstream myfile(Name);
    if (myfile.is_open())
    {
        for (int i = 0; i < W_size; i++) {
            myfile << W[i] << "\n";
        }
        myfile.close();
    }
    else cout << "Unable to open file";

    cout << "Save finish" << endl;
}

extern "C" void readModelLinear(char* Name) {

    string line;
    ifstream myfile(Name);
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            cout << line << '\n';
        }
        myfile.close();
    }

    else cout << "Unable to open file";

    cout << "Read finish" << endl;
}

extern "C" float* loadModelLinear(char* Name) {

    string line;
    ifstream myfile(Name);

    int filesize = 0;
    string value[10]= {""};

    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            //cout << stof(line) << '\n';
            value[filesize] = line;
            filesize++;
        }
        myfile.close();
    }
    else cout << "Unable to open file" << endl;
    
    float* W = new float[filesize];
    int c = 0;


    for (int i = 0; i < filesize; i++) {
        W[i] = stof(value[i]);
        cout << "W " << W[i] << endl;
    }

    cout << "Load finish" << endl;

    return W;
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

extern "C" float* train_regression_linear(float* W, int W_size, float* X, float* Y,int Y_size, int count, float step, int X_flatten_size, int dim) {
    if (dim == 2) {
        int col;

        col = X_flatten_size / (W_size - 1);
        int row = (W_size - 1);

        float** x = new float* [col];

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
        float** xt = TransposeMat(x, col, row);
        //float** xtx = DotMatDouble(x, col, row, xt, row, col);
        float** xtx = DotMatDouble(xt, row, col, x, col, row);
        float** xtx_cache = xtx;

        float** xtx_inv = InverseMat(xtx, row, row);

        if (isnan(xtx_inv[0][0])) {
            while (isnan(xtx_inv[0][0])) {
                cout << "pas inversible" << endl;
                float** ridge = new float* [col];
                for (int i = 0; i < col; i++) {
                    ridge[i] = new float[row];
                }

                for (int i = 0; i < col; i++) {
                    for (int j = 0; j < col; j++) {
                        if (i == j) {
                            ridge[i][j] = 0.1;
                        }
                        else {
                            ridge[i][j] = 0.0;
                        }
                    }
                }
                cout << "ridge " << endl;
                for (int i = 0; i < col; i++) {
                    for (int j = 0; j < col; j++) {
                        cout << " " << ridge[i][j];
                    }
                    cout << endl;
                }
                xtx_cache = AddMat(xtx_cache, ridge, row);
                xtx_inv = InverseMat(xtx_cache, row, row);

            }
        }
        cout << "inversible" << endl;

        cout << "xtx_inv " << endl;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < row; j++) {
                cout << " " << xtx_inv[i][j];
            }
            cout << endl;
        }

        W = DotMatSimple(DotMatDouble(xtx_inv, row, row, xt, row, col), row, col, Y, Y_size);

        //float** test = DotMat(xtx_inv, col, col, xt, row, col);

        return W;
    }
    if (dim == 1) {
        int col;

        col = X_flatten_size;

        float** x = new float* [col];
        for (int i = 0; i < col; i++) {
            x[i] = new float[1];
        }
        //cout << "col " << col << " row " << row << endl;
        int x_count = 0;
        for (int i = 0; i < col; i++) {
            for (int j = 0; j < 1; j++) {
                x[i][j] = X[x_count];
                x_count++;
            }
        }

        float** xt = TransposeMat(x, col, 1);
        float** xtbis = xt;
        float** xtx = DotMatDouble(xt, 1, col, x, col, 1);

        float invxtx = 0;

        invxtx = 1/xtx[0][0];
        
        cout << "invxtx " << invxtx << endl;

        for (int i = 0; i < col; i++) {
            xtbis[0][i] *= invxtx;
            cout << "xtbis " << xtbis[0][i] << endl;
        }

        float* w = new float[1];
        w[0] = 0;
        for (int i = 0; i < Y_size; i++) {
            w[0] += xtbis[0][i] * Y[i];
        }
        
        cout << w[0] << endl;
        W = w;
        return W;
    }
}

float** InverseMat(float** mat, int col, int row) {
    /*float** inv = new float* [col];
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

    cout << "mat " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << mat[i][j];
        }
        cout << endl;
    }

    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (i == j)
            {
                mat[i][j + col] = 1;
            }
            else
            {
                mat[i][j + col] = 0;
            }
        }
    }

    cout << "mat " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << mat[i][j];
        }
        cout << endl;
    }*/

    int order = col;
    cout << "col " << col << endl;
    float temp;
    float** inv = new float* [20];
    for (int i = 0; i < 20; i++)
        inv[i] = new float[20];

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < 2 * order; j++) {
            inv[i][j] = 0;
        }
    }
    cout << "mat " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << mat[i][j];
        }
        cout << endl;
    }


    /*inv[0][0] = 6; inv[0][1] = 9; inv[0][2] = 5;
    inv[1][0] = 8; inv[1][1] = 3; inv[1][2] = 2;
    inv[2][0] = 1; inv[2][1] = 4; inv[2][2] = 7;*/

    cout << "mat " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << mat[i][j];
            inv[i][j] = mat[i][j];
        }
        cout << endl;
    }

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < 2 * order; j++) {
            if (j == (i + order))
                inv[i][j] = 1;
        }
    }
    cout << "matinvbeforetest " << endl;
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order * 2; j++) {
            cout << " " << inv[i][j];
        }
        cout << endl;
    }
    for (int i = order - 1; i > 0; i--) {
        if (inv[i - 1][0] < inv[i][0]) {
            float* temp = inv[i];
            inv[i] = inv[i - 1];
            inv[i - 1] = temp;
        }
    }
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            if (j != i) {
                temp = inv[j][i] / inv[i][i];
                for (int k = 0; k < 2 * order; k++) {
                    inv[j][k] -= inv[i][k] * temp;
                }
            }
        }
    }
    for (int i = 0; i < order; i++) {
        temp = inv[i][i];
        for (int j = 0; j < 2 * order; j++) {
            inv[i][j] = inv[i][j] / temp;
        }
    }

    cout << "matinvaftertest " << endl;
    for (int i = 0; i < order; i++) {
        for (int j = order; j < order * 2; j++) {
            printf("%.3f\t", inv[i][j]);
        }
        cout << endl;
    }

    float** invfinal = new float* [col];
    for (int i = 0; i < col; i++)
        invfinal[i] = new float[col];

    for (int i = 0; i < order; i++) {
        for (int j = order; j < order * 2; j++) {
            invfinal[i][j-order] = inv[i][j];
        }
    }

    return invfinal;
}

float** DotMatDouble(float** mat1, int col1, int row1, float** mat2, int col2, int row2) {


    float** mat_dot = new float* [col1];
    for (int i = 0; i < col1; i++) {
        mat_dot[i] = new float[row2];
    }
    cout << "col2 " << col2 << " row2 " << row2<<endl;
    cout << "col1 " << col1 << " row1 " << row1<<endl;

    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row2; j++) {
            mat_dot[i][j] = 0;
        }
    }

    cout << "mat dot before" << endl;
    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row2; j++) {
            cout << " " << mat_dot[i][j];
        }
        cout << endl;
    }

    /*cout << "mat1" << endl;
    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row1; j++) {
            cout << " " << mat1[i][j];
        }
        cout << endl;
    }
    cout << "mat2" << endl;
    for (int i = 0; i < col2; i++) {
        for (int j = 0; j < row2; j++) {
            cout << " " << mat2[i][j];
        }
        cout << endl;
    }*/

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
    cout << "mat dot after" << endl;
    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row2; j++) {
            cout << " " << mat_dot[i][j];
        }
        cout << endl;
    }

    return mat_dot;
}

float* DotMatSimple(float** mat1, int col1, int row1, float* mat2, int size) {


    float* mat_dot = new float [col1];

    for (int i = 0; i < col1; i++) {
        mat_dot[i] = mat2[i];
    }
    /*cout << "Mat1 " << endl;
    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row1; j++) {
            cout << " " << mat1[i][j];
        }
        cout << endl;
    }
    cout << "Mat2 " << endl;
    for (int i = 0; i < size; i++) {
        cout << " " << mat2[i] << endl;
    }*/

    for (int i = 0; i < col1; i++) {
        mat_dot[i] = (mat_dot[i] * mat1[i][0]) + (mat_dot[i] * mat1[i][1]) + (mat_dot[i] * mat1[i][2]);
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

    cout << "mat dot simple" << endl;
    for (int i = 0; i < col1; i++) {
        cout << " " << mat_dot[i];
        cout << endl;
    }

    return mat_dot;
}

float** TransposeMat(float** mat, int col, int row) {

    cout << "mat before transpose " << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            cout << " " << mat[i][j];
        }
        cout << endl;
    }
    float** mat_t = new float* [row];
    for (int i = 0; i < row; i++) {
        mat_t[i] = new float[col];
    }

    int test = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            mat_t[i][j] = mat[j][i];
        }
    }
    cout << "mat transpose " << endl;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << " " << mat_t[i][j];
        }
        cout << endl;
    }
    return mat_t;
}

float** AddMat(float** mat1, float** mat2, int size) {
    float** mat_add = new float* [size];
    for (int i = 0; i < size; i++) {
        mat_add[i] = new float[size];
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat_add[i][j] = 0;
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat_add[i][j] = mat_add[i][j] + mat1[i][j] + mat2[i][j];
            //cout << "mat1 dot " << mat1[i][k] << " mat2 " << mat2[k][j] << endl;
        }
    }

    return mat_add;
}

extern "C" float predict_regression(float* W,int W_size, float* value, int dim) {
    float sum = 0;
    if (dim == 1) {
        for (int i = 0; i < 1; i++) {
            //cout << "W " << W[i] << endl;
            //cout << "Y " << value[i] << endl;

            sum = sum + (W[i] * value[i]);
            //cout << "sum " << sum << endl;
        }
    }
    else {
        for (int i = 0; i < W_size; i++) {
            //cout << "W " << W[i] << endl;
            //cout << "Y " << value[i] << endl;

            sum = sum + (W[i] * value[i]);
            //cout << "sum " << sum << endl;
        }
    }
    

    return sum;
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

///////////////////// PMC methods /////////////////////


/* Init datas from pointers
// model is an array of pointer
// model[0] = int* sizeNpl: number of layer
// model[1] = int* d[]: node per layer, array of size sizeNpl
// model[2] = int* maxN: max number of node on a layer
// model[3] = float* X[]: array of size sizeNpl*maxN
// model[4] = float* deltas[]: array of size sizeNpl*maxN
// model[5] = float* W[]: array of size sizeNpl*maxN*maxN
*/ 
extern "C" void** createModelPMC(int* _npl, int _sizeNpl)
{
    void **model = new void *[6];

    int *sizeNpl = new int(_sizeNpl);
    int *d = nullptr;
    int *maxN = new int();
    float *X = nullptr;
    float *deltas = nullptr;
    float *W = nullptr;

    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1, 1);

    //init d
    d = new int[*sizeNpl];
    for (int i = 0; i < *sizeNpl; ++i)
      d[i] = _npl[i];

    // init maxN
    int max = 0;
    for (int i = 0; i < *sizeNpl; ++i)
      if (d[i] > max)
        max = d[i];
    *maxN = max + 1;

    // init X and deltas
    X = new float[(*sizeNpl) * (*maxN)];
    deltas = new float[(*sizeNpl) * (*maxN)];
    for (int i = 0; i < (*sizeNpl) * (*maxN); ++i) {
      X[i] = 0.f;
      deltas[i] = 0.f;
    }
      
    for (int l = 0; l < *sizeNpl; ++l) {
        for (int j = 0; j < d[l] + 1; ++j) {
        X[l * (*maxN) + j] = j == 0 ? 1.0f : 0.f;
        }
    }
    
   
    // init W
    W = new float[(*sizeNpl) * (*maxN) * (*maxN)];
    for (int i = 0; i < (*sizeNpl) * (*maxN) * (*maxN); ++i)
      W[i] = 0.f;
    
    for (int l = 1; l < *sizeNpl; ++l)
    {
        for (int i = 0; i < d[l - 1] + 1; ++i)
        {
            for (int j = 0; j < d[l] + 1; ++j)
            {
                W[l*(*maxN)*(*maxN) + i*(*maxN) + j] = j == 0 ? 0.f : distribution(generator);
            }
        }
    }

    model[0] = (void *)sizeNpl;
    model[1] = (void *)d;
    model[2] = (void *)maxN;
    model[3] = (void *)X;
    model[4] = (void *)deltas;
    model[5] = (void *)W;
    return model;

}

void propagatePMC(float *inputs, bool isClassification, int *d, int* p_sizeNpl, int* p_maxN, float *X, float *W)
{
  int sizeNpl = *p_sizeNpl;
  int maxN = *p_maxN;
  int L = sizeNpl - 1;
    for (int j = 1; j < d[0] + 1; ++j)
        X[0*maxN + j] = inputs[j - 1];

    for (int l = 1; l < sizeNpl; ++l)
    {
        for (int j = 1; j < d[l] + 1; ++j)
        {
            float total = 0.f;
            for (int i = 0; i < d[l - 1] + 1; ++i)
                total += W[l*maxN*maxN +i*maxN +j] * X[(l-1)*maxN + i];

            X[l*maxN + j] = total;
            if (isClassification || l < L)
                X[l*maxN + j] = tanh(total);
        }
    }
}

extern "C" float* predictPMC(float *inputs, bool isClassification, void** model)
{   

    int* sizeNpl = (int*)model[0];
    int *maxN = (int *)model[2];
    int *d = (int *)model[1];
    float *X = (float *)model[3];
    int L = (*sizeNpl)-1;
    float* new_arr = new float[d[L]];

    propagatePMC(inputs, isClassification, d, sizeNpl, maxN, X,(float*)model[5]);
    memcpy(new_arr, &X[L*(*maxN) + 1], d[L] * sizeof(float));
    return new_arr;
}


extern "C" void trainPMC(int sizeT, float *xTrain, int sizeDataXTrain, float *yTrain, int sizeDataYTrain, bool isClassification, float alpha, int nbIter, void** model)
{

    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, sizeT);
    int *sizeNpl = (int *)model[0];
    int *maxN = (int *)model[2];
    int L = (*sizeNpl) - 1;
    int *d = (int *)model[1];
    float *X = (float *)model[3];
    float *deltas = (float *)model[4];
    float *W = (float *)model[5];
    

    for (int it = 0; it < nbIter; ++it)
    {
        int k = distribution(generator);
        float* Xk = &xTrain[k*sizeDataXTrain];
        float* Yk = &yTrain[k*sizeDataYTrain];

        propagatePMC(Xk, isClassification, d, sizeNpl, maxN, X, W);
        for (int j = 1; j < d[L]+1; ++j)
        {
          deltas[L * (*maxN) + j] = X[L * (*maxN) + j] - Yk[j - 1];
            if (isClassification)
            deltas[L * (*maxN) + j] = 
                deltas[L * (*maxN) + j] * (1 - (X[L * (*maxN) + j] * X[L * (*maxN) + j]));
        }

        for (int l = *(sizeNpl)-1 ; l >= 2; --l)
        {
            for (int i = 1; i < d[l - 1]+1; ++i)
            {
                float total = 0.f;
                for (int j = 1; j < d[l]+1; ++j)
                  total += W[l * (*maxN) * (*maxN) + i * (*maxN) + j] * deltas[l * (*maxN) + j];
                deltas[(l - 1) * (*maxN) + i] = 
                    (1 - (X[(l - 1) * (*maxN) + i] * X[(l - 1) * (*maxN) + i])) * total;
            }
        }

        for (int l = 1; l < *sizeNpl; ++l)
        {
            for (int i = 0; i < d[l - 1] + 1; ++i)
            {
                for (int j = 1; j < d[l] + 1; ++j)
                W[l * (*maxN) * (*maxN) + i * (*maxN) + j] +=
                      -alpha * X[(l - 1) * (*maxN) + i] * deltas[l * (*maxN) + j];
            }
        }
    }
}

extern "C" MACHINELEARNINGLIB_API void **loadModelPMC(char *filename) {
  void** model = new void*[6];
  int * sizeNpl = new int();
  int *d = nullptr;
  int *maxN = new int();
  float *X = nullptr;
  float* deltas = nullptr;
  float *W = nullptr;

  string line, seq, val;
  ifstream myfile(filename);

  if (myfile.is_open()) {
    while (getline(myfile, line)) {
      // load sizeNpl
      if (line == "#sizeNpl") {
        //cout <<  endl << "found sizeNql" << endl;
        getline(myfile, val);
        *sizeNpl = stoi(val);
      }

      // load d
      if (line == "#d") {
        //cout << endl << "found d" << endl;
        d = new int[*sizeNpl];
        getline(myfile, line);
        stringstream ss(line);
        for (int i = 0; i < *sizeNpl; ++i) {
          getline(ss, val, ',');
          d[i] = stoi(val);
        }
      }

      // load maxN
      if (line == "#maxN") {
        //cout << endl << "found maxN" << endl;
        getline(myfile, val);
        *maxN = stoi(val);
      }
      
      // load X
      if (line == "#X") {
        //cout << endl << "found X" << endl;
        X = new float[(*sizeNpl) * (*maxN)];
        for (int i = 0; i < (*sizeNpl) * (*maxN); ++i)
          X[i] = 0.f;
        
        
        for (int l = 0; l < (*sizeNpl); ++l) {
          getline(myfile, line);
          stringstream ss(line);
          for (int j = 0; j < d[l] + 1; ++j) {
            getline(ss, val, ',');
            X[l * (*maxN) + j] = stof(val);
          }
        }
      }
      
      // load deltas
      if (line == "#deltas") {
        //cout << endl << "found deltas" << endl;
        deltas = new float[(*sizeNpl) * (*maxN)];
        for (int i = 0; i < (*sizeNpl) * (*maxN); ++i)
          deltas[i] = 0.f;

        for (int l = 0; l < (*sizeNpl); ++l) {
          getline(myfile, line);
          stringstream ss(line);
          for (int j = 0; j < d[l] + 1; ++j) {
            getline(ss, val, ',');
            deltas[l * (*maxN) + j] = stof(val);
          }
        }
      }
      
      // load W
      if (line == "#W") {
        //cout << endl << "found W" << endl;
        W = new float[(*sizeNpl) * (*maxN) * (*maxN)];
        for (int i = 0; i < (*sizeNpl) * (*maxN) * (*maxN); ++i)
          W[i] = 0.f;

        for (int l = 0; l < (*sizeNpl); ++l) {
          if (l == 0)
            continue;
          getline(myfile, line);
          stringstream ss(line);
          for (int i = 0; i < d[l - 1] + 1; ++i) {
            getline(ss, seq, ';');
            stringstream ss2(seq);
            for (int j = 0; j < d[l] + 1; ++j) {
              getline(ss2, val, ',');
              W[l * (*maxN) * (*maxN) + i * (*maxN) + j] = stof(val);
            }
          }
        }
      }
      
    }
    myfile.close();
    model[0] = (void *)sizeNpl;
    model[1] = (void *)d;
    model[2] = (void *)maxN;
    model[3] = (void *)X;
    model[4] = (void *)deltas;
    model[5] = (void *)W;
    cout << "Load finish" << endl;

  } else
    cout << "Unable to open file, failed to load model" << endl;

  return model;
}

extern "C" MACHINELEARNINGLIB_API void saveModelPMC(void** model, char *filename) {
    int *sizeNpl = (int *)model[0];
    int* d = (int*) model[1];
    int* maxN = (int*) model[2];
    float* X = (float*) model[3];
    float* deltas = (float*) model[4];
    float* W = (float*) model[5];

  ofstream myfile(filename, ofstream::out | ofstream::trunc);
  if (myfile.is_open()) {

    // save sizeNpl
    myfile << "#sizeNpl"
           << "\n";
    myfile << *sizeNpl << "\n";

    // save d
    myfile << "#d" << "\n";
    for (int i = 0; i < *sizeNpl; ++i) {
      myfile << d[i];
      if (i != *sizeNpl - 1)
        myfile << ",";
    }
    myfile << "\n";

    // save maxN
    myfile << "#maxN" << "\n";
    myfile << *maxN << "\n";

    
    // save X
    myfile << "#X" << "\n";
    for (int l = 0; l < (*sizeNpl); ++l) {
      for (int j = 0; j < d[l] + 1; ++j) {
        myfile << X[l * (*maxN) + j];
        if (j != d[l]) myfile << ",";
      }
      myfile << "\n";
    }
    
    // save deltas
    myfile << "#deltas" << "\n";
    for (int l = 0; l < (*sizeNpl); ++l) {
      for (int j = 0; j < d[l] + 1; ++j) {
        myfile << deltas[l * (*maxN) + j];
        if (j != d[l])
          myfile << ",";
      }
      myfile << "\n";
    }
    
    // save W
    myfile << "#W" << "\n";
    for (int l = 0; l < (*sizeNpl); ++l) {
      if (l == 0)
        continue;
      for (int i = 0; i < d[l - 1] + 1; ++i) {
        for (int j = 0; j < d[l] + 1; ++j) {
          myfile << W[l * (*maxN) * (*maxN) + i * (*maxN) + j];
          if (j != d[l])myfile << ",";
        }
        if (i != d[l-1])
          myfile << ";";
      }
      myfile << "\n";
    }
    
    myfile.close();
    cout << "Model saved" << endl;

  } else
    cout << "Unable to create/open file, failed to save model";

}
