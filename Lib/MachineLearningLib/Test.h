#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <random>
#include <numeric>

using namespace std;

#ifdef MACHINELEARNINGLIB_EXPORTS
#define MACHINELEARNINGLIB_API __declspec(dllexport)
#else
#define MACHINELEARNINGLIB_API __declspec(dllimport)
#endif

// Array Method

float *create_array(int arr_size);

void destroy_array(float *arr, int arr_size);

// Linear Model

extern "C" MACHINELEARNINGLIB_API float* create_model_linear(int size);

extern "C" MACHINELEARNINGLIB_API float* train_rosenblatt_linear(float *W, int W_size, float *X, float *Y, int count, float step, int size);

extern "C" MACHINELEARNINGLIB_API float* train_regression_linear(float* W, int W_size, float* X, float* Y, int Y_size, int count, float step, int X_flatten_size, int dim);

float** TransposeMat(float** mat, int col, int row);
float** DotMatDouble(float** mat1, int col1, int row1, float** mat2, int col2, int row2);
float* DotMatSimple(float** mat1, int col1, int row1, float* mat2, int size);
float** InverseMat(float** mat, int col, int row);
float** AddMat(float** mat1, float** mat2, int size);

extern "C" MACHINELEARNINGLIB_API int ReadArrayValue(float *arr);

extern "C" MACHINELEARNINGLIB_API int test();

// PMC Method

extern "C" MACHINELEARNINGLIB_API void createModelPMC(int* npl, int sizeNpl, int maxN, float* X, float* deltas, float* W);

void propagatePMC(float* inputs, bool isClassification, int* d, int sizeNpl, int L, int maxN, float* X, float* W);

extern "C" MACHINELEARNINGLIB_API float* predictPMC(float* inputs, bool isClassification, int* d, int sizeNpl, int maxN, float* X, float* W);

extern "C" MACHINELEARNINGLIB_API void trainPMC(int sizeT, float* xTrain, int sizeDataXTrain, float* yTrain, int sizeDataYTrain, bool isClassification, float alpha, int nbIter, int* d, int sizeNpl, int maxN, float* X, float* deltas, float* W);