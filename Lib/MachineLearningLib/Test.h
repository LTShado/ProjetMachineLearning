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

extern "C" MACHINELEARNINGLIB_API int ReadArrayValue(float *arr);

extern "C" MACHINELEARNINGLIB_API int test();

// PMC Method

extern "C" MACHINELEARNINGLIB_API void create_model_pmc(int* npl, int sizeNpl, int maxN, float* X, float* deltas, float* W);

void propagate(float* inputs, bool isClassification, int* d, int sizeNpl, int L, int maxN, float* X, float* W);

extern "C" MACHINELEARNINGLIB_API float* predict(float* inputs, bool isClassification, int* d, int sizeNpl, int maxN, float* X, float* W);

extern "C" MACHINELEARNINGLIB_API void train(float* xTrain, int sizeXTrain, float* yTrain, int sizeYTrain, bool isClassification, float alpha, int nbIter, int* d, int sizeNpl, int maxN, float* X, float* deltas, float* W);