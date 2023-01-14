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

extern "C" MACHINELEARNINGLIB_API void destroy_array(double *arr, int arr_size);

extern "C" MACHINELEARNINGLIB_API float* create_model_linear(int size);

extern "C" MACHINELEARNINGLIB_API float* train_rosenblatt_linear(float* W, float* X, float* Y, int count, float step, int size);

extern "C" MACHINELEARNINGLIB_API float ReadArrayValue(float* arr);

extern "C" MACHINELEARNINGLIB_API int test(int a, int b, int c, int d, int e, int f);