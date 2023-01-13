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

extern "C" MACHINELEARNINGLIB_API double *create_array(int arr_size);

extern "C" MACHINELEARNINGLIB_API void destroy_array(double *arr, int arr_size);

extern "C" MACHINELEARNINGLIB_API double *initModelLinear();

extern "C" MACHINELEARNINGLIB_API double *trainning();
