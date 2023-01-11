#pragma once

#ifdef MACHINELEARNINGLIB_EXPORTS
#define MACHINELEARNINGLIB_API __declspec(dllexport)
#else
#define MACHINELEARNINGLIB_API __declspec(dllimport)
#endif


extern "C" MACHINELEARNINGLIB_API int division(int x, int y);
