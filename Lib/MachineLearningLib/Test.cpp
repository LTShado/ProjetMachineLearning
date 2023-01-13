#include "pch.h"
#include "Test.h"

extern "C" double *create_array(int arr_size)
{
    double *tab = new double[arr_size];
    for (int i = 0; i < arr_size; ++i)
        tab[i] = i;
    return tab;
}

extern "C" void destroy_array(double *arr, int arr_size)
{
    delete[] (arr);
}

/*
 * retourne un array de 3 nombre random entre [-1, 1]
 */
extern "C" double *initModelLinear()
{
    double *W = create_array(3);

    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (int i = 0; i < 3; i++)
        W[i] = distribution(generator);

    return W;
}

/*
 *
 */
extern "C" double *trainning()
{
    return nullptr;
}
