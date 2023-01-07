#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <random>
#include <numeric>

using namespace std;

class LinearModel
{
private:
    vector<vector<int>> points{{1, 1}, {2, 1}, {2, 2}};
    vector<int> classes{1, 1, -1};
    vector<float> W;

    void trainning();
    void display();

public:
    LinearModel();
    ~LinearModel();
};
