#include <iostream>
#include <array>
#include <vector>
#include <string>
#include "PMC.hpp"

using namespace std;

int main()
{
    PMC model = PMC({2, 1});
    vector<vector<float>> test_points;
    vector<string> test_colors;
    for (int row = 0; row < 300; ++row)
    {
        for (int col = 0; col < 300; ++col)
        {
            vector<float> p{col / 100.0f, row / 100.0f};
            string c = (model.predict(p, true)[0] >= 0) ? "lightcyan" : "pink";
            test_points.push_back(p);
            test_colors.push_back(c);
        }
    }
    return 0;
}
