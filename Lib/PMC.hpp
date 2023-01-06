#include <vector>
#include <random>
#include <ctime>

using namespace std;

#pragma once

class PMC
{
private:
    int L;
    vector<int> d;
    vector<vector<int>> X;
    vector<vector<int>> deltas;
    vector<vector<vector<int>>> W;

public:
    ~PMC();
    PMC();
    PMC(vector<int> npl);

    void propagate(vector<float> inputs, bool isClassification);
    vector<float> predict(vector<float> inputs, bool isClassification);
    void train(vector<vector<float>> xTrain, vector<vector<float>> yTrain, bool isClassification, float alpha = 0.01, int nb_iter = 10000);
};
