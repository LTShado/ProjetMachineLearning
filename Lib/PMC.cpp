#include "PMC.hpp"

PMC::~PMC() {}

PMC::PMC() {}

PMC::PMC(vector<int> npl)
{
    L = npl.size() - 1;

    for (int k = 0; k < npl.size(); k++)
        d.push_back(npl[k]);

    // initialisation des W:
    for (int l = 0; l < d.size(); l++)
    {
        // W.push_back();
        if (l == 0)
            continue;
        for (int i = 0; i < d[l - 1] + 1; i++)
        {
            // W[l].push_back();
            for (int j = 0; j < d[l] + 1; j++)
            {
                // W[l][i].push_back(j == 0 ? 1 : 0);
            }
        }
    }

    // initialisation des X et des deltas:
    for (int l = 0; l < d.size(); l++)
    {
        // X.push_back();
        // deltas.push_back();
        for (int j = 0; j < d[l] + 1; j++)
        {
            // deltas[l].push_back(0);
            // X[l].push_back(j == 0 ? 1 : 0);
        }
    }
}

void PMC::propagate(vector<float> inputs, bool isClassification) {}

vector<float> PMC::predict(vector<float> inputs, bool isClassification)
{
    return vector<float>();
}

void PMC::train(vector<vector<float>> xTrain, vector<vector<float>> yTrain, bool isClassification, float alpha, int nbIter) {}
