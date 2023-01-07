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
        W.push_back(vector<vector<int>>());
        if (l == 0)
            continue;
        for (int i = 0; i < d[l - 1] + 1; i++)
        {
            W[l].push_back(vector<int>());
            for (int j = 0; j < d[l] + 1; j++)
            {
                uniform_real_distribution<double> distribution(-1, 1);
                unsigned seed = time(nullptr);
                default_random_engine generator(seed);
                float r = distribution(generator);
                W[l][i].push_back(j == 0 ? 1 : r);
            }
        }
    }

    // initialisation des X et des deltas:
    for (int l = 0; l < d.size(); l++)
    {
        X.push_back(vector<int>());
        deltas.push_back(vector<int>());
        for (int j = 0; j < d[l] + 1; j++)
        {
            deltas[l].push_back(0);
            X[l].push_back(j == 0 ? 1 : 0);
        }
    }
}

void PMC::propagate(vector<float> inputs, bool isClassification)
{
    for (int j = 1; j < d[0] + 1; j++)
        X[0][j] = inputs[j - 1];

    for (int l = 1; l < d.size(); l++)
    {
        for (int j = 1; j < d[l] + 1; j++)
        {
            int total = 0;
            for (int i = 0; i < d[l - 1] + 1; i++)
                total += W[l][i][j] * X[l - 1][i];

            X[l][j] = total;
            if (isClassification || l < L)
                X[l][j] = tanh(total);
        }
    }
}

vector<float> PMC::predict(vector<float> inputs, bool isClassification)
{
    propagate(inputs, isClassification);
    return vector<float>(X[L].begin() + 1, X[L].end());
}

void PMC::train(vector<vector<float>> xTrain, vector<vector<float>> yTrain, bool isClassification, float alpha, int nbIter)
{
    uniform_int_distribution<int> distribution(0, xTrain.size() - 1);
    mt19937 generator;

    for (int it = 0; it < nbIter; ++it)
    {
        int k = distribution(generator);
        vector<float> Xk = xTrain[k];
        vector<float> Yk = yTrain[k];

        propagate(Xk, isClassification);
        for (int j = 1; j <= d[L]; ++j)
        {
            deltas[L][j] = X[L][j] - Yk[j - 1];
            if (isClassification)
                deltas[L][j] = deltas[L][j] * (1 - X[L][j] * X[L][j]);
        }

        for (int l = L - 1; l >= 1; --l)
        {
            for (int i = 1; i <= d[l - 1]; ++i)
            {
                double total = 0.0;
                for (int j = 1; j <= d[l]; ++j)
                    total += W[l][i][j] * deltas[l][j];
                deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * total;
            }
        }

        for (int l = 1; l < d.size(); ++l)
        {
            for (int i = 0; i <= d[l - 1]; ++i)
            {
                for (int j = 1; j <= d[l]; ++j)
                    W[l][i][j] += -alpha * X[l - 1][i] * deltas[l][j];
            }
        }
    }
}
