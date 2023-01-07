#include "LinearModel.hpp"

LinearModel::~LinearModel() {}

LinearModel::LinearModel()
{
    array<float, 3> W;
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (float &w : W)
        w = distribution(generator);

    for (int j = 0; j < W.size(); ++j)
        cout << W[j] << endl;

    display();
    trainning();
}

void LinearModel::display()
{
    // Affichage
    vector<array<float, 2>> test_points;
    vector<string> test_colors;
    for (int row = 0; row < 300; ++row)
    {
        for (int col = 0; col < 300; ++col)
        {
            array<float, 2> p{{col / 100.0f, row / 100.0f}};
            vector<float> x{1.0, p[0], p[1]};
            string c = (inner_product(W.begin(), W.end(), x.begin(), 0.0) >= 0) ? "lightcyan" : "pink";
            test_points.push_back(p);
            test_colors.push_back(c);
        }
    }
}

void LinearModel::trainning()
{
    // Trainning
    mt19937 generator;
    uniform_int_distribution<int> distribution(0, points.size() - 1);
    for (int i = 0; i < 10000; ++i)
    {
        int k = distribution(generator);
        int yk = classes[k];
        vector<int> Xk{1, points[k][0], points[k][1]};
        float gXk = (inner_product(W.begin(), W.end(), Xk.begin(), 0.0) >= 0) ? 1.0 : -1.0;
        transform(W.begin(), W.end(), Xk.begin(), W.begin(), [&](float w, float xk)
                  { return w + 0.01 * (yk - gXk) * xk; });
    }
}
