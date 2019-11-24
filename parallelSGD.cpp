#include <iostream>
#include <fstream>
#include <vector>
#include "gradient.h"

using namespace std;

ostream& operator<<(ostream& os, TrainingExample& te)
{
    int n = te.getFeatures().size();
    os << "(";
    for (int i = 0; i < n; i++)
        os << te.getFeature(i) << (i != n-1 ? ", " : ""); 
    os << ")";
    return os;
}

int main()
{
    int M = 0, N = 0;
    vector<TrainingExample> ts;
    cin >> M >> N;
    cout << "M = " << M << ", N = " << N << endl;
    for (int i = 0; i < M; i++)
    {
        vector<double> feat(N+1, 1);
        double y = 0;
        for (int j = 0; j < N; j++)
            cin >> feat[j+1];
        cin >> y;

        TrainingExample te(feat, y);
        ts.push_back(te);
    }

    // for (unsigned i = 0; i < ts.size(); i++)
        // cout << "Example " << i << ": " << ts[i] << endl;

    Hypothesis hyp(ts);
    vector<double> theta = hyp.gradientDescent(10);

    cout << endl;
    for (size_t i = 0; i < theta.size(); i++)
        cout << "th" << i << " = " << theta[i] << " ";
    cout << endl;

    vector <int> x(N);

    for(int i=0;i<theta.size()-1;i++){
        cin >> x[i];
    }

    double pred = theta[0];

    for(int i=0;i<theta.size()-1;i++){
        pred += x[i]*theta[i+1];
    }
    
    cout<<"Prediction : "<<pred<<endl<<endl;

    return 0;
}