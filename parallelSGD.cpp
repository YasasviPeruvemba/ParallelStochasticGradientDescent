#include <iostream>
#include <fstream>
#include <vector>
#include "serialgrad.h"

using namespace std;
using namespace std::chrono;

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
    auto start = high_resolution_clock::now();
    vector<double> theta = hyp.gradientDescent(5);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"Time in serial : "<<duration.count()<<endl;

    cout << endl;
    for (size_t i = 0; i < theta.size(); i++)
        cout << "th" << i << " = " << theta[i] << " ";
    cout << endl;

    vector <double> x(N);

    for(int i=0;i<theta.size()-1;i++){
        cin >> x[i];
    }

    double pred=theta[0];
    // vector<double> pred(theta.size(), 0.0);
    // pred[0] = theta[0];
    // #pragma omp parallel for
    for(int i=0;i<theta.size()-1;i++){
        pred += x[i]*theta[i+1];
    }
    
    cout<<"Prediction : "<<pred<<endl<<endl;

    return 0;
}