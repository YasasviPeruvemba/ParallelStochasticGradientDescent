#ifndef GRADIENT_H
#define GRADIENT_H

#include <bits/stdc++.h>
#include <iostream>
#include<fstream>
#include <cmath>
#include <omp.h>

using std::vector;
using std::cout;
using std::endl;
using std::random_shuffle;
using std::ofstream;
using namespace std;
fstream file("loss.txt", ios::out);
template<typename T>
T efficientSum(vector<T>& v){

    int size = v.size();

    int miniSize = log2(size);
    
    int nsets = ceil(1.0*size/miniSize);
    
    vector<T> sum(pow(2, ceil((log2(nsets)))), 0);

    #pragma omp parallel for                            // divide into log(n) parts and take sequential sum. Work = log(n), Time = log(n).
    for(int i=0; i<nsets; ++i){
        T sm = 0;
        int lim = (size < (i+1)*miniSize)?size:(i+1)*miniSize;
        for(int j=miniSize*i; j<lim; ++j){
            sm += v[j];
        }
        sum[i] = sm;
    }

    int steps = log2(sum.size());
    int offset = 2;
    nsets = sum.size();
    for(int h=steps; h>=0; --h){
        int limit = nsets/offset;
        #pragma omp parallel for
        for(int i=0; i < limit; ++i){
            sum[i] = sum[i] + sum[i + nsets/offset];
        }
        offset = offset*2;
    }
    return sum[0];

}


class TrainingExample
{
    private:
        vector<int> features;
        int target;
    public:
        TrainingExample(vector<int>& feat, int tar)
        {
            features = feat;
            target = tar;
        }
        int getFeature(int i) { return features.at(i); }
        vector<int>& getFeatures() { return features; }
        int getTarget() { return target; }
};

class Hypothesis
{
    private:
        vector<double> theta;
        vector<TrainingExample> ts;
        unsigned mExamples, nFeatures;

        double H(vector<double>& ntheta, vector<int>& features)
        {
            //cout << "H(x) = ";
            vector<double> sum(nFeatures, 0.0);
            #pragma omp parallel for
            for (unsigned i = 0; i < nFeatures; i++)
            {
                //cout << ntheta[i] << "*" << features[i] << " ";
                sum[i] = ntheta[i]*features[i];
            }
            //cout << " = " << sum << endl;
            return efficientSum(sum);
        }

        double J()
        {
            vector<double> sum(mExamples, 0.0);
            #pragma omp parallel for
            for (unsigned i = 0; i < mExamples; i++)
            {
                double diff = H(theta, ts[i].getFeatures()) - ts[i].getTarget();
                sum[i] = diff*diff;
            }
            return efficientSum(sum)/2.0;
        }

    public:
        Hypothesis(vector<TrainingExample>& examples)
        {
            nFeatures = 0;
            ts = examples;
            mExamples = examples.size();
            if (mExamples > 0)
                nFeatures = examples[0].getFeatures().size();
            theta = vector<double>(nFeatures, 2000.0);
            cout<<"Features : "<<nFeatures<<endl;
        }

        vector<double> gradientDescent()
        {
            
            const double alpha = 0.0000001;
            const double eps   = 0.00001;
            const int batchSize = 1;
            const int numBatches = mExamples/batchSize;
            cout<<"Batches : "<<numBatches<<" BatchSize = "<<batchSize<<endl;
            cout<<"Eg : "<<ts.size()<<endl;
            bool converge = false;
            int debug = 0;
            int cnt=0;
            while(!converge)
            {
                cnt++;
                vector<double> newTheta = theta;
                double cumulGrad = 0.0;
                double cumulLoss = 0.0;

                //Shuffle Data
                random_shuffle(ts.begin(),ts.end());
                // cout << "J(theta) = " << J() << endl << endl;
                
                file << std::fixed << std::setprecision(20) << J() << endl;
                for (unsigned i = 0; i < numBatches; i++)
                {
                    vector <double> grad(mExamples,0.0);
                    vector <double> loss(mExamples,0.0);
                    double featureSum = 0.0;
                    cumulGrad = 0.0;
                    cumulLoss = 0.0;
                    //cout << "Using example" << i << endl;

                    //Parallelise
                    #pragma omp parallel for
                    for(int j = i*batchSize; j < (i+1)*batchSize; j++){
                        // cout<<"j : "<<j<<endl;
                        TrainingExample ex = ts[j];
                        double HH = H(newTheta, ex.getFeatures());
                        double diff = (ex.getTarget() - HH)*alpha;
                        loss[j] = HH;
                        for(int k = 0; k < nFeatures; k++){
                            cumulGrad += ex.getFeature(k);
                        }
                        featureSum = efficientSum(ex.getFeatures());
                        // cumulGrad*=diff;

                        grad[j] = diff * (featureSum);
                        // cout << ex.getTarget() << "(T) - " << HH <<"(H) = " << diff <<endl;
                    }

                    cumulGrad = efficientSum(grad);
                    cumulLoss = efficientSum(loss);
                    cumulGrad/=batchSize;
                    cumulLoss/=batchSize;
                    // cout<<"cumulGrad : "<<cumulGrad<<endl;
                    //Parallelise
                    #pragma omp parallel for
                    for (unsigned j = 0; j < nFeatures; j++){
                        newTheta[j] += cumulGrad;
                    }

                    // for (int k = 0; k < newTheta.size(); k++)
                    //     cout << "newTh" << k << " = " << newTheta[k] << " ";
                    // cout << endl;

                }

                // cout<<"Loss : "<<fabs(cumulLoss)<<endl<<endl;

                // if(fabs(cumulLoss) < tol){                    
                //     cout<<"Loss is tolerable!\n";
                //     converge = false;
                //     break;
                // }

                for (unsigned i = 0; i < theta.size(); i++){
                    cout<<"Theta "<<i<<" change : "<<fabs(theta[i]-newTheta[i])<<endl;
                    if(fabs(theta[i]-newTheta[i]) < eps){
                        converge = true;
                    }
                    else{
                        converge = false;
                        theta = newTheta;
                        break; 
                    }
                }
            }
            return theta;
        }
};

#endif
