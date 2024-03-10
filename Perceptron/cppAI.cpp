#include <bits/stdc++.h>
using namespace std;
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}
double dot(vector<double> x,vector<double> w){
    double y = 0;
    int siz=w.size();
    if(siz>x.size())x.resize(siz);
    for(int i=0; i<siz; i++){
        y+=x[i]*w[i];
    }return y;
}
int main()
{
    vector<double> x={0.0,1.0};
    vector<double> w={2.0, 2.0, -2};
    for(double i=0; i<2; i++){
        for(double j=0; j<2; j++){
            cout << i << ' ' << j << ' ' << sigmoid(dot({i,j},w)) << '\n';
        }
    }
}
