#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <sstream>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include "mcnp_random.h"

using namespace std;

void  RN_test_basic(void);
void  RN_init_problem(unsigned long long*,int*);
void  RN_init_particle(unsigned long long*);
double rang(void);

double sum(vector<double> vec) {
    // Sums all elements in a vector
    double result;
    for (double n : vec) {
        result += n;
    }
    return result;
}

vector<double> elMult( vector<double> orig, double constant) {
    // Multiplies each element in a vector by a constant
    const int length = orig.size();
    vector<double> result;
    result.reserve(length);

    for (int i=0; i<length; i++) {
        result.push_back(orig[i]*constant);
    }
    return result;
}

vector<double> cumulate( vector<double> pmf) {
    // Converts a PMF to a CMF (discrete)
    pmf = elMult(pmf, 1/sum(pmf));
    vector<double> cmf;
    
    double total = 0;
    for (int i = 0; i<pmf.size();i++) {
        total += pmf[i];
        cmf.push_back (total);
    }
    return cmf;
}

int sampleCMF( vector<double> cmf) {
    // Samples a CMF, returns index of result (0,1,2...)
    int length = cmf.size();
    double r = rang();
    for (int i=0; i<length; i++) {
        if (r<cmf[i]) {return i; break;}
    }
    return 1000;
}

double sampleTime( double rateNull ) {
    // Samples time to next interaction, given combined interaction rate
    double step = -log(rang())/rateNull;
    return step;
}

double singleGeneration(int m, double p_fiss, double Tcutoff, double rateSpont, vector<double> nuCMF, 
vector<double>spCMF, double rateAbsorption, int maxN) {
    // Gets extinction time for a single generation, where rxRate is {fission, capture, leakage}
    // NOT normalized to tau
    double t=0;
    int n = m;
    double r;
    while (n>0 && t<Tcutoff) {
        t += sampleTime(n*(rateAbsorption)+rateSpont);
        r=rang();
        if (r < rateSpont/(rateSpont+ n*(rateAbsorption))) {
            n+=(sampleCMF(spCMF));
        }else{
            r=rang();
            if (r>p_fiss) {
                n--;
            } else {
                n+= (sampleCMF(nuCMF)-1);
            }
        }
        // cout << n << endl;
        if (n>maxN) {
            return 0;
            break;
        }
    }
    return t;
}

vector<int> numDistGeneration(double rateAbsorption, double rateSpont,double p_fiss, vector<double> mesh,
 vector<double> spCMF, vector<double> nuCMF, int m){
    vector<int> distribution;

    distribution.push_back(m);
    while(distribution.size() < 2*mesh.size()){
        distribution.push_back(0);
    }
    int index = 0;
    double cutoff = mesh.back();
    int size = mesh.size();
    double r;
    double t=0;
    double temp;
    int n = m;
    int fiss=0;
    int prev_n;
    int prev_fiss;
    int power;
    int base;
    // int heat particles = ---------
    double msh_time;
    for (int i=1; i < size; i++){
        while (t<mesh[i] && n>0){
            prev_n = n;
            prev_fiss = fiss;
            t += sampleTime(n*(rateAbsorption)+rateSpont);
            r=rang();
            if (r < rateSpont/(rateSpont+ n*(rateAbsorption))) {
                n+=(sampleCMF(spCMF));
            }else{
                r=rang();
                if (r>p_fiss) {
                    n--;
                } else {
                    // cout << (sampleCMF(nuCMF)-1) << endl;
                    n+= (sampleCMF(nuCMF)-1);
                    fiss++;
                }
            }
        }
        if (prev_fiss != 0){
            power = pow(10, floor(log10(prev_fiss)));
            base = floor(prev_fiss/power);
            prev_fiss = base*power;
            distribution[i+size] = prev_fiss;
        }else{
            distribution[i+size] = prev_fiss;
        }
        if (n > 0) {
            power = pow(10, floor(log10(prev_n)));
            // cout << power << endl;
            base = floor(prev_n/power);
            prev_n = base*power;
            distribution[i] = prev_n;

 
        }else{break;}
    }
    // back-fill n-vector
    int j = size-1;
    while (distribution[j] == 0){j--;}
    for (int i= 0; i < j; i++){
        if (distribution[i] == 0){
            distribution[i] = distribution[i-1];
        }
    }
    // distribution[size]=0;
    // forward-fill fissions
    j = 2*size - 1;
    while (j > size && distribution[j] == 0){j--;}
    if (distribution[j] != 0){
        j++;
        while (j < size*2){
            distribution[j] = distribution[j-1];
            j++;
        }
    }
    
    // for (int i : distribution) {
    //     cout << i << ", ";
    // }

    // cout << "\n";
    
    return distribution;
}

extern "C" int numDist(int m, int Gens, double rC, double rF, double rL, double S, const char* datafile, const char* t_mesh, int rank) {
    // Inputs NOT normalized to tau
    // Random Number Initialization
    int   prnt=1;
    unsigned long long  seed=1234567+rank, zero=0;
    unsigned long long nps;
    RN_init_problem( &seed, &prnt );

    // parse input to get fission multiplicity
    string val;
    vector<double> pmf;
    ifstream file("p_nu_f.txt");
    while (getline(file, val, ' ')){
       pmf.push_back(stod(val)); 
    }
    file.close();
    file.clear();

    vector<double> pmf_spont;
    ifstream qfile("q_nu_s.txt");
    while (getline(qfile, val, ' ')){
       pmf_spont.push_back(stod(val)); 
    }
    qfile.close();
    qfile.clear();

    vector<double> mesh;
    ifstream meshfile;
    meshfile.open(t_mesh);
    while (getline(meshfile, val)){
       mesh.push_back(stod(val)); 
    //    cout << val << endl;
    }
    meshfile.close();
    

    vector<double> cmf = cumulate(pmf);
    vector<double> cmf_spont = cumulate(pmf_spont);
    auto start = chrono::system_clock::now();
    double rateAbs = rF + rC + rL;
    double p_fiss = rF/(rateAbs);
    double tau = 1/(rateAbs);

    int num_meshpoints = mesh.size();

    // cout << "mesh size= "<<num_meshpoints<<endl;

    vector<vector<int>> data;
    vector<vector<int>> inverse_data;
    vector<vector<int>> counts;

    vector<int> buffer;

    for (int i=0; i<Gens; i++) {
        buffer = numDistGeneration(rateAbs, S, p_fiss, mesh, cmf_spont, cmf, m);
        for (int i : buffer){
        }
        data.push_back(buffer);
    }

    for (int i=0; i<2*num_meshpoints; i++){
        buffer = {};
        for (int j=0; j<Gens; j++){
            buffer.push_back(data[j][i]);
        }
        inverse_data.push_back(buffer);
    }

    unordered_set<int> set;
    vector<int> vec_set;
    for (int i = 0; i < Gens; i++) {
        for (int j = 0; j<2*num_meshpoints; j++){
            set.insert(data[i][j]);
        }
    }
    for (int x : set) {
        vec_set.push_back(x);
    }
    sort(vec_set.begin(), vec_set.end());

    counts.push_back(vec_set);

    vector<int> b;

    int int_buff;
    for (int i=0; i < 2*num_meshpoints; i++){
        b={};
        for (int j : vec_set){
            
            buffer = inverse_data[i];
            int_buff = count(buffer.begin(), buffer.end(), j);
            b.push_back(int_buff);
        }
        counts.push_back(b);
    }
    // cout<< endl << endl;

    ofstream myfile;
    myfile.open(datafile);


    for (int j=0; j < 2*num_meshpoints+1; j++){
        for (int k=0; k < vec_set.size(); k++){
            if (k!= vec_set.size()-1){
                myfile << counts[j][k]<<", ";
            }else{
                myfile << counts[j][k];
            }
            
        }
        // cout << endl;
        myfile << endl;
    }

    myfile.close();



    

    
    return 0;
    
}

extern "C" int xpdf(int m, int Gens, double rC, double rF, double rL, double S, const char* datafile, 
int rank, int maxN) {
    // Inputs NOT normalized to tau
    // Random Number Initialization
    int   prnt=1;
    unsigned long long  seed=1234567+rank, zero=0;
    unsigned long long nps;
    RN_init_problem( &seed, &prnt );

    // parse input to get fission multiplicity
    string val;
    vector<double> pmf;
    ifstream file("p_nu_f.txt");
    while (getline(file, val, ' ')){
       pmf.push_back(stod(val)); 
    }

    vector<double> pmf_spont;
    ifstream qfile("q_nu_s.txt");
    while (getline(qfile, val, ' ')){
       pmf_spont.push_back(stod(val)); 
    }
    vector<double> cmf = cumulate(pmf);
    vector<double> cmf_spont = cumulate(pmf_spont);
    auto start = chrono::system_clock::now();
    double rateAbs = rF + rC + rL;
    double tau = 1/(rateAbs);
    ofstream myfile;
    myfile.open(datafile);

    double maxT = 30000*tau;
    double p_fiss = rF/rateAbs;

    for (int i=0; i<Gens-1; i++) {
        myfile << (singleGeneration(m, p_fiss, maxT, S, cmf, cmf_spont, rateAbs, maxN)) << ",";
    }
    
    myfile << (singleGeneration(m, p_fiss, maxT, S, cmf, cmf_spont, rateAbs, maxN));
    myfile.close();
    auto end = chrono::system_clock::now();

    chrono::duration<double> elapsed_seconds = end-start;

    return 1;
}

int main() {
    return 0;
}