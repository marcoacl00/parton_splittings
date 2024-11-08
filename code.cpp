//GENERAL C++ LIBRARIES
#include <iostream>
#include <iomanip>
#include <chrono>
#include <regex>
#include <map>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <complex>
#include <utility>  
#include <cmath> // for std::isnan

// ROOT
/* #include "TROOT.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "TLeaf.h" // include TLeaf header file
#include "TVectorT.h" 
#include <ROOT/RDataFrame.hxx>*/

// Aliases
template <typename T>
using vec = std::vector<T>;
using double_c = std::complex<double>; //complex number shortcut
using fourD_grid = vec<vec<vec<vec<double_c>>>>;


// Imaginary constant
double_c I(0.0, 1.0); // I = sqrt(-1)
const double fm = 5.067; //1 fm = 5.067 GeV^-1 (1 GeV = 5.067 fm^-1)

// Physics parameters
const double E = 10 * fm;
const double z = 0.5; // splitting fraction
const double omega = E*z*(1-z);   
const double pi = 3.141592653589793;
const double qF = 1.5 * fm * fm;
const double_c Omega = (1.0 - I)/2.0 * std::sqrt(qF / omega);  
const double L = 2.0  ; //  L in fm
const double theta = 0.5;
const int Nc = 3;
const double CF = (Nc * Nc - 1)/(2*Nc);

// Differential equation parameters
const int N = 32;          // Grid size for u and v;
double dt;
const double origin_dx_factor = 1.0;     // Time step
const double dx1 = 2*L / (N-3 + 2/origin_dx_factor);      // Spatial step for u and v
const double dx0 = dx1/origin_dx_factor;
const double max_time = 1; // Number of time steps
const double origin = N/2-1;
const double prec = dt/10;


//adaptative space step 
double dx(int i){ //dx on the right 
    double d;
    if (i == origin || i == origin - 1) {
         d = dx0;
    } 
    else{
        d = dx1;
    }
    return d;
}


// Cotangent function
double_c cot(double_c x) {
    return 1.0 / tan(x);
}

vec<double_c> generate_zeros_Vector() { //N entrances list with positions (steps of dx)
    vec<double_c> vect(N);
    // Fill the vector with values 0, dx, ..., L - dx
    for (int i = 0; i < N; ++i) {
        vect[i] = 0.0;
    };
    return vect;
}

// Using u = (u1, u2) and v = (v1, v2)
vec<double> generateVector() { //N entrances list with positions (steps of dx)
    vec<double> vect(N);
    vect[0] = -L;
    // Fill the vector with values 0, dx, ..., L - dx
    for (int i = 0; i < N-1; ++i) {
        vect[i+1] = vect[i] + dx(i);
    };
    return vect;
}

//Vector with spatial coordinates
vec<double> V = generateVector();


// Grid initialization for F(u1, u2, v1, v2, t)
fourD_grid generate_zeros_grid(int N_entries){ //4D vector

    //Define 4D grid
    fourD_grid grid(N, vec<vec<vec<double_c>>> (N_entries, vec<vec<double_c>>(N_entries, vec<double_c>(N_entries, 0.0))));

    return grid;
}


vec<double_c> delta;
vec<double_c> d_delta;

//right hand side term of the shrodinger equation (dF/dt = rhs, where F = (Fa, Fb))
void compute_rhs(const fourD_grid& Fa, const fourD_grid& Fb,
                double_c (*Maa)(int, int, int, int), double_c (*Mab)(int, int, int, int),
                double_c (*Mba)(int, int, int, int), double_c (*Mbb)(int, int, int, int),
                fourD_grid& rhs_a, fourD_grid& rhs_b, double T) {    
    
    
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            for (int k = 1; k < N-1; ++k) {
                for (int l = 1; l < N-1; ++l) {

                    double_c Faijkl = Fa[i][j][k][l];
                    double_c Faplus1 = Fa[i+1][j][k][l] + Fa[i][j+1][k][l] - Fa[i][j][k+1][l] - Fa[i][j][k][l+1];
                    double_c Faminus1 = Fa[i-1][j][k][l] + Fa[i][j-1][k][l] - Fa[i][j][k-1][l] - Fa[i][j][k][l-1];
                    
                    double_c Fbijkl = Fb[i][j][k][l];
                    double_c Fbplus1 = Fb[i+1][j][k][l] + Fb[i][j+1][k][l] - Fb[i][j][k+1][l] - Fb[i][j][k][l+1];
                    double_c Fbminus1 = Fb[i-1][j][k][l] + Fb[i][j-1][k][l] - Fb[i][j][k-1][l] - Fb[i][j][k][l-1];
                    

                    double_c laplacian_a =  I * (Faplus1 + Faminus1) / (2 * omega) / (dx0 * dx0);
                    double_c laplacian_b =  I * (Fbplus1 + Fbminus1) / (2 * omega) / (dx0 * dx0);

                    double_c matrix_aa_element = Maa(i,j,k,l); 
                    double_c matrix_ab_element = Mab(i,j,k,l); 
                    double_c matrix_ba_element = Mba(i,j,k,l); 
                    double_c matrix_bb_element = Mbb(i,j,k,l); 

                    double_c matrix_term_a = matrix_aa_element * Faijkl + matrix_ab_element* Fbijkl;
                    double_c matrix_term_b = matrix_ba_element * Faijkl + matrix_bb_element* Fbijkl;


                    double_c non_hom_term;

                    double_c constant = I * omega / pi;
                    double_c phase_fac = std::exp(0.5 * I * omega * Omega  / tan(Omega * (T)) * (V[i]*V[i] + V[j]*V[j]));

                    non_hom_term = constant * phase_fac * (V[i] * d_delta[k] * delta[l] + V[j] * d_delta[l] * delta[k])/ (V[i]*V[i] + V[j]*V[j]);
            
                    // laplacian_a + matrix_term_a + 
                    // laplacian_b + matrix_term_b +

                    
                    rhs_a[i][j][k][l] = (laplacian_a + matrix_term_a + non_hom_term)  ;
                    rhs_b[i][j][k][l] = (laplacian_b + matrix_term_b + non_hom_term)  ;

                    //std::cout << i << j << k << l << std::endl;



                }
            }
        }
    }

    
}


void rk4_step(fourD_grid& F_a, fourD_grid& F_b, fourD_grid& F_a_aux, fourD_grid& F_b_aux, fourD_grid& RHS_a, fourD_grid& RHS_b, 
fourD_grid& k1_a, fourD_grid& k2_a, fourD_grid& k3_a, fourD_grid& k4_a, fourD_grid& k1_b, fourD_grid& k2_b, fourD_grid& k3_b, fourD_grid& k4_b, 
double_c (*M_aa)(int, int, int, int), double_c (*M_ab)(int, int, int, int), double_c (*M_ba)(int, int, int, int), double_c (*M_bb)(int, int, int, int), double T){

    const double half_dt = dt / 2.0;
    double dt_6 = dt/6.0;

    compute_rhs(F_a, F_b, M_aa, M_ab, M_ba, M_bb, k1_a, k1_b, T); //Compute k1s

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int l = 0; l < N; ++l) {
                    F_a_aux[i][j][k][l] = F_a[i][j][k][l] + half_dt * k1_a[i][j][k][l];
                    F_b_aux[i][j][k][l] = F_b[i][j][k][l] + half_dt * k1_b[i][j][k][l];    
                }
            }
        }
    }

    compute_rhs(F_a_aux, F_b_aux, M_aa, M_ab, M_ba, M_bb, k2_a, k2_b, T); //Compute k2s

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int l = 0; l < N; ++l) {
                    F_a_aux[i][j][k][l] = F_a[i][j][k][l] + half_dt * k2_a[i][j][k][l];
                    F_b_aux[i][j][k][l] = F_b[i][j][k][l] + half_dt * k2_b[i][j][k][l];    
                }
            }
        }
    }

    compute_rhs(F_a_aux, F_b_aux, M_aa, M_ab, M_ba, M_bb, k3_a, k3_b, T); //Compute k3s

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int l = 0; l < N; ++l) {
                    F_a_aux[i][j][k][l] = F_a[i][j][k][l] + dt * k3_a[i][j][k][l];
                    F_b_aux[i][j][k][l] = F_b[i][j][k][l] + dt * k3_b[i][j][k][l];    
                }
            }
        }
    }

    compute_rhs(F_a_aux, F_b_aux, M_aa, M_ab, M_ba, M_bb, k4_a, k4_b, T); //Compute k4s


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int l = 0; l < N; ++l) {
                    F_a[i][j][k][l] += dt_6 * (k2_a[i][j][k][l] + 2.0 * k2_a[i][j][k][l]  + 2.0 * k3_a[i][j][k][l] + k4_a[i][j][k][l]);
                    F_b[i][j][k][l] += dt_6 * (k2_b[i][j][k][l] + 2.0 * k2_b[i][j][k][l]  + 2.0 * k3_b[i][j][k][l] + k4_b[i][j][k][l]);    
                }
            }
        }
    }

}


//Build matrix elements A - finite Nc
double_c Matrix_aa(int i1, int j1, int k1, int l1){ //u^2 + v^2

        return -qF/(4.0 * CF) * (CF * (V[i1]*V[i1] + V[j1]*V[j1] + V[k1]*V[k1] + V[l1]*V[l1]) + 1.0/Nc * (V[i1]*V[k1] + V[j1]*V[l1]));
    }

double_c Matrix_ab(int i1, int j1, int k1, int l1){ // 0

        return -qF/(4.0 * CF) * ( -1.0/Nc * (V[i1]*V[k1] + V[j1]*V[l1]));
    }

double_c Matrix_ba(int i1, int j1, int k1, int l1){ //2z(1-z)(u^2 - v^2)

        return -qF/(4.0 * CF) * (Nc*z*(1-z)*((V[i1] - V[k1])*(V[i1] - V[k1]) + (V[j1] - V[l1])*(V[j1] - V[l1])));
    }

double_c Matrix_bb(int i1, int j1, int k1, int l1){ //z^2(1-z)^2(u^2 - v^2)

        return -qF/(4.0 * CF) * ((CF - Nc*z*(1-z)) * ((V[i1] - V[k1])*(V[i1] - V[k1]) + (V[j1] - V[l1])*(V[j1] - V[l1])));
    }

//Compute F(p)
double_c compute_one_fourier_transform(fourD_grid F_test, double px, double py){

        double_c sum = 0.0;
        for(int i2 = 1; i2 < N; ++i2){
            for (int j2 = 1; j2 < N; ++j2){
                for(int k2 = 1; k2 < N; ++k2){
                    for (int l2 = 1; l2 < N; ++l2){
                            sum +=  F_test[i2][j2][k2][l2] * std::exp( -I * (px * (V[i2] - V[k2]) + py * (V[j2] - V[l2])) );
                        }
                    }
                }  
            }
        
    return sum * dx1 * dx1 * dx1 * dx1;
     
    }



int main(int argc, char* argv[]) { 

    std::chrono::steady_clock sc;   
    auto start = sc.now(); 
    

    for(int i = 0; i < 2; i++){
        std::cout << V[i] << std::endl;
    }

    delta = generate_zeros_Vector();
    d_delta = delta;

    delta[origin] = 1/(2*dx0);
    delta[origin + 1] = 1/(2*dx0); 

    d_delta[origin] = 1/(dx0*dx0);
    d_delta[origin+1] = -1/(dx0*dx0) ;


    fourD_grid F_a_sol = generate_zeros_grid(N);
    fourD_grid F_b_sol = F_a_sol;

    
    fourD_grid Fa_aux = F_a_sol;
    fourD_grid Fb_aux = F_a_sol;

    fourD_grid RHSa = F_a_sol;
    fourD_grid RHSb = F_a_sol;
    fourD_grid k1a = F_a_sol;
    fourD_grid k2a = F_a_sol;
    fourD_grid k3a = F_a_sol;
    fourD_grid k4a = F_a_sol;
    fourD_grid k1b = F_a_sol;
    fourD_grid k2b = F_a_sol;
    fourD_grid k3b = F_a_sol;
    fourD_grid k4b = F_a_sol;


    std::ofstream out_hist("F_time.txt", std::ios::trunc);

    double t0 = 0.0005;
    dt = 0.005;

    for (int t = 0; t < 2000; t++){
        
        double_c FT_b = compute_one_fourier_transform(F_b_sol, 0.0, omega*theta) * theta * theta / 2.0;
        

        std::cout << "Step " << t << ": " << "t = " << t0 + t*dt << " " << std::real(FT_b)<< std::endl;


        out_hist << t*dt << "     " << std::real(FT_b) << std::endl;

        rk4_step(F_a_sol, F_b_sol, Fa_aux, Fb_aux, RHSa, RHSb, k1a, k2a, k3a, k4a, k1b, k2b, k3b, k4b, Matrix_aa, Matrix_ab, Matrix_ba, Matrix_bb, t0 + t*dt);


    }



    

    auto end = sc.now(); 
  	auto time_span = static_cast<std::chrono::duration<double>>(end - start);
	std::cout << std::endl << "Total time: " << time_span.count() << " seconds." << std::endl;

    return 0;
}



