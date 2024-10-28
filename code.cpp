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

//HOLIS

// Aliases
template <typename T>
using vec = std::vector<T>;
using double_c = std::complex<double>; //complex number shortcut
using fourD_grid = vec<vec<vec<vec<double_c>>>>;


// Imaginary constant
double_c I(0.0, 1.0); // I = sqrt(-1)

// Physics parameters
const double fm = 5.067;
const double omega = 10;   // Omega value
const double pi = 3.141592653589793;
const double qF = 1.5 / fm;
const double_c Omega = (1.0 - I)/2.0 * std::sqrt(qF / omega);  // Omega * L
const double L = 2.0;
const double z = 0.5; // splitting fraction

// Differential equation parameters
const int N = 21;          // Grid size for u and v
const double dt = 0.001;     // Time step
const double dx1 = L/(N-1);      // Spatial step for u and v
const double dx0 = dx1;
const double max_time = 1; // Number of time steps
const double origin = (N+1)/2;

//adaptative space step 
double dx(int i){
    double d;
    if (i == origin) {
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



// Using u = (u1, u2) and v = (v1, v2)
vec<double> generateVector() { //N entrances list with positions (steps of dx)
    vec<double> vect(N);
    vect[0] = -L/2;
    // Fill the vector with values 0, dx, ..., L - dx
    for (int i = 1; i < N; ++i) {
        vect[i] = vect[i-1] + dx(i);
    }

    return vect;
}

//Vector with spatial coordinates
vec<double> V = generateVector();


// Grid initialization for F(u1, u2, v1, v2, t)
fourD_grid generate_zeros_grid(int N_entries){ //4D vector

    //Define 4D grid
    fourD_grid grid(N, vec<vec<vec<double_c>>> (N_entries, vec<vec<double_c>>(N_entries, vec<double_c>(N_entries, 0.0))));

    //Set initial conditions
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int l = 0; l < N; ++l) {
                    grid[i][j][k][l] = 0.0;
                }
            }
        }
    }

    return grid;
}


//Compute laplacian term in Schrodinger equation du^2 - dv^2
double_c laplacian_term(const fourD_grid& F, 
                         int i, int j, int k, int l) {

    double_c F_cont_i, F_cont_j, F_cont_k, F_cont_l;

    if (i == 0){
        F_cont_i = (F[i+1][j][k][l] - F[i][j][k][l]) / (dx1 * dx1);
    }
    else{
        F_cont_i = (F[i+1][j][k][l] - 2.0 * F[i][j][k][l] + F[i-1][j][k][l]) / (dx(i) * dx(i));
    }

    if (j == 0){
        F_cont_j = (F[i][j+1][k][l] - F[i][j][k][l]) / (dx1 * dx1);
    }
    else{
        F_cont_j = (F[i][j+1][k][l] - 2.0 * F[i][j][k][l] + F[i][j-1][k][l]) / (dx(j) * dx(j));
    }

    if (k == 0){
        F_cont_k = (F[i][j][k+1][l] - F[i][j][k][l]) / (dx1 * dx1);
    }
    else{
        F_cont_k = (F[i][j][k+1][l] - 2.0 * F[i][j][k][l] + F[i][j][k-1][l]) / (dx(k) * dx(k));
    }

    if (l == 0){
        F_cont_l = (F[i][j][k][l+1] - F[i][j][k][l]) / (dx1 * dx1);
    }
    else{
        F_cont_l = (F[i][j][k][l+1] - 2.0 * F[i][j][k][l] + F[i][j][k][l-1]) / (dx(l) * dx(l));
    }



    return F_cont_i + F_cont_j - F_cont_k - F_cont_l;
}

//Compute the dirac delta grid in v coordinates
void generate_dirac_delta_v(fourD_grid& domain_delta){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            domain_delta[i][j][origin][origin] = 1.0 / (dx(origin) * dx(origin));
        }
    }
}



//Compute grid of partial derivatives of the dirac delta in v coordinates
void compute_dirac_partials(const fourD_grid& dirac_grid,
                      fourD_grid& grad_v1,
                      fourD_grid& grad_v2) {


    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            for (int z = 1; z < N - 1; ++z) {
                for (int w = 1; w < N - 1; ++w) {

                    // partial in v1-direction
                    grad_v1[i][j][z][w] = (dirac_grid[i][j][z+1][w] - dirac_grid[i][j][z][w]) /(dx(z));
                    // partial in v2-direction
                    grad_v2[i][j][z][w] = (dirac_grid[i][j][z][w+1] - dirac_grid[i][j][z][w])  / (dx(w));
                }
            }
        }
    }
}

fourD_grid grad_delta_v1;
fourD_grid grad_delta_v2;

//right hand side term of the shrodinger equation (dF/dt = rhs, where F = (Fa, Fb))
void compute_rhs(const fourD_grid& Fa, const fourD_grid& Fb,
                double_c (*Maa)(int, int, int, int), double_c (*Mab)(int, int, int, int),
                double_c (*Mba)(int, int, int, int), double_c (*Mbb)(int, int, int, int),
                fourD_grid& rhs_a, fourD_grid& rhs_b) {    
    
    
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            for (int k = 0; k < N-1; ++k) {
                for (int l = 0; l < N-1; ++l) {
                    double_c laplacian_a =  laplacian_term(Fa, i, j, k, l) / (2 * omega);
                    double_c laplacian_b =  laplacian_term(Fb, i, j, k, l) / (2 * omega);

                    double_c matrix_aa_element = Maa(i,j,k,l); 
                    double_c matrix_ab_element = Mab(i,j,k,l); 
                    double_c matrix_ba_element = Mba(i,j,k,l); 
                    double_c matrix_bb_element = Mbb(i,j,k,l); 

                    double_c matrix_term_a = matrix_aa_element * Fa[i][j][k][l] + matrix_ab_element* Fb[i][j][k][l];
                    double_c matrix_term_b = matrix_ba_element * Fa[i][j][k][l] + matrix_bb_element* Fb[i][j][k][l];

                    double_c non_hom_term = I * omega / pi * (V[i] * grad_delta_v1[i][j][k][l] + V[j]* grad_delta_v2[i][j][k][l]) *
                                            std::exp(I * omega * Omega / 2.0 * cot(Omega * L) * (V[i]*V[i] + V[j]*V[j])) 
                                            / ((V[i]*V[i] + V[j]*V[j]));

                    //std::cout << non_hom_term << std::endl;
                    //double_c non_hom_term = 0;
                    rhs_a[i][j][k][l] = I * laplacian_a + matrix_term_a + non_hom_term;
                    rhs_b[i][j][k][l] = I * laplacian_b +  matrix_term_b + non_hom_term;
                }
            }
        }
    }

}



//4th order RK step
void rk4_step(fourD_grid& F_a, fourD_grid& F_b, 
             double_c (*M_aa)(int, int, int, int), double_c (*M_ab)(int, int, int, int), 
              double_c (*M_ba)(int, int, int, int), double_c (*M_bb)(int, int, int, int)) {

    
    //k and RHS fields for a RK4 step
    fourD_grid k1a, k2a, k3a, k4a, k1b, k2b, k3b, k4b;

    fourD_grid RHS_a = generate_zeros_grid(N);
    fourD_grid RHS_b = RHS_a;

    compute_rhs(F_a, F_b, M_aa, M_ab, M_ba, M_bb, RHS_a, RHS_b);

    // Auxiliary fields to compute the RK4 k fields
    auto F_aux_a = F_a;
    auto F_aux_b = F_b;
    

    // Computation of k1
    k1a = RHS_a;
    k1b = RHS_b;

    //std::cout << std::endl << k1a[0][0][0][0] << k1b[0][0][0][0] << std::endl;

    // Computation of k3 
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            for (int k = 0; k < N-1; ++k) {
                for (int l = 0; l < N-1; ++l) {
                    F_aux_a[i][j][k][l] = F_a[i][j][k][l] + 0.5 * dt * k1a[i][j][k][l];
                    F_aux_b[i][j][k][l] = F_b[i][j][k][l] + 0.5 * dt * k1b[i][j][k][l];
                }
            }
        }
    }
    compute_rhs(F_aux_a, F_aux_b, M_aa, M_ab, M_ba, M_bb, RHS_a, RHS_b);

    k2a = RHS_a;
    k2b = RHS_b;

    
    // Computation of k2
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            for (int k = 0; k < N-1; ++k) {
                for (int l = 0; l < N-1; ++l) {
                    F_aux_a[i][j][k][l] = F_a[i][j][k][l] + 0.5 * dt * k2a[i][j][k][l];
                    F_aux_b[i][j][k][l] = F_b[i][j][k][l] + 0.5 * dt * k2b[i][j][k][l];
                }
            }
        }
    }

    compute_rhs(F_aux_a, F_aux_b, M_aa, M_ab, M_ba, M_bb, RHS_a, RHS_b);

    k3a = RHS_a;
    k3b = RHS_b;

    // Compute k4
    for (int i = 0; i < N-1; ++i) {
        for (int j = 0; j < N-1; ++j) {
            for (int k = 0; k < N-1; ++k) {
                for (int l = 0; l < N-1; ++l) {
                    F_aux_a[i][j][k][l] = F_a[i][j][k][l] + dt * k2a[i][j][k][l];
                    F_aux_b[i][j][k][l] = F_b[i][j][k][l] + dt * k2b[i][j][k][l];
                }
            }
        }
    }

    k4a = RHS_a;
    k4b = RHS_b;

    /*std::cout << "k1: " << k1a[4][4][4][4] << std::endl;
    std::cout << "k2: " << k2a[origin][origin][origin][origin] << std::endl;
    std::cout << "k3: " << k3a[origin][origin][origin][origin] << std::endl;
    std::cout << "k4: " << k4a[origin][origin][origin][origin] << std::endl;*/
    // Update F
    for (int i = 0; i < N-1; ++i) {
        for (int j = 0; j < N-1; ++j) {
            for (int k = 0; k < N-1; ++k) {
                for (int l = 0; l < N-1; ++l) {
                    //std::cout << i << j << k << l << (dt / 6.0) * (k1a[i][j][k][l] + 2.0 * k2a[i][j][k][l] + 2.0 * k3a[i][j][k][l] + k4a[i][j][k][l]) << std::endl;
                    F_a[i][j][k][l] += (dt / 6.0) * (k1a[i][j][k][l] + 2.0 * k2a[i][j][k][l] + 2.0 * k3a[i][j][k][l] + k4a[i][j][k][l]);
                    F_b[i][j][k][l] += (dt / 6.0) * (k1a[i][j][k][l] + 2.0 * k2a[i][j][k][l] + 2.0 * k3a[i][j][k][l] + k4a[i][j][k][l]);
                    if (std::real(F_a[i][j][k][l]) != 0){
                        //std::cout << F_a[i][j][k][l] << std::endl;
                    }
                }
            }
        }
    }

}

//Build matrix elements A - large Nc for now
double_c Matrix_aa(int i1, int j1, int k1, int l1){ //u^2 + v^2

        return -qF/4 * (V[i1]*V[i1] + V[j1]*V[j1] + V[k1]*V[k1] + V[l1]*V[l1]);
    }

double_c Matrix_ab(int i1, int j1, int k1, int l1){ // 0

        return 0.0 + I * 0.0;
    }

double_c Matrix_ba(int i1, int j1, int k1, int l1){ //2z(1-z)(u^2 - v^2)

        return -qF/4 * (2*z*(1-z)*((V[i1] - V[k1])*(V[i1] - V[k1]) + (V[j1] - V[l1])*(V[j1] - V[l1])));
    }

double_c Matrix_bb(int i1, int j1, int k1, int l1){ //z^2(1-z)^2(u^2 - v^2)

        return -qF/4 * ((z * z + (1-z)*(1-z)) * ((V[i1] - V[k1])*(V[i1] - V[k1]) + (V[j1] - V[l1])*(V[j1] - V[l1])));
    }


double_c compute_one_fourier_transform(fourD_grid Fa, fourD_grid Fb, double px, double py){

        double_c integral = 0;

        for(int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                for (int l = 0; l < N; ++l){
                //std::cout <0< F_b_sol[i][j][k][l] << std::endl;
                    integral += Fa[i][j][k][l] * std::exp(-I * (px * (V[i] - V[k]) + py * (V[j] - V[l]))) * dx(i) * dx(j) * dx(k) * dx(l);
                //std::cout << i << j << k << l << std::endl;
                    }
                }
            }  
    
        }
        return integral;
    }



int main(int argc, char* argv[]) { 

    std::chrono::steady_clock sc;   
    auto start = sc.now(); 

    fourD_grid dirac_delta = generate_zeros_grid(N);
    grad_delta_v1 = dirac_delta;
    grad_delta_v2 = dirac_delta;
    
    generate_dirac_delta_v(dirac_delta);
    
    compute_dirac_partials(dirac_delta, grad_delta_v1, grad_delta_v2);

    
    fourD_grid F_a_sol = generate_zeros_grid(N);
    fourD_grid F_b_sol = F_a_sol;

    /*for (int i = 0; i < N; ++i){
        std::cout << V[i] << std::endl;
    }*/

    //rk4_step(F_a_sol, F_b_sol, Matrix_aa, Matrix_ab, Matrix_ba, Matrix_bb);


    /*for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            for (int k = 1; k < N-1; ++k) {
                for (int l = 1; l < N-1; ++l) {
                    
                }
            }
        }
    }*/

    


    

    auto end = sc.now(); 
  	auto time_span = static_cast<std::chrono::duration<double>>(end - start);
	std::cout << std::endl << "Total time: " << time_span.count() << " seconds." << std::endl;

    return 0;
}



