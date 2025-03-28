//fourdgrid.h
#ifndef FOURDGRID_H  
#define FOURDGRID_H


#include <iostream>
#include <map>
#include <stdio.h>
#include <complex>
#include <vector>
#include <stdlib.h>

using double_c = std::complex<double>;

class FourDGrid_gqq{
    private:
        int Nsig, Nux, Nuy, Nvx, Nvy;
        double fm = 5.067;
        double L_grid;
        double E, qhat, z, theta, omega; 
        double_c Omega;
        std::vector<std::vector<std::vector<std::vector<std::vector<double_c>>>>> grid;
    
    public:
        //Grid initializer
        ;

        void set_parameters(double E, double qhat, double z);

        FourDGrid_gqq(double L_grid, int Nux, int Nuy, int Nvx, int Nvy);
              
        //Set value of the grid at a certain point
        void setValue(double_c val, int sig, int i1, int i2, int j1, int j2);

        //Get the value of the grid at some point
        double_c getValue(int sig, int i1, int i2, int j1, int j2);

        //Get the value of the grid at an actual coordinate (the closest to the chosen one)
        double_c getValue_coor(int sig, double ux, double uy, double vx, double vy);

        //define linear combination of grids
        //template <typename... FourDGrids_gqq>
        //static FourDGrid_gqq  linearCombination(double_c weight, const FourDGrid_gqq& first, FourDGrids_gqq... rest);

        double_c getPotential(int sig, int sigp, int i1, int i2, int j1, int j2);

};

#endif