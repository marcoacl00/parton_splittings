#include "fourdgrid.h"

FourDGrid_gqq::FourDGrid_gqq(double L_grid, int Nux, int Nuy, int Nvx, int Nvy) : Nux(Nux), Nuy(Nuy), Nvx(Nvx), Nvy(Nvy){
    grid.resize(2, std::vector<std::vector<std::vector<std::vector<double_c>>>>(
        Nux, std::vector<std::vector<std::vector<double_c>>>(
            Nuy, std::vector<std::vector<double_c>>(
                Nvx, std::vector<double_c>(Nvy, 0.0)
                )
            )
        )
    );
}

void FourDGrid_gqq::setValue(double_c val, int sig, int i1, int i2, int j1, int j2){
    grid[sig][i1][i2][j1][j2] = val;
}

double_c FourDGrid_gqq::getValue(int sig, int i1, int i2, int j1, int j2){
    return grid[sig][i1][i2][j1][j2];
}

double_c FourDGrid_gqq::getValue_coor(int sig, double ux, double uy, double vx, double vy){

    std::cout << "Soon to be implemented" << std::endl;

    return 0.0;
}

void set_parameters(double E_, double qhat_, double z_){
    E = E_;

}

double_c getPotential(int sig, int sigp, int i1, int i2, int j1, int j2){

}










