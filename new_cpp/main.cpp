#include <iostream>
#include "fourdgrid.h"

int main(){
    std::cout << "1" << std::endl;
    FourDGrid_gqq gridtest(2.0, 4, 4, 4, 4);
    std::cout << "2" << std::endl;
    gridtest.setValue(12, 0, 2, 2, 2, 2);
    std::cout << "3" << std::endl;
    std::cout << gridtest.getValue(0,2,2,2,2) << std::endl;

    return 0;
}