#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

#include <iostream>

// Include local CUDA header files.
#include "cuda_kernel.cuh"

using namespace std;
using namespace cv;

int main()
{

    std::cout << "CUDA turtorial " << std::endl;
    int arraySize = 256;
    double A[arraySize],B[arraySize], Result[2];
    
    for(int i=0;i<arraySize;i++)
    {
        A[i] = (double)i * 1.0;
       // A[i] = 1.0;
        B[i] = 1.0;
    }
    A[0] = 10000.0;

    kernel(A,B,Result, arraySize);
    std::cout << "Result = " << Result[0] << std::endl;

    int rs =0;
    for(int i=0;i<2;i++)
    {
    // Print out result.
    std::cout << "Result[" << i << "] = " << Result[i] << std::endl;
    rs += Result[i];
    }
    std::cout << "rs = " << rs <<std::endl;
   
}
