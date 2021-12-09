#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

float* initMat2D(bool init, bool host);
float* initMat2DHelper(bool init, bool host, int size);
float* initMat1D(bool init, bool host);
float* initMat1DHelper(bool init, bool host, int size);
void printMat2D(float* mat);
void printMat1D(float* mat);
__global__ void multiplyMats2D(float* a, float* b, float* c);
__global__ void multiplyMats2D1D(float* a, float* b, float* c);
void addMats2D(float* a, float* b, float* c);
void subMats2D(float* a, float* b, float* c);
void addMats1D(float* a, float* b, float* c);
void subMats1D(float* a, float* b, float* c);
float* jacobiMethod(float* a, float* b, float* x);
float* jacobiIterate(float* dinv, float* l, float* u, float* b, float* x);
void dluDecomp(float* a, float* dinv, float* l, float* u);
