#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

float* initMat2D(bool init, bool host);
float* initMat2DHelper(bool init, bool host, int size);
float* initMat1D(bool init, bool host);
float* initMat1DHelper(bool init, bool host, int size);
void printMat2D(float* mat);
void printMat2DHelper(float* mat, int size);
void printMat1D(float* mat);
void printMat1DHelper(float* mat, int size);
__device__ void multiplyMats2DHelper(float* a, float* b, float* c, int lN, int lNT, int lNB);
__global__ void multiplyMats2D(float* a, float* b, float* c);
__global__ void multiplyMats2DTB(float* a, float* b, float* c, int lN, int lNT, int lNB);
__device__ void multiplyMats2D1DHelper(float* a, float* b, float* c, int lN, int lNT, int lNB);
__global__ void multiplyMats2D1D(float* a, float* b, float* c);
__global__ void multiplyMats2D1DTB(float* a, float* b, float* c, int lN, int lNT, int lNB);
void addMats2D(float* a, float* b, float* c);
void subMats2D(float* a, float* b, float* c);
void addMats1D(float* a, float* b, float* c);
void subMats1D(float* a, float* b, float* c);
float* jacobiMethod(float* a, float* b, float* x);
float* jacobiIterate(float* dinv, float* l, float* u, float* b, float* x);
void dluDecomp(float* a, float* dinv, float* l, float* u);
