#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

float* initX();
float* initMat2D(bool init, bool host);
float* initMat2DHelper(bool init, bool host, int size);
float* initMat2DHelper(bool init, bool host, int size);
float* initMat1D(bool init, bool host);
float* initMat1DHelper(bool init, bool host, int size);
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
__global__ void addMats2D(float* a, float* b, float* c);
__global__ void addMats2DTB(float* a, float* b, float* c, int lN, int lNT, int lNB);
__global__ void addMats1D(float* a, float* b, float* c);
__global__ void addMats1DTB(float* a, float* b, float* c, int lN, int lNT, int lNB);
__device__ void addMatsHelper(float* a, float* b, float* c, int lN, int lNT, int lNB);
__global__ void subMats2D(float* a, float* b, float* c);
__global__ void subMats2DTB(float* a, float* b, float* c, int lN, int lNT, int lNB);
__global__ void subMats1D(float* a, float* b, float* c);
__global__ void subMats1DTB(float* a, float* b, float* c, int lN, int lNT, int lNB);
__device__ void subMatsHelper(float* a, float* b, float* c, int lN, int lNT, int lNB);
__host__ void jacobiMethod(float* a, float* b, float* x);
__host__ void jacobiMethodTB(float* a, float* b, float* x, int lN, int lNT, int lNB, int lNK);
void dluDecomp(float* a, float* dinv, float* l, float* u);
void dluDecompTB(float* a, float* dinv, float* l, float* u, int lN, int lNT, int lNB);
