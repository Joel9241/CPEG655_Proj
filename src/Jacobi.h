#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

float* initX();
float* initMat1D(bool init);
float* initMat1DHelper(bool init, int lN);
float* initMat2D(bool init);
float* initMat2DHelper(bool init, int lN);
void printMat1D(float* mat);
void printMat1DHelper(float* mat, int lN);
void printMat2D(float* mat);
void printMat2DHelper(float* mat, int lN);
float* multiplyMats2D(float* a, float* b);
float* multiplyMats2DHelper(float* a, float* b, int lN);
float* multiplyMats2D1D(float* a, float* b);
float* multiplyMats2D1DHelper(float* a, float* b, int lN);
float* addMats2D(float* a, float* b);
float* addMats2DHelper(float* a, float* b, int lN);
float* subMats2D(float* a, float* b);
float* subMats2DHelper(float* a, float* b, int lN);
float* addMats1D(float* a, float* b);
float* addMats1DHelper(float* a, float* b, int lN);
float* subMats1D(float* a, float* b);
float* subMats1DHelper(float* a, float* b, int lN);
float* jacobiMethod(float* a, float* b, float* x);
float* jacobiMethodHelper(float* a, float* b, float* x, int lN);
float* jacobiIterate(float* dinv, float* l, float* u, float* b, float* x);
float* jacobiIterateHelper(float* dinv, float* l, float* u, float* b, float* x, int lN);
void dluDecomp(float* a, float* dinv, float* l, float* u);
void dluDecompHelper(float* a, float* dinv, float* l, float* u, int lN);
