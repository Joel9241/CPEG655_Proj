#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Mat1D {
	int size;
	float* mat;
} Mat1D;

typedef struct Mat2D {
	int size;
	float* mat;
} Mat2D;

Mat2D* initMat2D(bool init);
Mat1D* initMat1D(bool init);
void printMat1D(Mat1D* mat);
void printMat2D(Mat2D* mat);
Mat2D* multiplyMats2D(Mat2D* a, Mat2D* b);
Mat1D* multiplyMats2D1D(Mat2D* a, Mat1D* b);
Mat2D* addMats2D(Mat2D* a, Mat2D* b);
Mat2D* subMats2D(Mat2D* a, Mat2D* b);
Mat1D* addMats1D(Mat1D* a, Mat1D* b);
Mat1D* subMats1D(Mat1D* a, Mat1D* b);
Mat1D* jacobiMethod(Mat2D* a, Mat1D* b, Mat1D* x);
Mat1D* jacobiIterate(Mat2D* dinv, Mat2D* l, Mat2D* u, Mat1D* b, Mat1D* x);
void dluDecomp(Mat2D* a, Mat2D* dinv, Mat2D* l, Mat2D* u);
