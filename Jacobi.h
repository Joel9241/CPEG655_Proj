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
void printMat2D(Mat2D* mat);
Mat2D* multiplyMats2D(Mat2D* a, Mat2D* b);
void dluDecomp(Mat2D* a, Mat2D* dinv, Mat2D* l, Mat2D* u);
