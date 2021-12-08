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
void multiplyMats2D(Mat2D* a, Mat2D* b, Mat2D* c);
void multiplyMats2D1D(Mat2D* a, Mat1D* b, Mat1D* c);
void addMats2D(Mat2D* a, Mat2D* b, Mat2D* c);
void subMats2D(Mat2D* a, Mat2D* b, Mat2D* c);
void addMats1D(Mat1D* a, Mat1D* b, Mat1D* c);
void subMats1D(Mat1D* a, Mat1D* b, Mat1D* c);
Mat1D* jacobiMethod(Mat2D* a, Mat1D* b, Mat1D* x);
Mat1D* jacobiIterate(Mat2D* dinv, Mat2D* l, Mat2D* u, Mat1D* b, Mat1D* x);
void dluDecomp(Mat2D* a, Mat2D* dinv, Mat2D* l, Mat2D* u);
