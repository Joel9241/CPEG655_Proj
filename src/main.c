#include "Jacobi.h"

int main(){
	float* a = initMat2D(true);
	float* b = initMat1D(true);
	float* x = initX();
	clock_t begin = clock();
	float* xSol = jacobiMethod(a, b, x);
	clock_t end = clock();
	printf("Time elapsed is %f\n", (double) (end - begin) / CLOCKS_PER_SEC);
	return 0;
}
