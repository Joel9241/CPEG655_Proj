#include "JacobiRWD.cuh"

int main(){
	float* a = initMat2D(true, true);
	float* b = initMat1D(true, true);
	float* x = initX();
	clock_t begin = clock();
	jacobiMethod(a, b, x);
	clock_t end = clock();
	printf("Time elapsed is %f\n", (double) (end - begin) / CLOCKS_PER_SEC);
	return 0;
}
